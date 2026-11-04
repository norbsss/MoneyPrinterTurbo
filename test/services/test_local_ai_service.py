"""本机 AI 生图在主程序中的接入：服务客户端、素材来源与任务预检。"""

import base64
import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from app.config import config
from app.models.schema import VideoAspect, VideoParams
from app.services import local_ai, material, state as sm
from app.services import task as task_service
from local_ai import catalog, hardware, models


def _png_bytes(size=(16, 24)):
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def local_settings(tmp_path):
    settings = dict(
        config.app,
        local_ai_models_dir=str(tmp_path / "models"),
        local_ai_runtime_dir=str(tmp_path / "runtime"),
        local_ai_image_model="",
        local_ai_gpu_index="",
        local_ai_url="http://127.0.0.1:7999",
    )
    detected = hardware.Hardware(
        os="Windows", machine="AMD64", ram_gb=64,
        gpus=[hardware.Gpu(0, "big", "nvidia", 16), hardware.Gpu(1, "small", "nvidia", 8)],
        nvidia_driver="610.88",
    )
    with (
        patch.object(config, "app", settings),
        patch.object(local_ai, "detect_hardware", return_value=detected),
    ):
        yield SimpleNamespace(settings=settings, tmp_path=tmp_path)


def _install(model_id, models_dir):
    model = catalog.get(model_id)
    folder = models.model_dir(model, models_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / models.MARKER_NAME).write_text(json.dumps({"repo": model.repo}))


def _install_runtime(runtime_dir):
    venv = runtime_dir / "venv"
    venv.mkdir(parents=True, exist_ok=True)
    (venv / ".mpt-installed").write_text("cu128")


def test_readiness_names_the_missing_step(local_settings):
    assert local_ai.readiness() == (False, "local_ai.runtime_missing")
    _install_runtime(local_settings.tmp_path / "runtime")
    assert local_ai.readiness() == (False, "local_ai.model_missing")
    _install("realvisxl-v5-lightning", local_settings.settings["local_ai_models_dir"])
    assert local_ai.readiness() == (True, "")


def test_image_model_uses_selection_then_recommendation(local_settings):
    assert local_ai.image_model().id == "realvisxl-v5-lightning"
    local_settings.settings["local_ai_image_model"] = "dreamshaper-8"
    assert local_ai.image_model().id == "dreamshaper-8"
    local_settings.settings["local_ai_image_model"] = "removed-model"
    assert local_ai.image_model().id == "realvisxl-v5-lightning"


def test_gpu_index_limits_the_hardware_used_for_fitting(local_settings):
    local_settings.settings["local_ai_gpu_index"] = "1"
    assert [gpu.index for gpu in local_ai.gpu_hardware().gpus] == [1]
    assert catalog.recommend(local_ai.gpu_hardware()).mode == catalog.FIT_OFFLOAD


class _Response:
    def __init__(self, status, body):
        self.status_code = status
        self.ok = status < 400
        self._body = body
        self.text = json.dumps(body)

    def json(self):
        return self._body


class _Session:
    def __init__(self, response, calls):
        self.response = response
        self.calls = calls
        self.trust_env = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "trust_env": self.trust_env})
        return self.response


def _patch_session(response, calls):
    original = local_ai._session

    def make():
        session = original()
        fake = _Session(response, calls)
        fake.trust_env = session.trust_env
        return fake

    return patch.object(local_ai, "_session", make)


def test_generate_image_posts_to_local_server_without_proxies(local_settings):
    calls = []
    png = _png_bytes()
    response = _Response(200, {"data": [{"b64_json": base64.b64encode(png).decode()}]})
    with (
        patch.object(local_ai, "ensure_server", return_value={"ok": True}),
        _patch_session(response, calls),
    ):
        assert local_ai.generate_image("a lighthouse", "1024x1536") == png
    assert calls == [{
        "url": "http://127.0.0.1:7999/v1/images/generations",
        "json": {"model": "realvisxl-v5-lightning", "prompt": "a lighthouse", "n": 1, "size": "1024x1536"},
        "trust_env": False,
    }]


def test_generate_image_reports_server_detail(local_settings):
    response = _Response(409, {"detail": "model 'realvisxl-v5-lightning' is not downloaded"})
    with (
        patch.object(local_ai, "ensure_server", return_value={"ok": True}),
        _patch_session(response, []),
        pytest.raises(local_ai.LocalAIError, match="409.*not downloaded"),
    ):
        local_ai.generate_image("x", "1024x1024")


def test_ensure_server_does_not_start_when_autostart_is_off(local_settings):
    local_settings.settings["local_ai_autostart"] = False
    with (
        patch.object(local_ai, "server_status", return_value=None),
        patch.object(local_ai, "start_server") as start,
        pytest.raises(local_ai.LocalAIError, match="not running"),
    ):
        local_ai.ensure_server()
    start.assert_not_called()


def test_stop_server_ends_the_process_the_server_reports(local_settings):
    # Windows 虚拟环境的 python.exe 是启动器：服务真正的 pid 来自健康检查。
    with (
        patch.object(local_ai, "server_status", return_value={"ok": True, "pid": 4242}),
        patch.object(local_ai.os, "kill") as kill,
    ):
        local_ai.stop_server()
    kill.assert_called_once_with(4242, local_ai.signal.SIGTERM)


def test_generate_images_local_saves_png_as_image_material(local_settings, tmp_path):
    with patch.object(local_ai, "generate_image", return_value=_png_bytes((16, 24))) as generate:
        items = material.generate_images_local(
            "a lighthouse", 5, VideoAspect.portrait, save_dir=str(tmp_path)
        )
    generate.assert_called_once_with("a lighthouse", "1024x1536")
    assert len(items) == 1
    assert items[0].provider == "local_image"
    assert items[0].duration == 5
    assert items[0].source_info["rendition"]["width"] == 16
    assert Image.open(items[0].url).size == (16, 24)


def test_generate_images_local_stops_the_task_when_the_server_fails(local_settings, tmp_path):
    with (
        patch.object(local_ai, "generate_image", side_effect=local_ai.LocalAIError("server exited")),
        pytest.raises(material.LocalImageMaterialError, match="server exited"),
    ):
        material.generate_images_local("x", 5, VideoAspect.portrait, save_dir=str(tmp_path))
    # 失败属于 OpenAIImagePaidResultError，任务编排层已有的处理会终止素材阶段。
    assert issubclass(material.LocalImageMaterialError, material.OpenAIImagePaidResultError)


def test_download_videos_routes_local_image_to_the_local_generator(local_settings, tmp_path):
    with patch.object(material, "_download_videos_openai_image_on_demand", return_value=["clip.mp4"]) as flow:
        result = material.download_videos(
            task_id="local-task",
            search_terms=["a lighthouse"],
            source="local_image",
            video_aspect=VideoAspect.portrait,
            audio_duration=5,
            max_clip_duration=5,
        )
    assert result == ["clip.mp4"]
    options = flow.call_args.kwargs
    assert options["image_generator"] is material.generate_images_local
    assert options["provider"] == "local_image"


def test_task_preflight_fails_before_script_when_local_ai_is_not_ready(local_settings):
    params = VideoParams(video_subject="Local preflight", video_source="local_image")
    with (
        patch.object(task_service.sm, "state", sm.MemoryState()),
        patch.object(task_service, "generate_script") as generate_script,
    ):
        result = task_service.start("local-preflight", params, stop_at="materials")
    assert result["failed_stage"] == "preflight"
    assert "install it in Settings" in result["error"]
    generate_script.assert_not_called()
