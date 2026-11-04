"""Local AI package: hardware parsing, model choice, runtime plan, downloads and the GPU server.

None of these tests need torch, a GPU or the network: the server runs with a
stand-in pipeline, and downloads with a stand-in for huggingface_hub.
"""

import base64
import io
import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from local_ai import catalog, hardware, models, runtime
from local_ai import server as srv


def _hardware(vram=None, ram=32.0, driver="610.88", os_name="Windows", vendor="nvidia"):
    gpus = [] if vram is None else [hardware.Gpu(0, "GPU", vendor, vram)]
    return hardware.Hardware(os=os_name, machine="AMD64", ram_gb=ram, gpus=gpus, nvidia_driver=driver)


# ---------------------------------------------------------------- hardware


def test_parse_nvidia_gpus_reads_every_card_and_the_driver():
    output = (
        "0, NVIDIA GeForce RTX 4070 Ti SUPER, 16376, 14420, 610.88\n"
        "1, NVIDIA GeForce RTX 4060, 8188, 7956, 610.88\n"
        "garbage line\n"
    )
    gpus, driver = hardware.parse_nvidia_gpus(output)
    assert [(gpu.index, gpu.vram_gb) for gpu in gpus] == [(0, 16.0), (1, 8.0)]
    assert driver == "610.88"


@pytest.mark.parametrize(
    "header, expected",
    [
        ("| NVIDIA-SMI 550.54   Driver Version: 550.54   CUDA Version: 12.4 |", (12, 4)),
        ("| NVIDIA-SMI 610.88   KMD Version: 610.88   CUDA UMD Version: 13.3 |", (13, 3)),
        ("no cuda here", None),
    ],
)
def test_parse_cuda_version_handles_old_and_new_driver_headers(header, expected):
    assert hardware.parse_cuda_version(header) == expected


def test_best_gpu_is_the_one_with_most_memory():
    detected = hardware.Hardware(
        os="Windows", machine="AMD64", ram_gb=64,
        gpus=[hardware.Gpu(0, "small", "nvidia", 8), hardware.Gpu(1, "big", "nvidia", 16)],
    )
    assert detected.best_gpu.index == 1
    assert detected.to_dict()["best_gpu_index"] == 1


# ---------------------------------------------------------------- catalog


def test_catalog_entries_are_complete():
    entries = catalog.load()
    assert len({model.id for model in entries}) == len(entries)
    for model in entries:
        assert model.kind == "image"
        assert model.family in ("sdxl", "sd15", "flux")
        assert model.vram_offload_gb <= model.vram_gpu_gb
        assert model.allow_patterns


@pytest.mark.parametrize(
    "vram, ram, expected_id, expected_mode",
    [
        (16, 64, "realvisxl-v5-lightning", catalog.FIT_GPU),
        # Offloading SDXL costs seconds; it still beats SD 1.5 fully on the GPU.
        (8, 32, "realvisxl-v5-lightning", catalog.FIT_OFFLOAD),
        (4, 16, "dreamshaper-8", catalog.FIT_GPU),
        (None, 16, "dreamshaper-8", catalog.FIT_CPU),
        # FLUX would fit with offload here, but it is gated: an open model wins.
        (24, 64, "realvisxl-v5-lightning", catalog.FIT_GPU),
    ],
)
def test_recommend_picks_best_open_model_for_hardware(vram, ram, expected_id, expected_mode):
    choice = catalog.recommend(_hardware(vram, ram))
    assert (choice.model.id, choice.mode) == (expected_id, expected_mode)


def test_rank_puts_gpu_options_before_cpu_and_unusable_last():
    ranked = catalog.rank(_hardware(24, 64))
    assert ranked[0].model.id == "flux1-schnell"  # best quality among GPU options
    assert ranked[0].mode == catalog.FIT_OFFLOAD
    assert catalog.rank(_hardware(None, 4))[-1].mode == catalog.FIT_NONE


def test_recommend_returns_none_when_nothing_fits():
    assert catalog.recommend(_hardware(None, 2)) is None


# ---------------------------------------------------------------- runtime


@pytest.mark.parametrize(
    "driver, expected",
    [
        ("610.88", "cu128"),
        ("575.10", "cu128"),
        ("565.90", "cu126"),
        ("531.00", "cpu"),
    ],
)
def test_torch_index_matches_driver(driver, expected):
    assert runtime.torch_index(_hardware(16, driver=driver)).endswith(f"/{expected}")


def test_torch_index_uses_pypi_on_macos_and_cpu_wheels_without_gpu():
    assert runtime.torch_index(_hardware(None, os_name="Darwin")) is None
    assert runtime.torch_index(_hardware(None, os_name="Linux")).endswith("/cpu")


def test_install_commands_prefer_uv(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime.shutil, "which", lambda name: "/bin/uv" if name == "uv" else None)
    commands = runtime.install_commands(str(tmp_path), _hardware(16))
    assert [command[:2] for command in commands] == [["/bin/uv", "venv"], ["/bin/uv", "pip"], ["/bin/uv", "pip"]]
    assert "--index-url" in commands[1] and commands[1][-1].endswith("/cu128")


def test_install_marks_runtime_only_after_every_command_succeeds(monkeypatch, tmp_path):
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if len(calls) == 2 else 0)

    monkeypatch.setattr(runtime.subprocess, "run", run)
    log = tmp_path / "install.log"
    with pytest.raises(RuntimeError):
        runtime.install(str(tmp_path / "rt"), _hardware(16), str(log))
    assert not runtime.is_installed(str(tmp_path / "rt"))

    calls.clear()
    monkeypatch.setattr(runtime.subprocess, "run", lambda command, **_k: SimpleNamespace(returncode=0))
    (tmp_path / "rt" / "venv").mkdir(parents=True)
    runtime.install(str(tmp_path / "rt"), _hardware(16), str(log))
    assert runtime.is_installed(str(tmp_path / "rt"))
    assert "$ " in log.read_text(encoding="utf-8")


# ---------------------------------------------------------------- downloads


@pytest.fixture
def realvis():
    return catalog.get("realvisxl-v5-lightning")


def test_only_a_folder_with_a_complete_marker_counts_as_installed(tmp_path, realvis):
    folder = models.model_dir(realvis, str(tmp_path))
    folder.mkdir(parents=True)
    (folder / "model_index.json").write_text("{}")
    assert models.installed_path(realvis, str(tmp_path)) is None

    (folder / models.MARKER_NAME).write_text(json.dumps({"repo": "someone/else"}))
    assert models.installed_path(realvis, str(tmp_path)) is None

    (folder / models.MARKER_NAME).write_text(json.dumps({"repo": realvis.repo}))
    assert models.installed_path(realvis, str(tmp_path)) == str(folder)

    models.delete(realvis, str(tmp_path))
    assert not folder.exists()


def test_download_job_reports_progress_and_marks_complete(monkeypatch, tmp_path, realvis):
    import huggingface_hub

    def fake_snapshot(repo, local_dir, **kwargs):
        assert repo == realvis.repo
        assert kwargs["allow_patterns"] == list(realvis.allow_patterns)
        (tmp_path / realvis.id / "unet").mkdir(parents=True, exist_ok=True)
        (tmp_path / realvis.id / "unet" / "weights.bin").write_bytes(b"x" * 100)
        return local_dir

    monkeypatch.setattr(models, "remote_size", lambda model, token=None: 100)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    job = models.DownloadJob(realvis, str(tmp_path))
    job._run()
    assert job.state == models.STATE_DONE, job.error
    assert job.to_dict()["bytes_done"] == 100
    assert models.installed_path(realvis, str(tmp_path))


def test_download_refuses_when_the_disk_is_too_full(monkeypatch, tmp_path, realvis):
    monkeypatch.setattr(models, "remote_size", lambda model, token=None: 50 * 1024**3)
    monkeypatch.setattr(models, "free_disk_gb", lambda path: 20.0)
    job = models.DownloadJob(realvis, str(tmp_path))
    job._run()
    assert job.state == models.STATE_FAILED
    assert "GB is free" in job.error
    assert models.installed_path(realvis, str(tmp_path)) is None


# ---------------------------------------------------------------- server


class FakePipe:
    def __init__(self):
        self.devices = []

    def to(self, device):
        self.devices.append(device)
        return self


@pytest.fixture
def server(monkeypatch, tmp_path, realvis):
    folder = models.model_dir(realvis, str(tmp_path))
    folder.mkdir(parents=True)
    (folder / models.MARKER_NAME).write_text(json.dumps({"repo": realvis.repo}))
    painted, loaded, unloads = [], [], []
    monkeypatch.setattr(srv, "MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(srv, "DEFAULT_MODEL", realvis.id)
    monkeypatch.setattr(srv, "_hardware_for_server", lambda: _hardware(16, 64))
    monkeypatch.setattr(srv, "unload_ollama", lambda: unloads.append(True) or [])
    monkeypatch.setattr(srv.pipelines, "device", lambda: "cuda")
    monkeypatch.setattr(srv.pipelines, "empty_cache", lambda: None)

    def fake_load(model, path, mode):
        loaded.append((model.id, path, mode))
        return FakePipe()

    def fake_paint(pipe, model, prompt, negative, width, height):
        painted.append({"prompt": prompt, "size": (width, height)})
        return Image.new("RGB", (8, 8))

    monkeypatch.setattr(srv.pipelines, "load", fake_load)
    monkeypatch.setattr(srv.pipelines, "paint", fake_paint)
    monkeypatch.setattr(srv, "pipe", None)
    monkeypatch.setattr(srv, "loaded_model", None)
    monkeypatch.setattr(srv, "on_gpu", False)
    with TestClient(srv.app) as client:
        yield SimpleNamespace(client=client, painted=painted, loaded=loaded, unloads=unloads)


def test_server_paints_with_installed_model_on_gpu(server, realvis):
    response = server.client.post(
        "/v1/images/generations", json={"prompt": "a lighthouse", "size": "1024x1536"}
    )
    assert response.status_code == 200
    image = Image.open(io.BytesIO(base64.b64decode(response.json()["data"][0]["b64_json"])))
    assert image.size == (8, 8)
    assert server.loaded == [(realvis.id, str(models.model_dir(realvis, srv.MODELS_DIR)), catalog.FIT_GPU)]
    assert server.painted[0]["size"] == (832, 1216)
    assert server.painted[0]["prompt"].startswith("a lighthouse, ")
    assert server.unloads, "Ollama must give up the GPU before painting"
    health = server.client.get("/health").json()
    assert health["model"] == realvis.id and health["on_gpu"] is True


def test_server_rejects_models_that_are_unknown_or_not_downloaded(server):
    assert server.client.post(
        "/v1/images/generations", json={"prompt": "x", "model": "nope"}
    ).status_code == 404
    assert server.client.post(
        "/v1/images/generations", json={"prompt": "x", "model": "dreamshaper-8"}
    ).status_code == 409
    assert server.client.post("/v1/images/generations", json={"prompt": "  "}).status_code == 400
    assert [entry["id"] for entry in server.client.get("/v1/models").json()["data"]] == [
        "realvisxl-v5-lightning"
    ]


def test_idle_model_moves_back_to_cpu(server, monkeypatch):
    server.client.post("/v1/images/generations", json={"prompt": "a lighthouse"})
    monkeypatch.setattr(srv, "last_used", 0.0)
    assert srv.offload_if_idle() is True
    assert srv.pipe.devices[-1] == "cpu"
    assert srv.on_gpu is False


@pytest.mark.parametrize(
    "family, size, expected",
    [
        ("sdxl", "1024x1536", (832, 1216)),
        ("sdxl", "1536x1024", (1216, 832)),
        ("sd15", "1024x1536", (512, 768)),
        ("flux", None, (768, 1344)),
        ("sdxl", "bad", (768, 1344)),
    ],
)
def test_pick_size_snaps_to_trained_buckets(family, size, expected):
    assert srv.pipelines.pick_size(family, size) == expected
