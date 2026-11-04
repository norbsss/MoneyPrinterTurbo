"""视频库：缩略图、发布状态与弹窗交互。"""

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from app.config import config
from app.models import const
from app.services import material as _material  # noqa: F401  (resolve FFmpeg before patching)
from app.services import state as sm
from app.utils import utils
from webui.components import library

ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"


def _tiny_video(path: Path) -> Path:
    subprocess.run(
        [
            utils.get_ffmpeg_binary(), "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=red:s=64x112:d=1",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True, capture_output=True, timeout=60,
    )
    return path


def test_thumbnail_is_created_once_and_refreshed_for_newer_video(tmp_path):
    video = _tiny_video(tmp_path / "final-1.mp4")
    thumbnail = library.task_thumbnail(str(video))
    assert thumbnail == str(tmp_path / library.THUMBNAIL_NAME)
    assert os.path.getsize(thumbnail) > 0
    with patch.object(library.subprocess, "run") as run:
        assert library.task_thumbnail(str(video)) == thumbnail
    run.assert_not_called()
    future = os.path.getmtime(thumbnail) + 10
    os.utime(video, (future, future))
    with patch.object(library.subprocess, "run", wraps=subprocess.run) as run:
        library.task_thumbnail(str(video))
    run.assert_called()


def test_thumbnail_returns_none_when_ffmpeg_cannot_read_the_file(tmp_path):
    video = tmp_path / "final-1.mp4"
    video.write_bytes(b"not a video")
    assert library.task_thumbnail(str(video)) is None


@pytest.mark.parametrize(
    "task, expected",
    [
        ({}, "library.publish.none"),
        ({"cross_post_state": const.CROSS_POST_STATE_PROCESSING}, "library.publish.pending"),
        ({"cross_post_state": const.CROSS_POST_STATE_FAILED, "cross_post_error": "boom"},
         "library.publish.failed"),
        ({"cross_post_state": const.CROSS_POST_STATE_COMPLETE,
          "cross_post_results": [{"success": True}]}, "library.publish.done"),
        ({"cross_post_state": const.CROSS_POST_STATE_COMPLETE,
          "cross_post_results": [{"success": True, "scheduled": True, "job_id": "j"}]},
         "library.publish.queued"),
    ],
)
def test_publish_status_labels(task, expected):
    with patch.object(library, "tr", lambda key: key):
        label, detail = library.publish_status(task)
    assert label == expected
    assert detail == ("boom" if expected == "library.publish.failed" else "")


def test_scheduled_status_shows_the_local_time():
    task = {
        "cross_post_state": const.CROSS_POST_STATE_COMPLETE,
        "cross_post_results": [{"scheduled": True, "scheduled_date": "2026-11-04T13:00:00Z"}],
    }
    with patch.object(library, "tr", lambda key: "at {date}"):
        label, _ = library.publish_status(task)
    assert label.startswith("at 2026-11-0")


@pytest.fixture
def library_app(tmp_path, monkeypatch):
    tasks_dir = tmp_path / "storage" / "tasks"
    task_dir = tasks_dir / "library-task"
    task_dir.mkdir(parents=True)
    _tiny_video(task_dir / "final-1.mp4")
    (task_dir / "script.json").write_text('{"params": {"video_subject": "Coffee ritual"}}', encoding="utf-8")
    monkeypatch.setattr(utils, "task_dir", lambda *a: str(tasks_dir if not a else tasks_dir / a[0]))
    monkeypatch.setattr(sm.state, "get_all_tasks", lambda *_a, **_k: ([], 0))
    monkeypatch.setattr(
        sm.state, "get_task",
        lambda task_id: {"cross_post_state": const.CROSS_POST_STATE_COMPLETE,
                         "cross_post_results": [{"scheduled": True, "scheduled_date": "2026-11-04T13:00:00Z"}]},
    )
    with patch.object(config, "try_save_config", return_value=True):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "en"
        app.run()
        yield app, task_dir


def test_library_dialog_shows_videos_and_plays_them_inline(library_app):
    app, task_dir = library_app
    app.button(key="open_library_dialog_button").click().run()
    assert not app.exception
    captions = " ".join(item.value for item in app.caption)
    assert "Scheduled for 2026-11-0" in captions
    assert any("Coffee ritual" in item.value for item in app.markdown)
    assert (task_dir / library.THUMBNAIL_NAME).is_file()

    app.button(key="library_play_library_task").click().run()
    assert not app.exception
    assert app.session_state["library_preview_video"] == str(task_dir / "final-1.mp4")
    assert len(app.get("video")) == 1
