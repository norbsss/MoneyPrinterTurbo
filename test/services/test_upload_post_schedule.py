"""Upload-Post 定时发布：加入发布队列或延迟发布，由 Upload-Post 按时执行。"""

import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, mock_open, patch

import pytest

from app.services import task as tm
from app.services import upload_post
from app.services.state import MemoryState
from app.services.upload_post import UploadPostService

_CONFIG = {
    "upload_post_enabled": True,
    "upload_post_api_key": "test-key",
    "upload_post_username": "testuser",
}
_NOW = datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc)


def _service(**settings):
    with patch("app.services.upload_post.config.app", {**_CONFIG, **settings}):
        return UploadPostService().schedule_request(now=_NOW)


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({}, None),
        ({"upload_post_schedule": "unknown"}, None),
        ({"upload_post_schedule": "queue"}, {"mode": "queue"}),
        (
            {"upload_post_schedule": "delay", "upload_post_schedule_delay_hours": 3},
            {"mode": "delay", "scheduled_date": "2026-11-04T13:00:00Z"},
        ),
        # Upload-Post 只接受未来 365 天内的时间，也拒绝已经过去的时间。
        (
            {"upload_post_schedule": "delay", "upload_post_schedule_delay_hours": 10**6},
            {"mode": "delay", "scheduled_date": "2027-11-04T10:00:00Z"},
        ),
        (
            {"upload_post_schedule": "delay", "upload_post_schedule_delay_hours": "bad"},
            {"mode": "delay", "scheduled_date": "2026-11-04T12:00:00Z"},
        ),
    ],
)
def test_schedule_request_snapshot(settings, expected):
    assert _service(**settings) == expected


def _upload(schedule, body, status=202):
    response = MagicMock(status_code=status)
    response.json.return_value = body
    background = MagicMock()
    with (
        patch("app.services.upload_post.config.app", _CONFIG),
        patch("app.services.upload_post.os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=b"video")),
        patch("app.services.upload_post.requests.post", return_value=response) as post,
        patch.object(UploadPostService, "check_status") as status_check,
    ):
        result = UploadPostService().upload_video(
            "/v.mp4", "Title", ["tiktok"], schedule=schedule, on_background_start=background
        )
    return result, post.call_args.kwargs["data"], status_check, background


def test_delayed_post_sends_the_date_and_does_not_poll():
    result, data, status_check, background = _upload(
        {"mode": "delay", "scheduled_date": "2026-11-04T13:00:00Z"},
        {"success": True, "job_id": "job-1", "scheduled_date": "2026-11-04T13:00:00Z"},
    )
    assert ("scheduled_date", "2026-11-04T13:00:00Z") in data
    assert not any(name == "add_to_queue" for name, _ in data)
    assert result["success"] is True and result["scheduled"] is True
    assert result["job_id"] == "job-1"
    # 平台结果要到发布时间才有，轮询一小时只会把定时发布误报为失败。
    status_check.assert_not_called()
    background.assert_not_called()


def test_queued_post_uses_the_upload_post_queue():
    result, data, _status, _background = _upload(
        {"mode": "queue"},
        {"success": True, "job_id": "job-2", "queue_slot": "2026-11-05T09:00:00+00:00"},
    )
    assert ("add_to_queue", "true") in data
    assert not any(name == "scheduled_date" for name, _ in data)
    assert result["job_id"] == "job-2"


def test_scheduled_response_without_job_id_is_unconfirmed():
    result, *_ = _upload({"mode": "queue"}, {"success": True})
    assert result["success"] is False
    assert "unconfirmed" in result["error"]


def test_cross_post_forwards_the_queued_schedule():
    state = MemoryState()
    state.update_task(
        "scheduled", state=tm.const.TASK_STATE_COMPLETE, videos=["final.mp4"],
        cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
    )
    schedule = {"mode": "delay", "scheduled_date": "2026-11-04T13:00:00Z"}
    with (
        patch.object(tm.sm, "state", state),
        patch.object(tm.llm, "generate_social_metadata", return_value={"caption": "c"}),
        patch.object(
            tm.upload_post, "cross_post_video",
            return_value={"success": True, "scheduled": True, "job_id": "job-3"},
        ) as upload,
    ):
        tm._run_cross_post(
            "scheduled", ("final.mp4",), "Coffee", "Script", "en", ("tiktok",), "public",
            False, None, schedule,
        )
    assert upload.call_args.kwargs["schedule"] == schedule
    task = state.get_task("scheduled")
    assert task["cross_post_state"] == tm.const.CROSS_POST_STATE_COMPLETE
    assert task["cross_post_results"][0]["job_id"] == "job-3"


def test_queueing_snapshots_the_schedule(monkeypatch):
    submitted = {}

    class Executor:
        def submit(self, fn, *args):
            submitted["args"] = args
            future = MagicMock()
            return future

    # 独立的发布名额，避免占用模块级队列容量影响其它测试。
    monkeypatch.setattr(tm, "_cross_post_slots", threading.BoundedSemaphore(1))
    monkeypatch.setattr(tm, "_cross_post_executor", Executor())
    monkeypatch.setattr(
        upload_post.upload_post_service, "schedule_request",
        lambda: {"mode": "queue"},
    )
    monkeypatch.setattr(tm, "_register_cross_post_future", lambda *a: None)
    params = tm.VideoParams(video_subject="Coffee")
    assert tm._schedule_cross_post("t", ["final.mp4"], params, "Script", ["tiktok"], "public") is None
    assert submitted["args"][-1] == {"mode": "queue"}
