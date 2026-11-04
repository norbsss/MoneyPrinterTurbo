"""每个平台单独的发布文案与 AI 生成标注。"""

from unittest.mock import MagicMock, mock_open, patch

from app.services import task as tm
from app.services.state import MemoryState
from app.services.upload_post import UploadPostService

_CONFIG = {
    "upload_post_enabled": True,
    "upload_post_api_key": "test-key",
    "upload_post_username": "testuser",
    "upload_post_platforms": ["tiktok", "instagram", "youtube"],
}


def _post_data(platforms, config=None, **kwargs):
    response = MagicMock(status_code=200)
    response.json.return_value = {
        "success": True,
        "request_id": "r1",
        "results": {platform: {"success": True} for platform in platforms},
    }
    with (
        patch("app.services.upload_post.config.app", {**_CONFIG, **(config or {})}),
        patch("app.services.upload_post.os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=b"video")),
        patch("app.services.upload_post.requests.post", return_value=response) as post,
    ):
        result = UploadPostService().upload_video("/v.mp4", "Fallback", platforms, **kwargs)
    assert result["success"], result
    return post.call_args.kwargs["data"]


def _values(data, key):
    return [value for name, value in data if name == key]


def test_each_platform_gets_its_own_text_and_the_ai_label():
    data = _post_data(
        ["tiktok", "instagram"],
        platform_titles={"tiktok": "Short one #a", "instagram": "Longer story #b #c", "x": "ignored"},
    )
    assert _values(data, "title") == ["Fallback"]
    assert _values(data, "tiktok_title") == ["Short one #a"]
    assert _values(data, "instagram_title") == ["Longer story #b #c"]
    assert _values(data, "x_title") == []
    assert _values(data, "is_ai_generated") == ["true"]


def test_text_for_unselected_platforms_is_not_sent():
    data = _post_data(["tiktok"], platform_titles={"tiktok": "T", "instagram": "I"})
    assert _values(data, "instagram_title") == []


def test_ai_label_can_be_turned_off_and_is_not_sent_for_youtube_only():
    assert _values(_post_data(["tiktok"], config={"upload_post_ai_label": False}), "is_ai_generated") == []
    # YouTube 使用 containsSyntheticMedia，不需要这个字段。
    assert _values(_post_data(["youtube"]), "is_ai_generated") == []


def test_social_post_text_appends_hashtags_once():
    metadata = {"caption": "Morning coffee ritual #coffee", "hashtags": ["coffee", "#barista", " ", "#"]}
    assert tm._social_post_text(metadata) == "Morning coffee ritual #coffee\n\n#barista"
    assert tm._social_post_text({"title": "Only a title"}) == "Only a title"


def test_cross_post_generates_text_for_each_platform():
    state = MemoryState()
    state.update_task(
        "per-platform",
        state=tm.const.TASK_STATE_COMPLETE,
        videos=["final.mp4"],
        cross_post_state=tm.const.CROSS_POST_STATE_PENDING,
    )

    def metadata(platform, **_kwargs):
        return {"title": f"{platform} title", "caption": f"{platform} caption", "hashtags": [f"#{platform}"]}

    with (
        patch.object(tm.sm, "state", state),
        patch.object(tm.llm, "generate_social_metadata", side_effect=metadata) as generate,
        patch.object(tm.upload_post, "cross_post_video", return_value={"success": True}) as upload,
    ):
        tm._run_cross_post(
            "per-platform", ("final.mp4",), "Coffee", "Script", "en",
            ("tiktok", "instagram", "youtube"), "public",
        )

    assert sorted(call.kwargs["platform"] for call in generate.call_args_list) == [
        "instagram_reels", "tiktok", "youtube_shorts",
    ]
    kwargs = upload.call_args.kwargs
    assert kwargs["platform_titles"] == {
        "tiktok": "tiktok caption\n\n#tiktok",
        "instagram": "instagram_reels caption\n\n#instagram_reels",
    }
    assert kwargs["youtube_extra"]["youtube_title"] == "youtube_shorts title"
    assert state.get_task("per-platform")["cross_post_state"] == tm.const.CROSS_POST_STATE_COMPLETE
