"""视频库弹窗：以缩略图网格浏览已生成的视频，并查看自动发布状态。

视频库用弹窗而不是独立页面：切换页面时生成表单不会渲染，Streamlit 会清掉
这些控件的会话状态，用户正在填写的主题和文案就会丢失。
"""

import os
import subprocess
from datetime import datetime

import streamlit as st
from loguru import logger

from app.models import const
from app.services import state as sm
from app.utils import utils
from webui.common import localized_dialog, tr
from webui.components.tasks import (
    _collect_task_summaries,
    _delete_task,
    _format_task_time,
    _open_task_path,
    _queue_task_restore,
)

THUMBNAIL_NAME = "thumbnail.jpg"
_PAGE_SIZE = 12
_COLUMNS = 3
_THUMBNAIL_TIMEOUT_SECONDS = 30


def task_thumbnail(video_file: str) -> str | None:
    """返回视频首帧缩略图，缺失或比视频旧时用 ffmpeg 重新生成。"""
    thumbnail = os.path.join(os.path.dirname(video_file), THUMBNAIL_NAME)
    try:
        if os.path.getmtime(thumbnail) >= os.path.getmtime(video_file):
            return thumbnail
    except OSError:
        pass
    # 片头常是黑场或淡入，先取 0.5 秒处；更短的视频退回第一帧。
    for offset in ("0.5", "0"):
        try:
            subprocess.run(
                [
                    utils.get_ffmpeg_binary(),
                    "-y", "-loglevel", "error",
                    "-ss", offset, "-i", video_file,
                    "-frames:v", "1", "-vf", "scale=360:-2",
                    thumbnail,
                ],
                capture_output=True,
                timeout=_THUMBNAIL_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as error:
            logger.warning(f"failed to create task thumbnail: {video_file}, {error}")
            return None
        if os.path.isfile(thumbnail) and os.path.getsize(thumbnail) > 0:
            return thumbnail
    return None


def _format_iso_time(value) -> str:
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value or "")
    return moment.astimezone().strftime("%Y-%m-%d %H:%M")


def publish_status(task: dict) -> tuple[str, str]:
    """视频的发布状态：(标签, 补充说明)。"""
    state = task.get("cross_post_state")
    if not state:
        return tr("library.publish.none"), ""
    if state in (const.CROSS_POST_STATE_PENDING, const.CROSS_POST_STATE_PROCESSING):
        return tr("library.publish.pending"), ""
    if state == const.CROSS_POST_STATE_FAILED:
        return tr("library.publish.failed"), str(task.get("cross_post_error") or "")
    results = [item for item in task.get("cross_post_results") or [] if isinstance(item, dict)]
    scheduled = next((item for item in results if item.get("scheduled")), None)
    if scheduled is None:
        return tr("library.publish.done"), ""
    when = scheduled.get("scheduled_date") or scheduled.get("queue_slot")
    if when:
        return tr("library.publish.scheduled").format(date=_format_iso_time(when)), ""
    return tr("library.publish.queued"), ""


def _finished_videos():
    tasks = _collect_task_summaries(limit=200)
    return [
        task for task in tasks
        if task.get("video_file") and os.path.isfile(task["video_file"])
    ]


def _runtime_task(task_id: str) -> dict:
    try:
        return sm.state.get_task(task_id) or {}
    except Exception as error:
        logger.warning(f"failed to load task for the library: {task_id}, {error}")
        return {}


def _safe_key(task_id: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in task_id)[:40]


def _render_card(task: dict):
    task_id = task["task_id"]
    key = _safe_key(task_id)
    with st.container(border=True, key=f"library_card_{key}"):
        thumbnail = task_thumbnail(task["video_file"])
        if thumbnail:
            st.image(thumbnail, width="stretch")
        st.markdown(f"**{task['subject']}**")
        label, detail = publish_status(_runtime_task(task_id))
        st.caption(f"{_format_task_time(task['mtime'])} · {label}")
        if detail:
            st.caption(detail)
        actions = st.columns(4, gap="small")
        # 回调在页面重跑前执行，播放器位于网格上方，本次重跑就能显示。
        actions[0].button(
            tr("library.play"), key=f"library_play_{key}", icon=":material/play_arrow:",
            help=tr("library.play"), width="stretch",
            on_click=st.session_state.__setitem__,
            args=("library_preview_video", task["video_file"]),
        )
        if actions[1].button(
            tr("Open Task Folder"), key=f"library_open_{key}", icon=":material/folder_open:",
            help=tr("Open Task Folder"), width="stretch",
        ):
            _open_task_path(task["task_path"])
        has_restore_data = os.path.isfile(os.path.join(task["task_path"], "script.json"))
        if actions[2].button(
            tr("Regenerate Task"), key=f"library_restore_{key}", icon=":material/replay:",
            help=tr("Regenerate Task"), width="stretch", disabled=not has_restore_data,
        ):
            st.session_state.pop("library_dialog_open", None)
            _queue_task_restore(task_id)
            st.rerun()
        if actions[3].button(
            tr("Delete Task"), key=f"library_delete_{key}", icon=":material/delete:",
            help=tr("Delete Task"), width="stretch",
        ):
            if _delete_task(task_id, task["task_path"], task.get("state")):
                st.toast(tr("Task Deleted"))
                st.rerun()
            else:
                st.error(tr("Task Delete Failed"))


def _render_preview():
    preview = st.session_state.get("library_preview_video")
    if not preview:
        return
    tasks_root = os.path.abspath(utils.task_dir())
    # 只播放任务目录内的文件，避免会话状态被改成任意本地路径。
    if not (os.path.abspath(preview).startswith(tasks_root + os.sep) and os.path.isfile(preview)):
        st.session_state.pop("library_preview_video", None)
        return
    # 竖屏视频按弹窗全宽显示会高过整个屏幕，放在中间一栏。
    player = st.columns([1, 1, 1])[1]
    with player:
        st.video(preview, autoplay=True)
        if st.button(tr("Close"), key="library_close_preview", width="stretch"):
            st.session_state.pop("library_preview_video", None)
            st.rerun()
    st.divider()


def _dismiss_library():
    st.session_state.pop("library_dialog_open", None)
    st.session_state.pop("library_preview_video", None)


def open_library():
    st.session_state["library_dialog_open"] = True


@localized_dialog("library.title", width="large", on_dismiss=_dismiss_library)
def render_library_dialog():
    _render_preview()
    tasks = _finished_videos()
    if not tasks:
        st.info(tr("library.empty"))
        return
    shown = st.session_state.get("library_shown", _PAGE_SIZE)
    for start in range(0, min(shown, len(tasks)), _COLUMNS):
        columns = st.columns(_COLUMNS)
        for column, task in zip(columns, tasks[start:start + _COLUMNS]):
            with column:
                _render_card(task)
    if len(tasks) > shown and st.button(tr("library.show_more"), key="library_show_more"):
        st.session_state["library_shown"] = shown + _PAGE_SIZE
        st.rerun()
