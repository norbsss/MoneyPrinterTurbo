"""当前生成任务的进度、日志与结果展示。"""

import mimetypes
import os
import time
from collections.abc import Mapping

import streamlit as st
from loguru import logger

from app.config import config
from app.models import const
from app.services import webui_task
from app.services import state as sm
from webui.common import tr
from webui.components.tasks import (
    _build_video_download_name,
    _current_generation_task_id,
    _normalize_task_state,
    _remove_active_generation_task,
    open_task_folder,
)


def _render_generation_logs(task_id):
    """渲染后台任务日志快照，不从工作线程访问 Streamlit 会话状态。"""
    if config.ui.get("hide_log", False):
        return

    log_records = webui_task.get_task_logs(task_id)
    if not log_records:
        return

    st.code("\n".join(log_records))


def _render_generation_activity(task):
    activity = task.get("generation_activity") or {}
    phase = activity.get("phase", "")
    labels = {
        "queued": "Generation Waiting In Queue",
        "preparing": "Generation Preparing",
        "script": "Generation Writing Script",
        "terms": "Generation Preparing Terms",
        "audio": "Generation Preparing Audio",
        "subtitle": "Generation Preparing Subtitles",
        "materials": "Generation Preparing Materials",
        "submitting": "Generation Submitting",
        "checking": "Generation Checking",
        "remote_queued": "Generation Remote Queued",
        "generating": "Generation Remote Running",
        "generated": "Generation Remote Completed",
        "downloading": "Generation Downloading",
        "verifying": "Generation Verifying",
        "rendering": "Generation Rendering",
    }
    st.info(tr(labels.get(phase, "Generating Video")))
    now = time.time()
    started = (task.get("generation_submitted_at") or task.get("generation_started_at")
               or activity.get("started_at") or now)
    elapsed = max(0, int(now - float(started)))
    st.caption(tr("Generation Elapsed").format(time=f"{elapsed // 60:02d}:{elapsed % 60:02d}"))
    if phase in {"submitting", "remote_queued", "generating", "checking"}:
        st.caption(tr("Generation Long Wait Hint"))
        if activity.get("retrying"):
            st.warning(tr("Generation Poll Retry Hint"))
        if activity.get("confirmed_at"):
            age = max(0, int(now - float(activity["confirmed_at"])))
            st.caption(tr("Generation Last Checked").format(seconds=age))
    if phase == "downloading" and activity.get("downloaded_bytes"):
        downloaded = activity["downloaded_bytes"] / (1024 * 1024)
        st.caption(tr("Generation Downloaded").format(size=f"{downloaded:.1f}"))
    if activity.get("remote_id"):
        st.caption(tr("Generation Remote Task").format(task_id=activity["remote_id"]))


def _render_generation_task_snapshot(task_id, task):
    """根据状态存储中的快照渲染进度、失败原因或最终成片。"""
    if not task:
        st.info(tr("Generating Video"))
        _render_generation_logs(task_id)
        return

    state = _normalize_task_state(task.get("state"))
    progress = max(0, min(100, int(task.get("progress", 0) or 0)))
    if state == const.TASK_STATE_PROCESSING:
        _render_generation_activity(task)
        st.progress(
            progress,
            text=f"{tr('Task Progress')}: {progress}%",
        )
        _render_generation_logs(task_id)
        return

    video_files = task.get("videos") or []
    if state == const.TASK_STATE_FAILED:
        error = str(task.get("error") or "").strip()
        error_key = task.get("error_key")
        error_params = task.get("error_params") or {}
        if (
            isinstance(error_key, str)
            and error_key.startswith("codex.")
            and isinstance(error_params, Mapping)
        ):
            error = tr(error_key).format(**error_params)
        message = tr("Video Generation Failed")
        st.error(f"{message}: {error}" if error else message)
        activity = task.get("generation_activity") or {}
        if activity.get("remote_id"):
            st.warning(tr("Generation Existing Task Hint").format(task_id=activity["remote_id"]))
        if not video_files:
            _render_generation_logs(task_id)
            return
    elif state != const.TASK_STATE_COMPLETE or not video_files:
        st.error(tr("Video Generation Failed"))
        _render_generation_logs(task_id)
        return

    if state == const.TASK_STATE_COMPLETE:
        st.success(tr("Video Generation Completed"))
    for warning in task.get("warnings") or []:
        if isinstance(warning, Mapping) and warning.get("code") == "batch_materials_reused":
            st.warning(
                tr("Batch Material Reuse Warning").format(
                    index=warning.get("video_index", ""), count=warning.get("count", 0)
                )
            )
        elif isinstance(warning, Mapping) and warning.get("code") == "sonilo_bgm_failed":
            st.warning(
                tr("Sonilo BGM Fallback Warning").format(
                    index=warning.get("video_index", "")
                )
            )
        elif (
            isinstance(warning, Mapping)
            and warning.get("code") == "elevenlabs_bgm_failed"
        ):
            st.warning(
                tr("ElevenLabs BGM Fallback Warning").format(
                    index=warning.get("video_index", "")
                )
            )
        else:
            st.warning(str(warning))

    available_videos = [
        (index, url)
        for index, url in enumerate(video_files)
        if os.path.isfile(url)
    ]
    for index, url in enumerate(video_files):
        if os.path.isfile(url):
            continue
        logger.warning(
            f"generated video is unavailable: "
            f"task_id={task_id}, video_file={url}"
        )

    try:
        if not available_videos:
            st.warning(tr("Generated Video Files Unavailable"))
        else:
            player_cols = st.columns(len(available_videos) * 2 + 1)
            for player_index, (video_index, url) in enumerate(available_videos):
                with player_cols[player_index * 2 + 1]:
                    st.video(url)
                    download_label = tr("Download Video")
                    if len(video_files) > 1:
                        download_label = f"{download_label} {video_index + 1}"
                    download_name = _build_video_download_name(
                        task.get("video_subject"),
                        video_index + 1,
                        len(video_files),
                    )
                    with open(url, "rb") as video_file:
                        st.download_button(
                            download_label,
                            data=video_file,
                            file_name=download_name,
                            mime=mimetypes.guess_type(url)[0] or "video/mp4",
                            key=f"download_generated_video_{task_id}_{video_index}",
                            icon=":material/download:",
                            on_click="ignore",
                            use_container_width=True,
                        )
    except Exception as exc:
        logger.exception(
            f"failed to render generated video preview: task_id={task_id}, "
            f"video_files={video_files}, error={exc}"
        )

    _render_generation_logs(task_id)
    if (
        state == const.TASK_STATE_COMPLETE
        and st.session_state.get("handled_generation_task_id") != task_id
    ):
        # Fragment 可能重复渲染同一个完成任务。无论是否开启自动打开目录，
        # 每个任务都只处理一次完成事件，避免重复弹出资源管理器或重复写入日志。
        st.session_state["handled_generation_task_id"] = task_id
        if config.ui.get("open_task_folder_on_completion", True):
            open_task_folder(task_id)
        logger.info(f"{tr('Video Generation Completed')}: task_id={task_id}")


@st.fragment(run_every=webui_task.TASK_LOG_REFRESH_INTERVAL_SECONDS)
def _render_running_generation_task(task_id):
    """只在任务运行期间轮询；结束后切回静态结果，停止不必要的定时刷新。"""
    try:
        task = sm.state.get_task(task_id)
    except Exception as exc:
        logger.exception(
            f"failed to query WebUI generation task: task_id={task_id}, error={exc}"
        )
        st.warning(tr("Generation Status Unavailable"))
        return

    if not task and not webui_task.is_generation_active(task_id):
        st.warning(tr("Generation Task Unavailable"))
        return
    state = _normalize_task_state((task or {}).get("state"))
    if state in {const.TASK_STATE_COMPLETE, const.TASK_STATE_FAILED}:
        _remove_active_generation_task(task_id)
        # 完整页面脚本现在没有耗时生成逻辑，可以安全 rerun 并把结果改为静态
        # 渲染。这样任务结束后不会让浏览器永久保留一个两秒轮询的 Fragment。
        st.rerun(scope="app")

    _render_generation_task_snapshot(task_id, task)
    if not webui_task.is_generation_active(task_id):
        # 持久状态可能属于重启前或另一服务进程，不能断言远端任务已停止。
        # 只允许解除当前页面的跟踪，不取消任务、不改写其真实状态。
        st.warning(tr("Generation Untracked Worker Hint"))
        if st.button(tr("Stop Tracking Generation"), key=f"stop_tracking_generation_{task_id}"):
            _remove_active_generation_task(task_id)
            st.session_state["current_generation_task_id"] = ""
            st.query_params.pop("generation_task", None)
            st.rerun(scope="app")


def _render_current_generation_task():
    """在生成按钮下方恢复当前页面最近提交任务的可查询 UI。"""
    task_id = _current_generation_task_id()
    if not task_id:
        return

    try:
        task = sm.state.get_task(task_id)
    except Exception as exc:
        logger.exception(
            f"failed to query current WebUI task: task_id={task_id}, error={exc}"
        )
        # 即使首次读取失败，也要注册 Fragment，让状态后端恢复后自动继续。
        _render_running_generation_task(task_id)
        return

    if not task and not webui_task.is_generation_active(task_id):
        st.warning(tr("Generation Task Unavailable"))
        return
    state = _normalize_task_state((task or {}).get("state"))
    if state in {const.TASK_STATE_COMPLETE, const.TASK_STATE_FAILED}:
        _remove_active_generation_task(task_id)
        _render_generation_task_snapshot(task_id, task)
        return

    _render_running_generation_task(task_id)
