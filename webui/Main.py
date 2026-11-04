import os
import sys
from pathlib import Path

import streamlit as st

# WebUI 作为独立入口运行时，需要让项目根目录优先于第三方依赖，
# 避免依赖中的同名 app 包遮蔽 MoneyPrinterTurbo 自己的 app 包。
root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if root_dir in sys.path:
    sys.path.remove(root_dir)
sys.path.insert(0, root_dir)

from app.models.schema import VideoParams

st.set_page_config(
    page_title="MoneyPrinterTurbo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Report a bug": "https://github.com/harry0703/MoneyPrinterTurbo/issues",
        "About": "# MoneyPrinterTurbo\nSimply provide a topic or keyword for a video, and it will "
        "automatically generate the video copy, video materials, video subtitles, "
        "and video background music before synthesizing a high-definition short "
        "video.\n\nhttps://github.com/harry0703/MoneyPrinterTurbo",
    },
)


# Streamlit 1.59 会在页面右上角默认展示 Deploy、skills nudge 等平台入口。
# MoneyPrinterTurbo 是面向终端用户的本地工具，这些入口会造成顶部大块空白，
# 也会让新用户误以为需要安装额外组件。这里统一隐藏 Streamlit 平台工具栏，
# 并压缩主容器顶部留白，只保留项目自己的标题、语言选择和业务设置区域。
style_file = Path(__file__).with_name("styles.css")
streamlit_style = f"<style>{style_file.read_text(encoding='utf-8')}</style>"
st.markdown(streamlit_style, unsafe_allow_html=True)

from webui.common import (
    _initialize_session_state,
    _save_runtime_config,
    init_log,
    tr,
)
from webui.components.audio_panel import _render_audio_settings
from webui.components.generation_controls import _render_generation_controls
from webui.components.library import render_library_dialog
from webui.components.script_panel import _render_script_settings
from webui.components.settings_dialog import _render_settings_dialog
from webui.components.settings_transfer import _apply_pending_settings_preset
from webui.components.subtitle_panel import _render_subtitle_settings
from webui.components.tasks import (
    _apply_pending_task_restore,
    _render_task_restore_dialog,
)
from webui.components.top_bar import _render_top_bar
from webui.components.video_panel import _render_video_settings

_initialize_session_state()


init_log()


def _render_application():
    """按固定顺序渲染顶部栏、弹窗、生成表单和任务结果。"""
    _render_top_bar()

    # Streamlit 同一时间只能打开一个弹窗，设置优先。
    if st.session_state.get("settings_dialog_open", False):
        _render_settings_dialog()
    elif st.session_state.get("library_dialog_open", False):
        render_library_dialog()

    if _apply_pending_settings_preset():
        st.success(tr("Settings Preset Imported"))

    restore_applied = _apply_pending_task_restore()
    restore_candidate_id = st.session_state.get("task_restore_candidate_id")
    if restore_candidate_id:
        _render_task_restore_dialog(restore_candidate_id)
    restore_succeeded = st.session_state.pop("task_restore_succeeded", False)
    if restore_applied or restore_succeeded:
        st.success(tr("Task Configuration Loaded"))

    with st.container(key="main_settings_grid"):
        panel = st.columns(4)
    left_panel = panel[0]
    middle_panel = panel[1]
    audio_panel = panel[2]
    right_panel = panel[3]

    params = VideoParams(video_subject="")
    params.match_materials_to_script = bool(
        st.session_state.get("match_materials_to_script", False)
    )
    _render_script_settings(left_panel, params)

    uploaded_files = _render_video_settings(middle_panel, params)
    uploaded_audio_file, uploaded_bgm_file, voice_mode = _render_audio_settings(
        audio_panel, params
    )

    _render_subtitle_settings(right_panel, params)

    generation_submitted = _render_generation_controls(
        params,
        uploaded_files,
        uploaded_audio_file,
        uploaded_bgm_file,
        voice_mode,
    )

    # 生成分支在启动后台线程前已经请求过保存。普通控件交互继续请求非阻塞保存；
    # 如果后台任务正在使用配置，配置层会在任务结束时自动应用并落盘最新值。
    if not generation_submitted:
        _save_runtime_config()


_render_application()
