"""设置弹窗中的本机 AI 面板：硬件、运行环境、模型下载与 GPU 服务。

AI 图片和 AI 视频两个素材分区都可能同时显示本面板，因此控件 key 都带上
分区类型，避免同一页面出现重复 key。
"""

import os

import streamlit as st

from app.config import config
from app.services import local_ai
from local_ai import catalog
from webui.common import _set_runtime_config, tr

_FIT_LABELS = {
    catalog.FIT_GPU: "local_ai.fit.gpu",
    catalog.FIT_OFFLOAD: "local_ai.fit.offload",
    catalog.FIT_CPU: "local_ai.fit.cpu",
    catalog.FIT_NONE: "local_ai.fit.none",
}
_KIND_TITLES = {"image": "local_ai.models.image", "video": "local_ai.models.video"}


def _gb(value_bytes):
    return f"{value_bytes / 1024**3:.1f} GB"


def _model_setting(kind):
    return f"local_ai_{kind}_model"


def _render_hardware(section):
    hardware = local_ai.detect_hardware()
    st.markdown(f"**{tr('local_ai.hardware')}**")
    if hardware.gpus:
        for gpu in hardware.gpus:
            st.caption(f"GPU {gpu.index}: {gpu.name} · {gpu.vram_gb:g} GB")
    else:
        st.caption(tr("local_ai.no_gpu"))
    st.caption(f"{tr('local_ai.ram')}: {hardware.ram_gb:g} GB")

    nvidia = [gpu for gpu in hardware.gpus if gpu.vendor == "nvidia"]
    if len(nvidia) > 1:
        # 配置中空值表示自动；控件用 "auto" 作为选项值，避免空字符串选项。
        options = ["auto"] + [str(gpu.index) for gpu in nvidia]
        saved = str(config.app.get("local_ai_gpu_index", "") or "") or "auto"
        selected = st.selectbox(
            tr("local_ai.gpu"),
            options,
            index=options.index(saved) if saved in options else 0,
            # 标签在本次渲染时算好：format_func 也会在页面运行之外被调用。
            format_func={
                "auto": tr("local_ai.gpu.auto"),
                **{str(gpu.index): f"GPU {gpu.index}: {gpu.name}" for gpu in nvidia},
            }.get,
            key=f"local_ai_gpu_index_select_{section}",
        )
        if selected != saved:
            _set_runtime_config("app", "local_ai_gpu_index", "" if selected == "auto" else selected)


@st.fragment(run_every="2s")
def _render_install_progress():
    status = local_ai.install_status() or {}
    if status.get("state") == "running":
        st.info(tr("local_ai.runtime.installing"))
        return
    # 安装结束后整页刷新，让运行环境与模型区域显示最新状态。
    st.rerun()


def _render_runtime(section):
    st.markdown(f"**{tr('local_ai.runtime')}**")
    status = local_ai.install_status() or {}
    if status.get("state") == "running":
        _render_install_progress()
        return
    if status.get("state") == "failed":
        st.error(f"{tr('local_ai.runtime.failed')}: {status.get('error', '')}")
        st.caption(f"`{local_ai.install_log_path()}`")
    if local_ai.is_runtime_installed():
        st.success(tr("local_ai.runtime.installed"))
        label = tr("local_ai.runtime.reinstall")
    else:
        st.caption(tr("local_ai.runtime.help"))
        label = tr("local_ai.runtime.install")
    if st.button(label, key=f"local_ai_install_runtime_{section}"):
        local_ai.start_runtime_install()
        st.rerun()


@st.fragment(run_every="2s")
def _render_download_progress(model_id):
    status = local_ai.download_status(model_id) or {}
    if status.get("state") in ("queued", "downloading"):
        total = status.get("bytes_total") or 0
        done = status.get("bytes_done") or 0
        st.progress(
            min(done / total, 1.0) if total else 0.0,
            text=f"{tr('local_ai.model.downloading')} {_gb(done)} / {_gb(total)}",
        )
        return
    st.rerun()


def _render_model(section, fit, selected_id, recommended_id):
    model = fit.model
    installed = local_ai.model_installed(model)
    status = local_ai.download_status(model.id) or {}
    with st.container(border=True):
        title = model.name
        if model.id == recommended_id:
            title += f" · {tr('local_ai.model.recommended')}"
        st.markdown(f"**{title}**")
        st.caption(
            f"{tr(_FIT_LABELS[fit.mode])} · {model.download_gb:g} GB · "
            f"{tr('local_ai.model.license')}: {model.license}"
        )
        if model.notes:
            st.caption(model.notes)
        if status.get("state") in ("queued", "downloading"):
            _render_download_progress(model.id)
            return
        if status.get("state") == "failed":
            st.error(status.get("error", ""))
        columns = st.columns(2)
        if installed:
            if model.id == selected_id:
                columns[0].success(tr("local_ai.model.active"))
            elif columns[0].button(
                tr("local_ai.model.use"), key=f"local_ai_use_{model.id}_{section}"
            ):
                _set_runtime_config("app", _model_setting(model.kind), model.id)
                st.rerun()
            if columns[1].button(
                tr("local_ai.model.delete"), key=f"local_ai_delete_{model.id}_{section}"
            ):
                try:
                    local_ai.delete_model(model.id)
                except local_ai.LocalAIError as error:
                    st.error(str(error))
                else:
                    st.rerun()
        elif fit.mode != catalog.FIT_NONE and columns[0].button(
            tr("local_ai.model.download"), key=f"local_ai_download_{model.id}_{section}"
        ):
            local_ai.start_download(model.id)
            # 当前所选模型还不可用时，改用正在下载的模型，下载完成即可生成。
            current = local_ai.video_model() if model.kind == "video" else local_ai.image_model()
            if current is None or not local_ai.model_installed(current):
                _set_runtime_config("app", _model_setting(model.kind), model.id)
            st.rerun()


def _render_models(section):
    hardware = local_ai.gpu_hardware()
    # 当前分区的模型排在前面；视频的图生视频模型也需要生图模型画出起始画面。
    for kind in (section, "image" if section == "video" else "video"):
        st.markdown(f"**{tr(_KIND_TITLES[kind])}**")
        recommended = catalog.recommend(hardware, kind)
        selected = local_ai.video_model() if kind == "video" else local_ai.image_model()
        for fit in catalog.rank(hardware, kind):
            _render_model(
                section,
                fit,
                selected.id if selected else "",
                recommended.model.id if recommended else "",
            )
    # 代码样式可避免 Markdown 把 Windows 路径中的反斜杠当作转义符吞掉。
    st.caption(f"{tr('local_ai.models.folder')}: `{local_ai.models_dir()}`")


def _render_server(section):
    st.markdown(f"**{tr('local_ai.server')}**")
    status = local_ai.server_status()
    columns = st.columns(2)
    if status:
        loaded = " · ".join(
            name for name in (status.get("model"), status.get("video_model")) if name
        )
        columns[0].success(f"{tr('local_ai.server.running')} · {loaded or '-'}")
        if columns[1].button(tr("local_ai.server.stop"), key=f"local_ai_stop_server_{section}"):
            local_ai.stop_server()
            st.rerun()
    else:
        columns[0].caption(tr("local_ai.server.stopped"))
        if local_ai.is_runtime_installed() and columns[1].button(
            tr("local_ai.server.start"), key=f"local_ai_start_server_{section}"
        ):
            try:
                with st.spinner(tr("local_ai.server.starting")):
                    local_ai.ensure_server()
            except local_ai.LocalAIError as error:
                st.error(str(error))
            else:
                st.rerun()


def _render_advanced(section):
    with st.expander(tr("local_ai.advanced")):
        token = st.text_input(
            tr("local_ai.hf_token"),
            value=str(config.app.get("local_ai_hf_api_token", "") or ""),
            type="password",
            help=tr("local_ai.hf_token.help"),
            key=f"local_ai_hf_api_token_input_{section}",
        )
        if token != str(config.app.get("local_ai_hf_api_token", "") or ""):
            _set_runtime_config("app", "local_ai_hf_api_token", token.strip())
        models_dir = st.text_input(
            tr("local_ai.models.folder"),
            value=str(config.app.get("local_ai_models_dir", "") or ""),
            placeholder=os.path.join(local_ai.local_ai_dir(), "models"),
            key=f"local_ai_models_dir_input_{section}",
        )
        if models_dir != str(config.app.get("local_ai_models_dir", "") or ""):
            _set_runtime_config("app", "local_ai_models_dir", models_dir.strip())


def render_local_ai_settings(section="image"):
    """``section`` 是显示本面板的素材分区：``"image"`` 或 ``"video"``。"""
    if section == "video":
        st.markdown(f"**{tr('local_ai.video.label')}**")
        st.caption(tr("local_ai.video.help"))
    else:
        st.markdown(f"**{tr('local_ai.label')}**")
        st.caption(tr("local_ai.help"))
    _render_hardware(section)
    _render_runtime(section)
    _render_models(section)
    _render_server(section)
    _render_advanced(section)
