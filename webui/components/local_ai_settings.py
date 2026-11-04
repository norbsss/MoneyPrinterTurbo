"""设置弹窗中的本机 AI 生图面板：硬件、运行环境、模型下载与 GPU 服务。"""

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


def _gb(value_bytes):
    return f"{value_bytes / 1024**3:.1f} GB"


def _render_hardware():
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
            format_func=lambda value: tr("local_ai.gpu.auto") if value == "auto" else next(
                f"GPU {gpu.index}: {gpu.name}" for gpu in nvidia if str(gpu.index) == value
            ),
            key="local_ai_gpu_index_select",
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


def _render_runtime():
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
    if st.button(label, key="local_ai_install_runtime"):
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


def _render_model(fit, selected_id, recommended_id):
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
            elif columns[0].button(tr("local_ai.model.use"), key=f"local_ai_use_{model.id}"):
                _set_runtime_config("app", "local_ai_image_model", model.id)
                st.rerun()
            if columns[1].button(tr("local_ai.model.delete"), key=f"local_ai_delete_{model.id}"):
                try:
                    local_ai.delete_model(model.id)
                except local_ai.LocalAIError as error:
                    st.error(str(error))
                else:
                    st.rerun()
        elif fit.mode != catalog.FIT_NONE and columns[0].button(
            tr("local_ai.model.download"), key=f"local_ai_download_{model.id}"
        ):
            local_ai.start_download(model.id)
            # 当前所选模型还不可用时，改用正在下载的模型，下载完成即可生成。
            current = local_ai.image_model()
            if current is None or not local_ai.model_installed(current):
                _set_runtime_config("app", "local_ai_image_model", model.id)
            st.rerun()


def _render_models():
    st.markdown(f"**{tr('local_ai.models')}**")
    hardware = local_ai.gpu_hardware()
    recommended = catalog.recommend(hardware, "image")
    selected = local_ai.image_model()
    for fit in catalog.rank(hardware, "image"):
        _render_model(
            fit,
            selected.id if selected else "",
            recommended.model.id if recommended else "",
        )
    # 代码样式可避免 Markdown 把 Windows 路径中的反斜杠当作转义符吞掉。
    st.caption(f"{tr('local_ai.models.folder')}: `{local_ai.models_dir()}`")


def _render_server():
    st.markdown(f"**{tr('local_ai.server')}**")
    status = local_ai.server_status()
    columns = st.columns(2)
    if status:
        model = status.get("model") or "-"
        columns[0].success(f"{tr('local_ai.server.running')} · {model}")
        if columns[1].button(tr("local_ai.server.stop"), key="local_ai_stop_server"):
            local_ai.stop_server()
            st.rerun()
    else:
        columns[0].caption(tr("local_ai.server.stopped"))
        if local_ai.is_runtime_installed() and columns[1].button(
            tr("local_ai.server.start"), key="local_ai_start_server"
        ):
            try:
                with st.spinner(tr("local_ai.server.starting")):
                    local_ai.ensure_server()
            except local_ai.LocalAIError as error:
                st.error(str(error))
            else:
                st.rerun()


def _render_advanced():
    with st.expander(tr("local_ai.advanced")):
        token = st.text_input(
            tr("local_ai.hf_token"),
            value=str(config.app.get("local_ai_hf_api_token", "") or ""),
            type="password",
            help=tr("local_ai.hf_token.help"),
            key="local_ai_hf_api_token_input",
        )
        if token != str(config.app.get("local_ai_hf_api_token", "") or ""):
            _set_runtime_config("app", "local_ai_hf_api_token", token.strip())
        models_dir = st.text_input(
            tr("local_ai.models.folder"),
            value=str(config.app.get("local_ai_models_dir", "") or ""),
            placeholder=os.path.join(local_ai.local_ai_dir(), "models"),
            key="local_ai_models_dir_input",
        )
        if models_dir != str(config.app.get("local_ai_models_dir", "") or ""):
            _set_runtime_config("app", "local_ai_models_dir", models_dir.strip())


def render_local_ai_settings():
    st.markdown(f"**{tr('local_ai.label')}**")
    st.caption(tr("local_ai.help"))
    _render_hardware()
    _render_runtime()
    _render_models()
    _render_server()
    _render_advanced()
