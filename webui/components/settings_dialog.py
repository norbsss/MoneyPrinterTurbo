"""设置弹窗：通用、模型、素材服务商、缓存与日志设置。"""

import os
import re
import webbrowser
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from loguru import logger

from app.config import config
from app.models.llm_provider import (
    DEFAULT_LLM_PROVIDER_ID,
    LLM_PROVIDER_REGISTRY,
    get_llm_provider,
    normalize_provider_override,
)
from app.services import (
    cache_manager,
    llm,
    metaso_minimax,
    muapi,
    infistar_video,
    ofox,
    volcengine_seedance,
)
from app.utils.logging_utils import LOG_LEVELS, get_logging_options
from app.utils.request_logging import trace_http_call
from webui.common import (
    CUSTOM_LLM_ENDPOINT_ID,
    OFOX_REFERRAL_URL,
    UPLOAD_POST_API_KEYS_URL,
    UPLOAD_POST_MANAGE_USERS_URL,
    VIDEO_SOURCE_GROUPS,
    _save_runtime_config,
    _set_runtime_config,
    locales,
    localized_dialog,
    tr,
    tr_optional,
)
from webui.components.settings_transfer import _render_key_backup_settings
from webui.components.top_bar import (
    _dismiss_settings_dialog,
    _open_settings_dialog,
)
from webui.components.video_panel import _render_jalapeno_provider_settings
from webui.components.widgets import (
    _save_settings_change,
    localized_widget_key,
    stable_selectbox,
)


def get_llm_provider_tips(provider_id, **kwargs):
    # LLM provider 说明文案统一使用 `llm_provider_tips.<provider_id>` 规则。
    # 这样新增 provider 时只需要在 locale 中补文案；没有文案时不展示提示块，
    # 避免 Main.py 里继续堆叠大量中英文硬编码说明。
    provider = get_llm_provider(provider_id)
    if provider is None:
        return ""

    # Provider 配置说明目前统一维护中文和英文两套规范模板；其它界面语言
    # 统一使用英文，避免在 locale 中复制英文后长期不同步。后续某个语种完成
    # 全量翻译后，再将它加入这里的独立维护范围。
    ui_language = st.session_state.get("ui_language", "en")
    tips_language = ui_language if ui_language in {"zh", "en"} else "en"
    tips = (
        locales.get(tips_language, {}).get("Translation", {}).get(provider.tips_key, "")
    )
    if not tips:
        return tips

    service_endpoint = provider.preferred_service_endpoint(
        prefer_international=tips_language == "en"
    )
    api_key_url = (
        service_endpoint.api_key_url
        if service_endpoint
        else provider.effective_api_key_url()
    )
    format_context = {
        "api_key_url": api_key_url,
        "default_model": provider.default_model,
        "default_base_url": (
            service_endpoint.base_url
            if service_endpoint
            else provider.effective_default_base_url
        ),
        "model_docs_url": (
            service_endpoint.model_docs_url
            if service_endpoint and service_endpoint.model_docs_url
            else provider.effective_model_docs_url(
                prefer_international=tips_language == "en"
            )
        ),
        **{
            f"default_{field.config_suffix}": field.default_value
            for field in provider.extra_fields
        },
        **kwargs,
    }
    try:
        return tips.format(**format_context)
    except Exception as e:
        logger.warning(f"format llm provider tips failed: {provider_id}, {e}")
        return tips


def format_llm_connection_error(provider_id, base_url, error):
    """为可明确定位的鉴权错误补充配置检查建议，同时保留原始响应。"""
    error_text = str(error or "").strip()
    if provider_id in {"openai_codex", "openai_chatgpt"}:
        if provider_id == "openai_chatgpt":
            from app.services.chatgpt_oauth import ERROR_MESSAGES
            message_prefix = "chatgpt.oauth"
        else:
            from app.services.codex_oauth import ERROR_MESSAGES
            message_prefix = "codex.oauth"

        for key, message in ERROR_MESSAGES.items():
            if error_text == message:
                return tr(f"{message_prefix}.{key}")
        status_match = re.fullmatch(
            r"OpenAI - (?:Codex|ChatGPT \(Native OAuth\)) returned HTTP (\d+)\. Check the selected model and your account access\.",
            error_text,
        )
        if status_match:
            return tr(f"{message_prefix}.http_error").format(status=status_match[1])
    normalized_error = error_text.lower()
    authentication_markers = (
        "401",
        "authentication",
        "invalid api key",
        "invalid_api_key",
        "unauthorized",
    )
    provider = get_llm_provider(provider_id)
    if provider is None or not provider.service_endpoints or not any(
        marker in normalized_error for marker in authentication_markers
    ):
        return error_text

    message = tr_optional(
        provider.authentication_error_key,
        fallback_language="en",
    )
    if not message:
        return error_text
    return message.format(base_url=base_url or "-", error=error_text)


def get_llm_provider_label(provider):
    return tr_optional(provider.label_key) or provider.default_label


@st.cache_data(ttl=300, show_spinner=False)
def get_groq_model_ids(api_key: str, base_url: str) -> list[str]:
    if not api_key:
        return []

    normalized_base_url = (
        (base_url or "https://api.groq.com/openai/v1").strip().rstrip("/")
    )
    models_url = f"{normalized_base_url}/models"

    try:
        response = trace_http_call(
            "GET", requests.get,
            models_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", [])

        model_ids = []
        for item in data:
            if isinstance(item, dict):
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id.strip():
                    model_ids.append(model_id.strip())

        return sorted(set(model_ids))
    except Exception as e:
        logger.warning(f"failed to fetch groq models: {e}")
        return []


def _get_material_api_keys(config_key):
    """将配置中的素材 API Key 统一转换为 WebUI 可编辑字符串。"""
    api_keys = config.snapshot_config_with_pending(config.app).get(config_key, [])
    if isinstance(api_keys, str):
        api_keys = [api_keys]
    return ", ".join(api_keys)


def _save_material_api_keys(config_key, value):
    """保存逗号分隔的素材 API Key，并允许用户显式清空旧配置。"""
    normalized_value = value.replace(" ", "")
    _set_runtime_config(
        "app",
        config_key,
        normalized_value.split(",") if normalized_value else [],
    )


def _format_file_size(size_bytes):
    """将字节数格式化为适合设置页展示的紧凑容量文本。"""
    size = float(max(0, size_bytes))
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.2f} {unit}"
        size /= 1024
    return f"{size_bytes} B"


@st.cache_data(ttl=30, show_spinner=False)
def _get_video_cache_stats_data(max_age_days=None):
    """
    短周期缓存目录统计，避免设置弹窗内普通控件交互反复扫描大量文件。

    缓存键包含清理天数，因此切换范围只会为每个范围扫描一次；主动刷新或清理
    完成后会显式清空，最多 30 秒的缓存不会影响实际删除时的二次扫描。

    这里只缓存纯 dict，而不是 VideoCacheStats 实例：st.cache_data 需要用 pickle
    序列化返回值，而 pickle 保存自定义类时按「模块名 + 类名」引用，并校验解析
    出来的类与实例的类是同一个对象。Streamlit 源码监视器在本地源码文件发生变化
    时（包含 Windows 上防病毒或索引器引起的偶发事件）会清空 sys.modules 中的
    被监视模块并重新导入；此时仍打开的设置弹窗（Dialog 继承 fragment 行为，
    内部交互不会重建 Main.py 顶层引用）持有的还是旧模块里的类，pickle 会抛
    PicklingError: it's not the same object as ...，并被 Streamlit 包装成
    UnserializableReturnValueError（对应 Streamlit 官方 issue #14593 的已知缺陷）。
    纯 dict 按值序列化、不含类引用，因此不受模块重新导入影响。
    """
    stats = cache_manager.get_video_cache_stats(max_age_days=max_age_days)
    # 字段名与 VideoCacheStats 完全一致，供下方 _get_video_cache_stats 直接展开
    # 还原；今后若 VideoCacheStats 增删字段，需要同步维护这里的字段映射。
    return {
        "file_count": stats.file_count,
        "total_size": stats.total_size,
        "oldest_mtime": stats.oldest_mtime,
        "newest_mtime": stats.newest_mtime,
    }


def _get_video_cache_stats(max_age_days=None) -> cache_manager.VideoCacheStats:
    """在缓存边界之外把纯数据还原为 VideoCacheStats，调用方属性访问方式不变。"""

    # 还原时使用当前生效的 cache_manager 模块；即使模块被重新导入，也只是重建
    # 一个轻量 dataclass，不会再触发 pickle 的「类引用身份」校验。
    return cache_manager.VideoCacheStats(
        **_get_video_cache_stats_data(max_age_days=max_age_days)
    )


def _render_cache_management_settings(panel):
    """渲染默认在线视频素材缓存的统计、预览和安全清理操作。"""
    with panel:
        cleanup_message = st.session_state.pop("video_cache_cleanup_message", None)
        if cleanup_message:
            message_type, message = cleanup_message
            if message_type == "success":
                st.success(message)
            else:
                st.warning(message)

        st.caption(tr("Video Cache Directory"))
        st.code(cache_manager.video_cache_dir(), language="text")

        total_stats = _get_video_cache_stats()
        metric_count, metric_size, metric_oldest = st.columns(3)
        metric_count.metric(tr("Cache File Count"), total_stats.file_count)
        metric_size.metric(
            tr("Cache Total Size"), _format_file_size(total_stats.total_size)
        )
        oldest_text = (
            datetime.fromtimestamp(total_stats.oldest_mtime).strftime("%Y-%m-%d")
            if total_stats.oldest_mtime is not None
            else "-"
        )
        metric_oldest.metric(tr("Oldest Cache Date"), oldest_text)

        st.caption(tr("Video Cache Management Help"))
        cleanup_options = (30, 7, 90, None)
        cleanup_labels = {
            30: tr("Cache Older Than 30 Days"),
            7: tr("Cache Older Than 7 Days"),
            90: tr("Cache Older Than 90 Days"),
            None: tr("All Video Cache"),
        }
        max_age_days = st.selectbox(
            tr("Cache Cleanup Range"),
            options=cleanup_options,
            format_func=lambda value: cleanup_labels[value],
            key="video_cache_cleanup_range",
        )
        cleanup_preview = _get_video_cache_stats(max_age_days=max_age_days)
        st.info(
            tr("Cache Cleanup Preview").format(
                count=cleanup_preview.file_count,
                size=_format_file_size(cleanup_preview.total_size),
            )
        )

        confirm_nonce = st.session_state.get("video_cache_cleanup_confirm_nonce", 0)
        confirmed = st.checkbox(
            tr("Confirm Cache Cleanup"),
            key=f"video_cache_cleanup_confirm_{confirm_nonce}",
        )
        refresh_col, open_col, cleanup_col = st.columns(3)
        if refresh_col.button(
            tr("Refresh Cache Stats"),
            key="refresh_video_cache_stats",
            use_container_width=True,
            icon=":material/refresh:",
        ):
            _get_video_cache_stats_data.clear()
            st.rerun(scope="fragment")

        if open_col.button(
            tr("Open Cache Directory"),
            key="open_video_cache_directory",
            use_container_width=True,
            icon=":material/folder_open:",
        ):
            webbrowser.open(Path(cache_manager.video_cache_dir()).as_uri())

        cleanup_disabled = not confirmed or cleanup_preview.file_count == 0
        if cleanup_col.button(
            tr("Clean Cache Now"),
            key="clean_video_cache_now",
            type="primary",
            disabled=cleanup_disabled,
            use_container_width=True,
            icon=":material/delete_sweep:",
        ):
            result = cache_manager.clean_video_cache(max_age_days=max_age_days)
            message_key = (
                "Cache Cleanup Completed With Failures"
                if result.failed_count
                else "Cache Cleanup Completed"
            )
            st.session_state["video_cache_cleanup_message"] = (
                "warning" if result.failed_count else "success",
                tr(message_key).format(
                    count=result.deleted_count,
                    size=_format_file_size(result.deleted_size),
                    failed=result.failed_count,
                ),
            )
            # Streamlit 不允许在控件实例化后修改同名 session_state。通过递增
            # nonce 让下一次 fragment rerun 创建未勾选的新控件，避免清理完成后
            # 危险确认状态被继续保留。
            st.session_state["video_cache_cleanup_confirm_nonce"] = confirm_nonce + 1
            _get_video_cache_stats_data.clear()
            st.rerun(scope="fragment")


def _render_logging_settings():
    level, log_requests = get_logging_options()
    with st.expander(tr("Diagnostic Logging")):
        st.caption(tr("Diagnostic Logging Help"))
        with st.form("diagnostic_logging_form"):
            selected_level = st.selectbox(
                tr("Log Level"),
                LOG_LEVELS,
                index=LOG_LEVELS.index(level),
                key="diagnostic_log_level",
            )
            selected_trace = st.checkbox(
                tr("Trace API Requests"),
                value=log_requests,
                key="diagnostic_log_requests",
                help=tr("Trace API Requests Help"),
            )
            apply = st.form_submit_button(tr("Apply Logging Settings"))
        if apply:
            # 诊断不改变生成参数，直接生效；持久化仍走非阻塞配置队列。
            config.update_logging_options(selected_level, selected_trace)
            _save_runtime_config()
            st.success(tr("Logging Settings Applied"))


# 设置属于低频操作，使用中等尺寸 Dialog 避免长期占用主页面纵向空间，
# 同时控制阅读行宽，避免弹窗在宽屏设备上显得过于松散。
# Dialog 继承 fragment 行为，内部控件交互只重绘弹窗；函数末尾单独保存配置，
# 关闭时通过回调触发整页同步，确保生成流程读取最新 Provider 和界面设置。
def _remember_settings_tab(widget_key, targets):
    selected = st.session_state.get(widget_key)
    for target, label in targets.items():
        if selected == label:
            st.session_state["settings_active_tab"] = target
            break


def _remember_settings_selection(memory_key, widget_key):
    st.session_state[memory_key] = st.session_state[widget_key]


def _remember_material_provider(group, widget_key):
    remembered = dict(st.session_state.get("settings_material_providers", {}))
    remembered[group] = st.session_state[widget_key]
    st.session_state["settings_material_group"] = group
    st.session_state["settings_material_providers"] = remembered


def _initialize_material_navigation(target, settings):
    groups = {
        group: providers
        for group, providers in VIDEO_SOURCE_GROUPS.items()
        if group != "local"
    }
    source = settings.get("video_source", "pexels")
    source_group = next(
        (group for group, providers in groups.items() if source in providers), None
    )
    explicit = {"stock": "stock_video", "video": "ai_video", "image": "ai_image"}.get(
        target
    )
    group = explicit or (source_group if target == "material" else None)
    group = group or st.session_state.get(
        "settings_material_group", source_group or "stock_video"
    )
    if group not in groups:
        group = "stock_video"
    remembered = dict(st.session_state.get("settings_material_providers", {}))
    if target == "material" and source_group:
        remembered[group] = source
    elif group not in remembered or remembered[group] not in groups[group]:
        remembered[group] = source if source in groups[group] else groups[group][0]
    st.session_state["settings_material_group"] = group
    st.session_state["settings_material_providers"] = remembered
    st.session_state[
        localized_widget_key(f"settings_material_provider_{group}_select")
    ] = remembered[group]


def _material_provider_labels():
    return {
        "pexels": tr("Pexels"),
        "pixabay": tr("Pixabay"),
        "coverr": tr("Coverr"),
        "metaso_minimax": tr("Metaso MiniMax H3"),
        "ofox": tr("OFox AI Video"),
        "loomloom": tr("Shengsuan Cloud AI Video"),
        "volcengine_seedance": tr("Volcano Engine Seedance"),
        "infistar": tr("Infistar AI Video"),
        "jalapeno": tr("Jalapeno Cloud AI Video"),
        "wavespeed": tr("WaveSpeed AI Video"),
        "muapi": tr("MuAPI AI Video"),
        "openai_image": tr("OpenAI Compatible Text-to-Image"),
        "codex_image": tr("codex.image.label"),
        "local_image": tr("local_ai.label"),
        "local_video": tr("local_ai.video.label"),
        "local": tr("Local file"),
    }


def _general_provider_readiness(settings):
    from app.services import chatgpt_oauth, codex_oauth, kimi_code_oauth
    from webui.settings_providers import (
        configured_llm_providers,
        configured_material_providers,
    )

    oauth = {}
    for provider, read_status in (
        ("openai_codex", codex_oauth.account_status),
        ("openai_chatgpt", chatgpt_oauth.account_status),
        ("kimi_code", kimi_code_oauth.is_authorized),
    ):
        try:
            oauth[provider] = bool(read_status())
        except (ValueError, OSError):
            oauth[provider] = False
    return {
        "llm": configured_llm_providers(settings, oauth),
        **configured_material_providers(
            settings, VIDEO_SOURCE_GROUPS, oauth["openai_codex"]
        ),
    }


def _sync_general_provider_widgets(name, value):
    prefix = f"{name}_"
    for widget_key in list(st.session_state):
        if widget_key.startswith(prefix) and not widget_key.endswith("_component"):
            st.session_state[widget_key] = value
    st.session_state[localized_widget_key(name)] = value


def _activate_general_provider(section):
    settings = config.snapshot_config_with_pending(config.app)
    selected = st.session_state.get(
        localized_widget_key(f"settings_general_{section}_provider_select")
    )
    if selected not in _general_provider_readiness(settings).get(section, ()):
        st.session_state["settings_general_activation_error"] = True
        return
    st.session_state.pop("settings_general_activation_error", None)
    if section == "llm":
        _set_runtime_config("app", "llm_provider", selected)
        _sync_general_provider_widgets("llm_provider_select", selected)
    else:
        _set_runtime_config("app", "video_source", selected)
        _sync_general_provider_widgets("video_source_select", selected)
        remembered = dict(st.session_state.get("settings_material_providers", {}))
        remembered[section] = selected
        st.session_state["settings_material_providers"] = remembered
        st.session_state["settings_material_group"] = section
        _sync_general_provider_widgets(
            f"settings_material_provider_{section}_select", selected
        )
    _save_runtime_config()


def _render_general_settings(settings):
    configured = _general_provider_readiness(settings)
    material_labels = _material_provider_labels()
    llm_labels = {
        provider.provider_id: get_llm_provider_label(provider)
        for provider in LLM_PROVIDER_REGISTRY
    }
    st.subheader(tr("General Providers Title"))
    st.caption(tr("General Providers Help"))
    if st.session_state.pop("settings_general_activation_error", False):
        st.warning(tr("Provider Configuration Changed"))
    cards = (
        ("llm", "LLM Provider", "psychology", "llm"),
        ("ai_video", "AI Video", "movie", "video"),
        ("ai_image", "AI Image", "image", "image"),
        ("stock_video", "Stock Video", "video_library", "stock"),
    )
    for section, title, icon, target in cards:
        labels = llm_labels if section == "llm" else material_labels
        active = (
            settings.get("llm_provider", DEFAULT_LLM_PROVIDER_ID)
            if section == "llm"
            else settings.get("video_source", "pexels")
        )
        is_active_category = section == "llm" or active in VIDEO_SOURCE_GROUPS[section]
        options = configured[section]
        widget_key = localized_widget_key(f"settings_general_{section}_provider_select")
        memory_key = f"settings_general_{section}_provider"
        active_key = f"settings_general_{section}_active_snapshot"
        active_value = active if is_active_category else None
        if st.session_state.get(active_key) != active_value and active_value in options:
            st.session_state[memory_key] = active_value
            st.session_state[widget_key] = active_value
        st.session_state[active_key] = active_value
        remembered = st.session_state.get(memory_key)
        if st.session_state.get(widget_key) not in options:
            st.session_state[widget_key] = (
                remembered
                if remembered in options
                else active
                if active in options
                else options[0]
                if options
                else None
            )
        with st.container(border=True):
            st.subheader(f":material/{icon}: {tr(title)}")
            if is_active_category and active in options:
                st.caption(
                    tr("Active Provider").format(
                        provider=labels.get(active, str(active))
                    )
                )
            if not options:
                st.info(tr("No Configured Providers"))
                st.button(
                    tr("Configure Providers"),
                    key=f"settings_general_configure_{section}",
                    width="stretch",
                    on_click=_open_settings_dialog,
                    args=(target,),
                )
                continue
            selection = st.selectbox(
                tr("Configured Providers"),
                options=options,
                key=widget_key,
                format_func=labels.get,
                index=None,
                on_change=_remember_settings_selection,
                args=(memory_key, widget_key),
            )
            if selection is not None:
                st.session_state[memory_key] = selection
            use_column, configure_column = st.columns(2)
            use_column.button(
                tr("Use Provider"),
                key=f"settings_general_use_{section}",
                type="primary",
                width="stretch",
                disabled=selection is None,
                on_click=_activate_general_provider,
                args=(section,),
            )
            configure_column.button(
                tr("Configure Providers"),
                key=f"settings_general_configure_{section}",
                width="stretch",
                on_click=_open_settings_dialog,
                args=(target,),
            )


def _save_settings_endpoint(provider, widget_key):
    endpoint_id = st.session_state.get(widget_key)
    endpoint = next(
        (
            item
            for item in provider.service_endpoints
            if item.endpoint_id == endpoint_id
        ),
        None,
    )
    if endpoint:
        _set_runtime_config(
            "app",
            provider.config_key("base_url"),
            normalize_provider_override(
                endpoint.base_url, provider.effective_default_base_url
            ),
        )


def _save_settings_api_keys(config_key, widget_key):
    _save_material_api_keys(config_key, st.session_state.get(widget_key, ""))


def _render_material_provider_settings(provider):
    app_settings = config.snapshot_config_with_pending(config.app)
    if provider in VIDEO_SOURCE_GROUPS["stock_video"]:
        labels = {
            "pexels": "Pexels API Key",
            "pixabay": "Pixabay API Key",
            "coverr": "Coverr API Key",
        }
        config_key = f"{provider}_api_keys"
        value = st.text_input(
            tr(labels[provider]),
            value=_get_material_api_keys(config_key),
            type="password",
            key=f"{config_key}_input",
            on_change=_save_settings_api_keys,
            args=(config_key, f"{config_key}_input"),
        )
        _save_material_api_keys(config_key, value)
    elif provider == "metaso_minimax":
        st.markdown(f"**{tr('Metaso MiniMax H3')}**")
        metaso_api_key = st.text_input(
            tr("Metaso MiniMax API Key"),
            value=str(app_settings.get("metaso_minimax_api_key", "") or "").strip(),
            type="password",
            help=tr("Metaso MiniMax API Key Help"),
            key="metaso_minimax_api_key_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "metaso_minimax_api_key",
                "metaso_minimax_api_key_input",
                lambda value: value.strip(),
            ),
        )
        _set_runtime_config("app", "metaso_minimax_api_key", metaso_api_key.strip())
        configured_metaso_base_url = str(
            app_settings.get(
                "metaso_minimax_base_url",
                metaso_minimax.DEFAULT_BASE_URL,
            )
            or metaso_minimax.DEFAULT_BASE_URL
        ).strip()
        metaso_base_url = st.text_input(
            tr("Metaso MiniMax Base URL"),
            value=(
                ""
                if configured_metaso_base_url == metaso_minimax.DEFAULT_BASE_URL
                else configured_metaso_base_url
            ),
            placeholder=metaso_minimax.DEFAULT_BASE_URL,
            key="metaso_minimax_base_url_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "metaso_minimax_base_url",
                "metaso_minimax_base_url_input",
                lambda value: value.strip() or metaso_minimax.DEFAULT_BASE_URL,
            ),
        )
        _set_runtime_config(
            "app",
            "metaso_minimax_base_url",
            metaso_base_url.strip() or metaso_minimax.DEFAULT_BASE_URL,
        )
        configured_metaso_resolution = (
            str(
                app_settings.get(
                    "metaso_minimax_resolution",
                    metaso_minimax.DEFAULT_RESOLUTION,
                )
            )
            .strip()
            .upper()
        )
        metaso_resolution_options = sorted(
            metaso_minimax.SUPPORTED_RESOLUTIONS,
            key=lambda value: value != metaso_minimax.DEFAULT_RESOLUTION,
        )
        resolution_is_valid = (
            configured_metaso_resolution in metaso_minimax.SUPPORTED_RESOLUTIONS
        )
        if not resolution_is_valid:
            # 分辨率直接影响计费。手工配置错误时保留原值并要求用户
            # 主动选择，不能在打开设置弹窗时静默改成价格更高的 2K。
            st.error(
                tr("Metaso MiniMax Invalid Resolution").format(
                    value=configured_metaso_resolution,
                    supported=", ".join(metaso_resolution_options),
                )
            )
        metaso_resolution = st.selectbox(
            tr("Metaso MiniMax Resolution"),
            options=metaso_resolution_options,
            index=(
                metaso_resolution_options.index(configured_metaso_resolution)
                if resolution_is_valid
                else None
            ),
            key="metaso_minimax_resolution_input",
            help=tr("Metaso MiniMax Resolution Help"),
            placeholder=tr("Select Metaso MiniMax Resolution"),
            on_change=_save_settings_change,
            args=(
                "app",
                "metaso_minimax_resolution",
                "metaso_minimax_resolution_input",
                lambda value: value,
            ),
        )
        if metaso_resolution is not None:
            _set_runtime_config("app", "metaso_minimax_resolution", metaso_resolution)
    elif provider == "ofox":
        st.markdown("**OfoxAI**")
        st.caption(f"[OfoxAI]({OFOX_REFERRAL_URL}) · {tr('OFox AI Video Help')}")
        ofox_api_key = st.text_input(
            tr("OFox API Key"),
            value=str(app_settings.get("ofox_api_key", "") or ""),
            type="password",
            key="ofox_api_key_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "ofox_api_key",
                "ofox_api_key_input",
                lambda value: value.strip(),
            ),
        )
        _set_runtime_config("app", "ofox_api_key", ofox_api_key.strip())
        ofox_model = st.text_input(
            tr("OFox Text-to-Video Model"),
            value=str(
                app_settings.get(
                    "ofox_text_to_video_model",
                    ofox.DEFAULT_MODEL_ID,
                )
                or ofox.DEFAULT_MODEL_ID
            ),
            key="ofox_text_to_video_model_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "ofox_text_to_video_model",
                "ofox_text_to_video_model_input",
                lambda value: value.strip(),
            ),
        )
        _set_runtime_config("app", "ofox_text_to_video_model", ofox_model.strip())
        configured_ofox_base_url = str(
            app_settings.get("ofox_base_url", ofox.DEFAULT_BASE_URL)
            or ofox.DEFAULT_BASE_URL
        ).strip()
        ofox_base_url = st.text_input(
            tr("OFox Base URL"),
            value=(
                ""
                if configured_ofox_base_url == ofox.DEFAULT_BASE_URL
                else configured_ofox_base_url
            ),
            placeholder=ofox.DEFAULT_BASE_URL,
            key="ofox_base_url_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "ofox_base_url",
                "ofox_base_url_input",
                lambda value: value.strip() or ofox.DEFAULT_BASE_URL,
            ),
        )
        _set_runtime_config(
            "app",
            "ofox_base_url",
            ofox_base_url.strip() or ofox.DEFAULT_BASE_URL,
        )
        ofox_vendor_options = [
            (tr("OFox Vendor BytePlus"), "byteplus"),
            (tr("OFox Vendor Volcengine"), "volcengine"),
            (tr("OFox Vendor Auto"), ""),
        ]
        configured_ofox_vendor = str(
            app_settings.get("ofox_provider", ofox.DEFAULT_PROVIDER_TYPE) or ""
        ).strip()
        if configured_ofox_vendor not in {value for _, value in ofox_vendor_options}:
            # 用户在 config.toml 手工钉定了其它厂商名时保留该选择，
            # 避免打开设置页就被下拉框覆盖回默认值。
            ofox_vendor_options.append((configured_ofox_vendor, configured_ofox_vendor))
        selected_ofox_vendor = stable_selectbox(
            tr("OFox Upstream Vendor"),
            options=[value for _, value in ofox_vendor_options],
            default_value=configured_ofox_vendor,
            key="ofox_provider_select",
            format_func=lambda value: dict(
                (v, label) for label, v in ofox_vendor_options
            )[value],
            help=tr("OFox Upstream Vendor Help"),
            on_change=_save_settings_change,
            args=(
                "app",
                "ofox_provider",
                localized_widget_key("ofox_provider_select"),
                lambda value: value,
            ),
        )
        _set_runtime_config("app", "ofox_provider", selected_ofox_vendor)
    elif provider == "loomloom":
        st.markdown(f"**{tr('Shengsuan Cloud AI Video')}**")
        app_config_snapshot = config.snapshot_config_with_pending(config.app)
        if (
            str(app_config_snapshot.get("llm_provider", "") or "").lower()
            == "shengsuanyun"
        ):
            # 大模型 Provider 已选胜算云时，视频生成直接复用
            # 同一密钥，不再展示一个容易引起歧义的独立输入框。
            st.caption(tr("Shengsuan Cloud API Key Reused"))
        else:
            configured_loomloom_token = str(
                app_config_snapshot.get("loomloom_api_token", "") or ""
            ).strip()
            loomloom_api_token = st.text_input(
                tr("Shengsuan Cloud API Key"),
                value=configured_loomloom_token,
                type="password",
                key="loomloom_api_token_input",
                help=tr("Shengsuan Cloud API Key Help"),
                placeholder=tr("Shengsuan Cloud API Key Placeholder"),
                on_change=_save_settings_change,
                args=(
                    "app",
                    "loomloom_api_token",
                    "loomloom_api_token_input",
                    lambda value: value.strip(),
                ),
            ).strip()
            _set_runtime_config("app", "loomloom_api_token", loomloom_api_token)
    elif provider == "volcengine_seedance":
        seedance_api_key_value = str(
            app_settings.get("volcengine_seedance_api_key", "") or ""
        ).strip()
        shared_ark_api_key = str(
            app_settings.get("volcengine_api_key", "") or ""
        ).strip()
        environment_ark_api_key = os.getenv("VOLCENGINE_ARK_API_KEY", "").strip()
        seedance_reuses_llm_key = bool(
            not seedance_api_key_value
            and not environment_ark_api_key
            and shared_ark_api_key
        )
        seedance_title = f"**{tr('Volcano Engine Seedance')}**"
        if seedance_reuses_llm_key:
            # 只有复用大模型密钥无法从当前输入框直接看出，保留该提示
            # 可以避免用户误以为必须重复填写；普通配置状态不再赘述。
            seedance_title += f" :blue[{tr('Reusing LLM API Key')}]"
        st.markdown(seedance_title)
        seedance_api_key = st.text_input(
            tr("Volcano Engine Ark API Key"),
            value=seedance_api_key_value,
            type="password",
            help=tr("Volcano Engine Ark API Key Help"),
            key="volcengine_seedance_api_key_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "volcengine_seedance_api_key",
                "volcengine_seedance_api_key_input",
                lambda value: value.strip(),
            ),
        )
        _set_runtime_config(
            "app", "volcengine_seedance_api_key", seedance_api_key.strip()
        )
        configured_seedance_model = str(
            app_settings.get(
                "volcengine_seedance_model",
                volcengine_seedance.DEFAULT_MODEL_ID,
            )
            or volcengine_seedance.DEFAULT_MODEL_ID
        ).strip()
        seedance_model = st.text_input(
            tr("Volcano Engine Seedance Model"),
            # 内置默认值通过 placeholder 展示，用户自定义的
            # 模型或接入点 ID 仍作为真实值展示和保存。
            value=(
                ""
                if configured_seedance_model == volcengine_seedance.DEFAULT_MODEL_ID
                else configured_seedance_model
            ),
            placeholder=volcengine_seedance.DEFAULT_MODEL_ID,
            key="volcengine_seedance_model_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "volcengine_seedance_model",
                "volcengine_seedance_model_input",
                lambda value: value.strip() or volcengine_seedance.DEFAULT_MODEL_ID,
            ),
        )
        _set_runtime_config(
            "app",
            "volcengine_seedance_model",
            seedance_model.strip() or volcengine_seedance.DEFAULT_MODEL_ID,
        )
        configured_seedance_base_url = str(
            app_settings.get(
                "volcengine_seedance_base_url",
                volcengine_seedance.DEFAULT_BASE_URL,
            )
            or volcengine_seedance.DEFAULT_BASE_URL
        ).strip()
        seedance_base_url = st.text_input(
            tr("Volcano Engine Ark Base URL"),
            value=(
                ""
                if configured_seedance_base_url == volcengine_seedance.DEFAULT_BASE_URL
                else configured_seedance_base_url
            ),
            placeholder=volcengine_seedance.DEFAULT_BASE_URL,
            key="volcengine_seedance_base_url_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "volcengine_seedance_base_url",
                "volcengine_seedance_base_url_input",
                lambda value: value.strip() or volcengine_seedance.DEFAULT_BASE_URL,
            ),
        )
        _set_runtime_config(
            "app",
            "volcengine_seedance_base_url",
            seedance_base_url.strip() or volcengine_seedance.DEFAULT_BASE_URL,
        )
    elif provider == "infistar":
        st.markdown(f"**{tr('Infistar AI Video')}**")
        st.caption(tr("Infistar Video Key Help"))
        infistar_video_key = st.text_input(
            tr("Infistar Video API Key"),
            value=str(app_settings.get("infistar_video_api_key", "") or ""),
            type="password",
            key="infistar_video_api_key_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "infistar_video_api_key",
                "infistar_video_api_key_input",
                lambda value: value.strip(),
            ),
        )
        _set_runtime_config("app", "infistar_video_api_key", infistar_video_key.strip())
        infistar_video_base_url = st.text_input(
            tr("Infistar Video Base URL"),
            value=str(app_settings.get("infistar_video_base_url", "") or ""),
            placeholder=infistar_video.DEFAULT_BASE_URL,
            key="infistar_video_base_url_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "infistar_video_base_url",
                "infistar_video_base_url_input",
                lambda value: value.strip(),
            ),
        )
        _set_runtime_config(
            "app", "infistar_video_base_url", infistar_video_base_url.strip()
        )
    elif provider == "jalapeno":
        _render_jalapeno_provider_settings()
    elif provider == "wavespeed":
        wavespeed_api_key = _get_material_api_keys("wavespeed_api_keys")
        st.markdown("**WaveSpeed**")
        wavespeed_api_key = st.text_input(
            tr("WaveSpeed API Key"),
            value=wavespeed_api_key,
            type="password",
            key="wavespeed_api_keys_input",
            on_change=_save_settings_api_keys,
            args=("wavespeed_api_keys", "wavespeed_api_keys_input"),
        )
        _save_material_api_keys("wavespeed_api_keys", wavespeed_api_key)
    elif provider == "muapi":
        st.markdown(f"**{tr('MuAPI AI Video')}**")
        st.caption(tr("MuAPI AI Video Help"))
        muapi_api_key = st.text_input(
            tr("MuAPI API Key"),
            value=str(app_settings.get("muapi_api_key", "") or ""),
            type="password",
            help=tr("MuAPI API Key Help"),
            key="muapi_api_key_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "muapi_api_key",
                "muapi_api_key_input",
                lambda value: value.strip(),
            ),
        )
        _set_runtime_config("app", "muapi_api_key", muapi_api_key.strip())
        configured_muapi_base_url = str(
            app_settings.get("muapi_base_url", muapi.DEFAULT_BASE_URL)
            or muapi.DEFAULT_BASE_URL
        ).strip()
        muapi_base_url = st.text_input(
            tr("MuAPI Base URL"),
            value=(
                ""
                if configured_muapi_base_url == muapi.DEFAULT_BASE_URL
                else configured_muapi_base_url
            ),
            placeholder=muapi.DEFAULT_BASE_URL,
            key="muapi_base_url_input",
            help=tr("MuAPI Base URL Help"),
            on_change=_save_settings_change,
            args=(
                "app",
                "muapi_base_url",
                "muapi_base_url_input",
                lambda value: value.strip() or muapi.DEFAULT_BASE_URL,
            ),
        )
        _set_runtime_config(
            "app",
            "muapi_base_url",
            muapi_base_url.strip() or muapi.DEFAULT_BASE_URL,
        )
        configured_muapi_endpoint = str(
            app_settings.get("muapi_video_endpoint", muapi.DEFAULT_ENDPOINT)
            or muapi.DEFAULT_ENDPOINT
        ).strip()
        muapi_endpoint = st.text_input(
            tr("MuAPI Video Endpoint"),
            value=(
                ""
                if configured_muapi_endpoint == muapi.DEFAULT_ENDPOINT
                else configured_muapi_endpoint
            ),
            placeholder=muapi.DEFAULT_ENDPOINT,
            key="muapi_video_endpoint_input",
            help=tr("MuAPI Video Endpoint Help"),
            on_change=_save_settings_change,
            args=(
                "app",
                "muapi_video_endpoint",
                "muapi_video_endpoint_input",
                lambda value: value.strip() or muapi.DEFAULT_ENDPOINT,
            ),
        )
        _set_runtime_config(
            "app",
            "muapi_video_endpoint",
            muapi_endpoint.strip() or muapi.DEFAULT_ENDPOINT,
        )
        configured_muapi_resolution = str(
            app_settings.get("muapi_resolution", muapi.DEFAULT_RESOLUTION)
            or muapi.DEFAULT_RESOLUTION
        ).strip()
        muapi_resolution = st.text_input(
            tr("MuAPI Resolution"),
            value=(
                ""
                if configured_muapi_resolution == muapi.DEFAULT_RESOLUTION
                else configured_muapi_resolution
            ),
            placeholder=muapi.DEFAULT_RESOLUTION,
            key="muapi_resolution_input",
            help=tr("MuAPI Resolution Help"),
            on_change=_save_settings_change,
            args=(
                "app",
                "muapi_resolution",
                "muapi_resolution_input",
                lambda value: value.strip() or muapi.DEFAULT_RESOLUTION,
            ),
        )
        _set_runtime_config(
            "app",
            "muapi_resolution",
            muapi_resolution.strip() or muapi.DEFAULT_RESOLUTION,
        )
    elif provider == "codex_image":
        st.markdown(f"**{tr('codex.image.label')}**")
        st.caption(tr("codex.image.help"))
        from webui.codex_models import render_codex_model_selector

        codex_image_model = render_codex_model_selector(
            str(app_settings.get("codex_image_model_name", "") or ""),
            tr,
            prefix="codex_image",
            on_change=lambda model: _set_runtime_config(
                "app", "codex_image_model_name", model
            ),
        )
        _set_runtime_config("app", "codex_image_model_name", codex_image_model)
        codex_image_timeout = st.text_input(
            tr("codex.image.timeout"),
            value=str(app_settings.get("codex_image_timeout") or "300"),
            placeholder="300",
            key="codex_image_timeout_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "codex_image_timeout",
                "codex_image_timeout_input",
                lambda value: value.strip(),
            ),
        )
        _set_runtime_config("app", "codex_image_timeout", codex_image_timeout.strip())
        from webui.codex_login import render_codex_login

        with st.container():
            render_codex_login(tr)
    elif provider in ("local_image", "local_video"):
        from webui.components.local_ai_settings import render_local_ai_settings

        render_local_ai_settings("video" if provider == "local_video" else "image")
    elif provider == "openai_image":
        st.markdown(f"**{tr('OpenAI Compatible Text-to-Image')}**")

        openai_image_base_url = st.text_input(
            tr("OpenAI Image Base URL"),
            value=str(app_settings.get("openai_image_base_url", "") or ""),
            placeholder="https://api.openai.com/v1",
            key="openai_image_base_url_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "openai_image_base_url",
                "openai_image_base_url_input",
                lambda value: value.strip(),
            ),
        )
        _set_runtime_config(
            "app", "openai_image_base_url", openai_image_base_url.strip()
        )

        openai_image_api_key = _get_material_api_keys("openai_image_api_keys")
        openai_image_api_key = st.text_input(
            tr("OpenAI Image API Key"),
            value=openai_image_api_key,
            type="password",
            help=tr("OpenAI Image API Key Help"),
            key="openai_image_api_keys_input",
            on_change=_save_settings_api_keys,
            args=("openai_image_api_keys", "openai_image_api_keys_input"),
        )
        _save_material_api_keys("openai_image_api_keys", openai_image_api_key)

        openai_image_model = st.text_input(
            tr("OpenAI Image Model"),
            value=str(app_settings.get("openai_image_model", "") or ""),
            placeholder="gpt-image-2",
            key="openai_image_model_input",
            on_change=_save_settings_change,
            args=(
                "app",
                "openai_image_model",
                "openai_image_model_input",
                lambda value: value.strip(),
            ),
        )
        _set_runtime_config("app", "openai_image_model", openai_image_model.strip())
        # 只展示参考值，不将 OpenAI 官方端点写成默认配置。
        # 兼容服务的 Base URL 和模型 ID 没有统一值；留空不会让
        # 用户在未知情时误连官方付费接口，也不会覆盖旧配置。
        st.caption(tr("OpenAI Image Configuration Example"))

        with st.expander(tr("OpenAI Image Advanced Settings"), expanded=False):
            openai_image_size = st.text_input(
                tr("OpenAI Image Size"),
                value=str(app_settings.get("openai_image_size", "") or ""),
                placeholder="1024x1536",
                help=tr("OpenAI Image Size Help"),
                key="openai_image_size_input",
                on_change=_save_settings_change,
                args=(
                    "app",
                    "openai_image_size",
                    "openai_image_size_input",
                    lambda value: value.strip(),
                ),
            )
            _set_runtime_config("app", "openai_image_size", openai_image_size.strip())

            openai_image_prompt_template = st.text_input(
                tr("OpenAI Image Prompt Template"),
                value=str(app_settings.get("openai_image_prompt_template", "") or ""),
                placeholder="cinematic photo of {term}, photorealistic",
                help=tr("OpenAI Image Prompt Template Help"),
                key="openai_image_prompt_template_input",
                on_change=_save_settings_change,
                args=(
                    "app",
                    "openai_image_prompt_template",
                    "openai_image_prompt_template_input",
                    lambda value: value.strip(),
                ),
            )
            _set_runtime_config(
                "app",
                "openai_image_prompt_template",
                openai_image_prompt_template.strip(),
            )


@localized_dialog(
    "Settings",
    width="medium",
    on_dismiss=_dismiss_settings_dialog,
)
def _render_settings_dialog():
    app_settings = config.snapshot_config_with_pending(config.app)
    ui_settings = config.snapshot_config_with_pending(config.ui)
    with st.container():
        # 历史 hide_config 只用于隐藏旧基础设置面板。改为固定设置入口后，该值
        # 不再有用户可见意义，统一迁移为 false，避免旧配置影响后续版本。
        _set_runtime_config("app", "hide_config", False)
        tab_keys = {
            "general": "General Settings Tab",
            "llm": "LLM Settings Tab",
            "material": "Material API Tab",
            "publish": "Auto-Publish Settings",
            "interface": "Interface Settings Tab",
            "keys": "Key Backup Tab",
            "cache": "Cache Management Tab",
        }
        settings_tab_targets = {key: tr(label) for key, label in tab_keys.items()}
        settings_tabs_key = localized_widget_key("settings_dialog_tabs")
        target_tab = st.session_state.pop("settings_dialog_target_tab", None)
        if target_tab in {"stock", "video", "image", "material"}:
            _initialize_material_navigation(target_tab, app_settings)
        if target_tab in {"stock", "video", "image"}:
            target_tab = "material"
        active = target_tab or st.session_state.get("settings_active_tab", "general")
        if active not in settings_tab_targets:
            active = "general"
        if (
            target_tab
            or st.session_state.get(settings_tabs_key)
            not in settings_tab_targets.values()
        ):
            st.session_state[settings_tabs_key] = settings_tab_targets[active]
        panels = st.tabs(
            list(settings_tab_targets.values()),
            key=settings_tabs_key,
            on_change=_remember_settings_tab,
            args=(settings_tabs_key, settings_tab_targets),
        )
        (
            general_config_panel,
            middle_config_panel,
            right_config_panel,
            publish_config_panel,
            left_config_panel,
            key_backup_panel,
            cache_config_panel,
        ) = panels
        _remember_settings_tab(settings_tabs_key, settings_tab_targets)
        with general_config_panel:
            if general_config_panel.open:
                _render_general_settings(app_settings)

        with publish_config_panel:
            if publish_config_panel.open:
                st.write(
                    tr(
                        "Automatically publish generated videos to social media using upload-post.com"
                    )
                )
                st.info(
                    tr("Upload-Post Setup Guide").format(
                        api_keys_url=UPLOAD_POST_API_KEYS_URL,
                        manage_users_url=UPLOAD_POST_MANAGE_USERS_URL,
                    )
                )

                is_enabled = app_settings.get("upload_post_enabled", False)
                is_auto = app_settings.get("upload_post_auto_upload", False)

                # 两个键各自独立:enabled 允许外部流程调用 Upload-Post,
                # auto_upload 才决定渲染完成后是否自动发布。合并成一个复选框会在
                # 两键不一致的配置下,仅打开设置对话框就把 enabled 改写为 False。
                upload_post_enabled = st.checkbox(
                    tr("Enable Upload-Post Integration"),
                    value=is_enabled,
                    key="upload_post_enabled_checkbox",
                    on_change=_save_settings_change,
                    args=(
                        "app",
                        "upload_post_enabled",
                        "upload_post_enabled_checkbox",
                        lambda value: value,
                    ),
                )
                if upload_post_enabled != is_enabled:
                    _set_runtime_config(
                        "app", "upload_post_enabled", upload_post_enabled
                    )

                upload_post_auto_upload = st.checkbox(
                    tr("Enable Auto-Publish"),
                    value=is_auto,
                    key="upload_post_auto_upload_checkbox",
                    on_change=_save_settings_change,
                    args=(
                        "app",
                        "upload_post_auto_upload",
                        "upload_post_auto_upload_checkbox",
                        lambda value: value,
                    ),
                )
                if upload_post_auto_upload != is_auto:
                    _set_runtime_config(
                        "app", "upload_post_auto_upload", upload_post_auto_upload
                    )

                upload_post_api_key = st.text_input(
                    tr("Upload-Post API Key"),
                    value=app_settings.get("upload_post_api_key", ""),
                    type="password",
                    help=tr("Upload-Post API Key Help").format(
                        api_keys_url=UPLOAD_POST_API_KEYS_URL
                    ),
                    key="upload_post_api_key_input",
                    on_change=_save_settings_change,
                    args=(
                        "app",
                        "upload_post_api_key",
                        "upload_post_api_key_input",
                        lambda value: value,
                    ),
                )
                if upload_post_api_key != app_settings.get("upload_post_api_key", ""):
                    _set_runtime_config(
                        "app", "upload_post_api_key", upload_post_api_key
                    )

                upload_post_username = st.text_input(
                    tr("Upload-Post Profile Username"),
                    value=app_settings.get("upload_post_username", ""),
                    help=tr("Upload-Post Profile Username Help").format(
                        manage_users_url=UPLOAD_POST_MANAGE_USERS_URL
                    ),
                    key="upload_post_username_input",
                    on_change=_save_settings_change,
                    args=(
                        "app",
                        "upload_post_username",
                        "upload_post_username_input",
                        lambda value: value,
                    ),
                )
                if upload_post_username != app_settings.get("upload_post_username", ""):
                    _set_runtime_config(
                        "app", "upload_post_username", upload_post_username
                    )

                upload_post_platforms = st.multiselect(
                    tr("Platforms"),
                    options=["tiktok", "instagram", "youtube"],
                    default=app_settings.get(
                        "upload_post_platforms", ["tiktok", "instagram"]
                    ),
                    help="Select platforms to publish to",
                    key="upload_post_platforms_multiselect",
                    on_change=_save_settings_change,
                    args=(
                        "app",
                        "upload_post_platforms",
                        "upload_post_platforms_multiselect",
                        lambda value: value,
                    ),
                )
                if upload_post_platforms != app_settings.get(
                    "upload_post_platforms", ["tiktok", "instagram"]
                ):
                    _set_runtime_config(
                        "app", "upload_post_platforms", upload_post_platforms
                    )

                if {"tiktok", "instagram"} & set(upload_post_platforms):
                    # 与 YouTube 的 containsSyntheticMedia 一致默认开启；Instagram
                    # 的标注发布后无法撤销，因此在这里明确提示。
                    upload_post_ai_label = st.checkbox(
                        tr("upload_post.ai_label"),
                        value=app_settings.get("upload_post_ai_label", True) is True,
                        help=tr("upload_post.ai_label.help"),
                        key="upload_post_ai_label_checkbox",
                    )
                    if upload_post_ai_label != (
                        app_settings.get("upload_post_ai_label", True) is True
                    ):
                        _set_runtime_config("app", "upload_post_ai_label", upload_post_ai_label)

                if "youtube" in upload_post_platforms:
                    yt_status_options = ["public", "private", "unlisted"]
                    yt_saved = app_settings.get(
                        "upload_post_youtube_privacy_status", "public"
                    )
                    if yt_saved not in yt_status_options:
                        yt_saved = "public"
                    upload_post_youtube_privacy_status = st.selectbox(
                        tr("YouTube Privacy Status"),
                        options=yt_status_options,
                        index=yt_status_options.index(yt_saved),
                        key="upload_post_youtube_privacy_status_selectbox",
                        on_change=_save_settings_change,
                        args=(
                            "app",
                            "upload_post_youtube_privacy_status",
                            "upload_post_youtube_privacy_status_selectbox",
                            lambda value: value,
                        ),
                    )
                    if upload_post_youtube_privacy_status != app_settings.get(
                        "upload_post_youtube_privacy_status", "public"
                    ):
                        _set_runtime_config(
                            "app",
                            "upload_post_youtube_privacy_status",
                            upload_post_youtube_privacy_status,
                        )

                    # 受众声明只影响 YouTube 发布，不改变生成内容或其它平台的请求。
                    # 使用真正的布尔选项，避免把展示文字或字符串当成 API 参数。
                    saved_audience = app_settings.get(
                        "upload_post_youtube_made_for_kids", False
                    )
                    audience_labels = {
                        False: tr("Not Made for Kids"),
                        True: tr("Made for Kids"),
                    }
                    made_for_kids = st.selectbox(
                        tr("YouTube Audience"),
                        options=[False, True],
                        # 非法配置保持未选择，不在打开设置时擅自改成非儿童声明。
                        index=int(saved_audience)
                        if isinstance(saved_audience, bool)
                        else None,
                        format_func=audience_labels.get,
                        help=tr("YouTube Audience Help"),
                        key="upload_post_youtube_made_for_kids_selectbox",
                        on_change=_save_settings_change,
                        args=(
                            "app",
                            "upload_post_youtube_made_for_kids",
                            "upload_post_youtube_made_for_kids_selectbox",
                            lambda value: value,
                        ),
                    )
                    if isinstance(made_for_kids, bool):
                        _set_runtime_config(
                            "app", "upload_post_youtube_made_for_kids", made_for_kids
                        )

        # 左侧面板 - 日志设置
        with left_config_panel:
            if left_config_panel.open:
                hide_log = st.checkbox(
                    tr("Hide Log"),
                    value=ui_settings.get("hide_log", False),
                    key="hide_log_checkbox",
                    on_change=_save_settings_change,
                    args=("ui", "hide_log", "hide_log_checkbox", lambda value: value),
                )
                _set_runtime_config("ui", "hide_log", hide_log)
                _render_logging_settings()

        if cache_config_panel.open:
            _render_cache_management_settings(cache_config_panel)
        # 密钥恢复会写回配置并清除密码控件状态，必须在下面渲染这些控件之前执行。
        if key_backup_panel.open:
            _render_key_backup_settings(key_backup_panel)

        # 中间面板 - LLM 设置

        with middle_config_panel:
            if middle_config_panel.open:
                # 下拉顺序、默认 label 和稳定 provider id 全部来自 Registry；locale
                # 只覆盖展示文案，不再让 Main.py 维护第二份 Provider 列表。
                llm_provider_ids = [
                    provider.provider_id for provider in LLM_PROVIDER_REGISTRY
                ]
                llm_provider_labels = {
                    provider.provider_id: get_llm_provider_label(provider)
                    for provider in LLM_PROVIDER_REGISTRY
                }
                saved_llm_provider = app_settings.get(
                    "llm_provider", DEFAULT_LLM_PROVIDER_ID
                ).lower()
                if saved_llm_provider not in llm_provider_ids:
                    saved_llm_provider = DEFAULT_LLM_PROVIDER_ID

                llm_provider = stable_selectbox(
                    tr("LLM Provider"),
                    options=llm_provider_ids,
                    default_value=saved_llm_provider,
                    key="llm_provider_select",
                    format_func=lambda provider_id: llm_provider_labels[provider_id],
                    on_change=_save_settings_change,
                    args=(
                        "app",
                        "llm_provider",
                        localized_widget_key("llm_provider_select"),
                        lambda value: value,
                    ),
                )
                # 配置表单和 Provider 说明并排展示，减少长说明在窄列中的换行，
                # 同时充分利用基础设置面板的横向空间。
                llm_form_panel, llm_help_panel = st.columns(
                    [0.9, 1.1],
                    gap="large",
                    vertical_alignment="top",
                )
                llm_helper = llm_help_panel.container()
                _set_runtime_config("app", "llm_provider", llm_provider)
                llm_provider_spec = get_llm_provider(llm_provider)
                if llm_provider_spec is None:
                    # 正常情况下下拉选项全部来自 Registry，不会进入该分支；保留
                    # 明确错误用于诊断损坏的 session state 或后续接入遗漏。
                    raise RuntimeError(f"unsupported llm provider: {llm_provider}")

                llm_api_key = app_settings.get(
                    llm_provider_spec.config_key("api_key"), ""
                )
                configured_llm_base_url = app_settings.get(
                    llm_provider_spec.config_key("base_url"), ""
                )
                llm_default_base_url = llm_provider_spec.effective_default_base_url
                llm_base_url = configured_llm_base_url or llm_default_base_url
                llm_model_name = llm_provider_spec.resolve_model_name(
                    app_settings.get(llm_provider_spec.config_key("model_name"), "")
                )

                provider_tip_context = {}
                selected_service_endpoint = None
                if llm_provider_spec.service_endpoints:
                    # Kimi 等 Provider 的中国站和国际站使用不同账号体系。只让用户
                    # 选择服务区域，再由 Registry 同步 API 申请入口和 Base URL，
                    # 避免手工组合错误。已有空 Base URL 配置继续沿用中国站，只有
                    # 尚未填写 Key 的全新配置才根据界面语言推荐对应入口。
                    selected_service_endpoint = (
                        llm_provider_spec.select_service_endpoint(
                            configured_llm_base_url,
                            has_api_key=bool(str(llm_api_key).strip()),
                            prefer_international=(
                                st.session_state.get("ui_language", "en") != "zh"
                            ),
                        )
                    )
                    endpoint_options = [
                        endpoint.endpoint_id
                        for endpoint in llm_provider_spec.service_endpoints
                    ] + [CUSTOM_LLM_ENDPOINT_ID]
                    default_endpoint_id = (
                        selected_service_endpoint.endpoint_id
                        if selected_service_endpoint
                        else CUSTOM_LLM_ENDPOINT_ID
                    )
                    endpoint_labels = {
                        endpoint.endpoint_id: (
                            tr_optional(
                                llm_provider_spec.endpoint_label_key(
                                    endpoint.endpoint_id
                                ),
                                fallback_language="en",
                            )
                            or endpoint.default_label
                        )
                        for endpoint in llm_provider_spec.service_endpoints
                    }
                    endpoint_labels[CUSTOM_LLM_ENDPOINT_ID] = (
                        tr_optional("Custom API Endpoint", fallback_language="en")
                        or "Custom API Endpoint"
                    )
                    with llm_form_panel:
                        selected_endpoint_id = stable_selectbox(
                            tr_optional(
                                llm_provider_spec.endpoint_selector_label_key,
                                fallback_language="en",
                            )
                            or tr("API Platform"),
                            options=endpoint_options,
                            default_value=default_endpoint_id,
                            key=f"{llm_provider}_service_endpoint_select",
                            on_change=_save_settings_endpoint,
                            args=(
                                llm_provider_spec,
                                localized_widget_key(
                                    f"{llm_provider}_service_endpoint_select"
                                ),
                            ),
                            format_func=lambda endpoint_id: endpoint_labels[
                                endpoint_id
                            ],
                            help=(
                                tr_optional(
                                    llm_provider_spec.endpoint_selector_help_key,
                                    fallback_language="en",
                                )
                                or None
                            ),
                        )
                    selected_service_endpoint = next(
                        (
                            endpoint
                            for endpoint in llm_provider_spec.service_endpoints
                            if endpoint.endpoint_id == selected_endpoint_id
                        ),
                        None,
                    )
                    if selected_service_endpoint:
                        llm_base_url = selected_service_endpoint.base_url
                        provider_tip_context.update(
                            {
                                "api_key_url": selected_service_endpoint.api_key_url,
                                "default_base_url": selected_service_endpoint.base_url,
                                "model_docs_url": selected_service_endpoint.model_docs_url,
                            }
                        )
                    else:
                        # 自定义模式只保留用户明确保存的地址，不将某个标准区域伪装
                        # 成自定义值。输入为空时配置不会持久化，下一次仍回到兼容默认。
                        llm_base_url = str(configured_llm_base_url or "").strip()

                if llm_provider == "ollama":
                    llm_default_base_url = config.get_default_ollama_base_url()
                    if not llm_base_url:
                        llm_base_url = llm_default_base_url
                    docker_hint = ""
                    if config.is_running_in_container():
                        docker_hint = tr_optional(
                            "llm_provider_tips.ollama.docker_hint",
                            fallback_language="en",
                        )
                    provider_tip_context["docker_hint"] = docker_hint

                tips = get_llm_provider_tips(llm_provider, **provider_tip_context)
                if tips:
                    with llm_helper:
                        st.info(tips)

                st_llm_api_key = llm_api_key
                if llm_provider_spec.show_api_key:
                    st_llm_api_key = llm_form_panel.text_input(
                        tr("API Key"),
                        value=llm_api_key,
                        type="password",
                        key=f"{llm_provider}_api_key_input",
                        on_change=_save_settings_change,
                        args=(
                            "app",
                            llm_provider_spec.config_key("api_key"),
                            f"{llm_provider}_api_key_input",
                            lambda value: value,
                        ),
                    )

                st_llm_base_url = llm_base_url
                if llm_provider_spec.show_base_url:
                    st_llm_base_url = llm_form_panel.text_input(
                        tr("Base Url"),
                        value=llm_base_url,
                        key=(
                            f"{llm_provider}_base_url_"
                            f"{selected_service_endpoint.endpoint_id}_input"
                            if selected_service_endpoint
                            else f"{llm_provider}_base_url_custom_input"
                        ),
                        disabled=selected_service_endpoint is not None,
                        on_change=_save_settings_change,
                        args=(
                            "app",
                            llm_provider_spec.config_key("base_url"),
                            f"{llm_provider}_base_url_"
                            f"{selected_service_endpoint.endpoint_id}_input"
                            if selected_service_endpoint
                            else f"{llm_provider}_base_url_custom_input",
                            lambda value, llm_default_base_url=llm_default_base_url: (
                                normalize_provider_override(value, llm_default_base_url)
                            ),
                        ),
                    )
                st_llm_model_name = ""
                if llm_provider == "openai_codex":
                    from webui.codex_models import render_codex_model_selector

                    with llm_form_panel:
                        st_llm_model_name = render_codex_model_selector(
                            llm_model_name,
                            tr,
                            on_change=lambda model, provider=llm_provider_spec: (
                                _set_runtime_config(
                                    "app",
                                    provider.config_key("model_name"),
                                    normalize_provider_override(
                                        model, provider.default_model
                                    ),
                                )
                            ),
                        )
                elif llm_provider == "openai_chatgpt":
                    from webui.chatgpt_login import render_chatgpt_login
                    from webui.chatgpt_models import render_chatgpt_model_selector

                    with llm_form_panel:
                        if middle_config_panel.open:
                            # Keep the nested polling fragment's delta paths separate
                            # from the model selector and connection-test controls.
                            with st.container(key="chatgpt_login_panel"):
                                render_chatgpt_login(tr)
                        st_llm_model_name = render_chatgpt_model_selector(
                            llm_model_name,
                            tr,
                            on_change=lambda model, provider=llm_provider_spec: (
                                _set_runtime_config(
                                    "app",
                                    provider.config_key("model_name"),
                                    normalize_provider_override(
                                        model, provider.default_model
                                    ),
                                )
                            ),
                        )
                elif llm_provider == "groq":
                    effective_api_key = st_llm_api_key or llm_api_key
                    effective_base_url = st_llm_base_url or llm_base_url
                    groq_models = get_groq_model_ids(
                        api_key=effective_api_key,
                        base_url=effective_base_url,
                    )

                    if groq_models:
                        selected_index = 0
                        if llm_model_name in groq_models:
                            selected_index = groq_models.index(llm_model_name)

                        st_llm_model_name = llm_form_panel.selectbox(
                            tr("Model Name"),
                            options=groq_models,
                            index=selected_index,
                            key="groq_model_name_select",
                            on_change=_save_settings_change,
                            args=(
                                "app",
                                llm_provider_spec.config_key("model_name"),
                                "groq_model_name_select",
                                lambda value, llm_provider_spec=llm_provider_spec: (
                                    normalize_provider_override(
                                        value, llm_provider_spec.default_model
                                    )
                                ),
                            ),
                        )
                    else:
                        st_llm_model_name = llm_form_panel.text_input(
                            tr("Model Name"),
                            value=llm_model_name,
                            key="groq_model_name_input",
                            on_change=_save_settings_change,
                            args=(
                                "app",
                                llm_provider_spec.config_key("model_name"),
                                "groq_model_name_input",
                                lambda value, llm_provider_spec=llm_provider_spec: (
                                    normalize_provider_override(
                                        value, llm_provider_spec.default_model
                                    )
                                ),
                            ),
                        )
                        if effective_api_key:
                            llm_form_panel.caption(tr("Groq Model List Load Failed"))
                        else:
                            llm_form_panel.caption(
                                tr("Groq API Key Required for Model List")
                            )
                else:
                    st_llm_model_name = llm_form_panel.text_input(
                        tr("Model Name"),
                        value=llm_model_name,
                        key=f"{llm_provider}_model_name_input",
                        on_change=_save_settings_change,
                        args=(
                            "app",
                            llm_provider_spec.config_key("model_name"),
                            f"{llm_provider}_model_name_input",
                            lambda value, llm_provider_spec=llm_provider_spec: (
                                normalize_provider_override(
                                    value, llm_provider_spec.default_model
                                )
                            ),
                        ),
                    )
                # 输入框展示 Registry 默认值，但配置只保存真实的用户覆盖值。
                # 这样默认模型、Base URL 更新后，未自定义的用户能够自动跟随。
                _set_runtime_config(
                    "app",
                    llm_provider_spec.config_key("api_key"),
                    st_llm_api_key,
                )
                _set_runtime_config(
                    "app",
                    llm_provider_spec.config_key("base_url"),
                    normalize_provider_override(
                        st_llm_base_url,
                        llm_default_base_url,
                    ),
                )
                _set_runtime_config(
                    "app",
                    llm_provider_spec.config_key("model_name"),
                    normalize_provider_override(
                        st_llm_model_name,
                        llm_provider_spec.default_model,
                    ),
                )

                # Provider 专用字段也由 Registry 声明。例如 Cloudflare AI Gateway
                # 需要 Account ID；以后新增类似字段时无需再在 Main.py 增加判断。
                for field in llm_provider_spec.extra_fields:
                    field_config_key = llm_provider_spec.config_key(field.config_suffix)
                    field_value = llm_form_panel.text_input(
                        tr(field.label_key),
                        value=(
                            app_settings.get(field_config_key, "")
                            or field.default_value
                        ),
                        type="password" if field.secret else "default",
                        key=f"{llm_provider}_{field.config_suffix}_input",
                        on_change=_save_settings_change,
                        args=(
                            "app",
                            field_config_key,
                            f"{llm_provider}_{field.config_suffix}_input",
                            lambda value, field=field: normalize_provider_override(
                                value, field.default_value
                            ),
                        ),
                    )
                    _set_runtime_config(
                        "app",
                        field_config_key,
                        normalize_provider_override(
                            field_value,
                            field.default_value,
                        ),
                    )

                if llm_provider == "openai_codex" and middle_config_panel.open:
                    from webui.codex_login import render_codex_login

                    with llm_form_panel.container():
                        render_codex_login(tr)

                if llm_provider == "kimi_code" and middle_config_panel.open:
                    from app.services import kimi_code_oauth
                    from webui.kimi_login import render_kimi_login

                    # 给嵌套 fragment 独立的容器，轮询重绘不占用后续控件的位置。
                    with llm_form_panel.container():
                        render_kimi_login(
                            kimi_code_oauth.resolve_region(st_llm_base_url), tr
                        )

                if llm_form_panel.button(
                    tr("Test LLM Connection"),
                    key="test_llm_connection_button",
                    use_container_width=True,
                    type="secondary",
                    icon=":material/network_check:",
                ):
                    with config.try_runtime_config_lock() as lock_acquired:
                        if not lock_acquired:
                            llm_form_panel.warning(tr("Runtime Configuration Busy"))
                        else:
                            with llm_form_panel.spinner(tr("Testing LLM Connection")):
                                connection_ok, connection_error, connection_elapsed = (
                                    llm.test_connection()
                                )

                    if not lock_acquired:
                        connection_ok = None
                    elif connection_ok:
                        llm_form_panel.success(
                            tr("LLM Connection Test Succeeded").format(
                                provider=llm_provider_labels[llm_provider],
                                model=st_llm_model_name or "-",
                                elapsed=f"{connection_elapsed:.2f}",
                            )
                        )
                    else:
                        connection_error = format_llm_connection_error(
                            llm_provider,
                            st_llm_base_url,
                            connection_error,
                        )
                        llm_form_panel.error(
                            tr("LLM Connection Test Failed").format(
                                error=connection_error
                            )
                        )

        # 右侧面板 - API 密钥设置
        with right_config_panel:
            if right_config_panel.open:
                if "settings_material_group" not in st.session_state:
                    _initialize_material_navigation("material", app_settings)
                sections = (
                    (
                        "ai_video",
                        "AI Video Generation APIs",
                        "AI Video Generation APIs Help",
                    ),
                    (
                        "ai_image",
                        "AI Image Generation APIs",
                        "AI Image Generation APIs Help",
                    ),
                    ("stock_video", "Stock Video APIs", "Stock Video APIs Help"),
                )
                remembered = dict(
                    st.session_state.get("settings_material_providers", {})
                )
                for group, title_key, help_key in sections:
                    providers = VIDEO_SOURCE_GROUPS[group]
                    default_provider = remembered.get(group, providers[0])
                    if default_provider not in providers:
                        default_provider = providers[0]
                    provider_key = localized_widget_key(
                        f"settings_material_provider_{group}_select"
                    )
                    st.session_state[provider_key] = default_provider
                    with st.container(border=True):
                        st.subheader(tr(title_key))
                        st.caption(tr(help_key))
                        provider = st.selectbox(
                            tr(title_key),
                            options=providers,
                            key=provider_key,
                            format_func=_material_provider_labels().get,
                            label_visibility="collapsed",
                            on_change=_remember_material_provider,
                            args=(group, provider_key),
                        )
                        remembered[group] = provider
                        _render_material_provider_settings(provider)
                st.session_state["settings_material_providers"] = remembered

    _save_runtime_config()
