"""视频面板：素材来源、AI 视频服务商与画面参数。"""

import hashlib
import json
import math
import os

import streamlit as st

from app.config import config
from app.models.schema import (
    ImageMotion,
    VideoAspect,
    VideoConcatMode,
    VideoFitMode,
    VideoTransitionMode,
)
from app.services import (
    metaso_minimax,
    muapi,
    infistar_video,
    jalapeno_video,
    volcengine_seedance,
)
from app.utils import utils
from webui.common import (
    DEFAULT_VIDEO_CODEC_OPTION,
    LOCAL_MATERIAL_EXTENSIONS,
    OFOX_REFERRAL_URL,
    VIDEO_SOURCE_GROUPS,
    VOICE_MODE_UPLOAD,
    _delete_runtime_config,
    _saved_ui_choice,
    _saved_ui_number,
    _set_runtime_config,
    tr,
)
from webui.components.audio_panel import _estimate_voiceover_duration_range
from webui.components.script_panel import (
    _effective_loomloom_api_token,
    _load_loomloom_video_capability,
    _render_loomloom_video_settings,
)
from webui.components.top_bar import _open_material_settings_dialog
from webui.components.widgets import (
    _save_settings_change,
    grouped_selectbox,
    localized_widget_key,
    stable_selectbox,
)


def sync_script_order_concat_mode():
    """在文案顺序匹配开启时固定使用顺序拼接，并在关闭后恢复原选择。"""
    widget_key = localized_widget_key("video_concat_mode_select")
    previous_key = "video_concat_mode_before_script_order_match"
    match_script_order = bool(st.session_state.get("match_materials_to_script", False))

    if match_script_order:
        current_mode = st.session_state.get(widget_key, VideoConcatMode.random.value)
        if current_mode != VideoConcatMode.sequential.value:
            st.session_state[previous_key] = current_mode
        st.session_state[widget_key] = VideoConcatMode.sequential.value
        return

    previous_mode = st.session_state.pop(previous_key, None)
    if previous_mode in {
        VideoConcatMode.sequential.value,
        VideoConcatMode.random.value,
    }:
        st.session_state[widget_key] = previous_mode


def _render_video_settings(panel, params):
    """渲染视频设置并返回本次选择的本地素材。"""
    uploaded_files = []
    with panel:
        with st.container(border=True):
            st.write(tr("Video Settings"))
            video_concat_modes = [
                (tr("Sequential"), "sequential"),
                (tr("Random"), "random"),
            ]
            video_source_labels = {
                "pexels": tr("Pexels"),
                "pixabay": tr("Pixabay"),
                "coverr": tr("Coverr"),
                "wavespeed": tr("WaveSpeed AI Video"),
                "volcengine_seedance": tr("Volcano Engine Seedance"),
                "infistar": tr("Infistar AI Video"),
                "jalapeno": tr("Jalapeno Cloud AI Video"),
                "ofox": tr("OFox AI Video"),
                "metaso_minimax": tr("Metaso MiniMax H3"),
                "muapi": tr("MuAPI AI Video"),
                "loomloom": tr("Shengsuan Cloud AI Video"),
                "openai_image": tr("OpenAI Compatible Text-to-Image"),
                "codex_image": tr("codex.image.label"),
                "local": tr("Local file"),
            }
            saved_video_source_name = str(
                config.snapshot_config_with_pending(config.app).get("video_source", "pexels")
                or "pexels"
            )
            params.video_source = grouped_selectbox(
                tr("Video Source"),
                groups=(
                    (tr("Stock Video"), VIDEO_SOURCE_GROUPS["stock_video"]),
                    (tr("AI Video"), VIDEO_SOURCE_GROUPS["ai_video"]),
                    (tr("AI Image"), VIDEO_SOURCE_GROUPS["ai_image"]),
                    (tr("Local Material"), VIDEO_SOURCE_GROUPS["local"]),
                ),
                default_value=saved_video_source_name,
                key="video_source_select",
                format_func=video_source_labels.get,
                settings_label=tr("Configure Material Sources"),
                on_settings=_open_material_settings_dialog,
            )
            _set_runtime_config("app", "video_source", params.video_source)

            loomloom_video_capability = None
            if params.video_source == "loomloom":
                # 尽早读取缓存，使下方画面比例控件直接受当前 Profile 约束。
                # 首次输入 Key 后 Streamlit 会 rerun，此处随即加载一次。
                loomloom_video_capability = _load_loomloom_video_capability(
                    _effective_loomloom_api_token()
                )

            if params.video_source == "infistar":
                st.caption(tr("Infistar AI Video Help"))
                selected_infistar_model = _render_infistar_model_selector()
            if params.video_source == "jalapeno":
                st.caption(tr("Jalapeno Cloud Video Help"))
                snapshot = config.snapshot_config_with_pending(config.app)
                st.caption(tr("Jalapeno Cloud Video Specification").format(
                    model=str(snapshot.get("jalapeno_video_model", "") or jalapeno_video.DEFAULT_MODEL),
                    resolution=str(snapshot.get("jalapeno_video_resolution", "") or jalapeno_video.DEFAULT_RESOLUTION),
                ))
            if params.video_source == "wavespeed":
                st.caption(tr("WaveSpeed AI Video Help"))
            if params.video_source == "volcengine_seedance":
                st.caption(tr("Volcano Engine Seedance Help"))
            if params.video_source == "ofox":
                st.caption(f"[OfoxAI]({OFOX_REFERRAL_URL}) · {tr('OFox AI Video Help')}")
            if params.video_source == "metaso_minimax":
                st.caption(tr("Metaso MiniMax H3 Help"))
            if params.video_source == "muapi":
                st.caption(tr("MuAPI AI Video Help"))
            if params.video_source == "local":
                # Streamlit 的文件类型校验对扩展名大小写敏感，这里同时放行大小写两种形式。
                local_file_types = sorted(
                    extension.removeprefix(".")
                    for extension in LOCAL_MATERIAL_EXTENSIONS
                )
                uploaded_files = st.file_uploader(
                    tr("Upload Local Files"),
                    type=local_file_types
                    + [file_type.upper() for file_type in local_file_types],
                    accept_multiple_files=True,
                    key="local_video_materials_uploader",
                )

            # 文案顺序匹配会从关键词生成到最终合成全程保持叙事顺序，因此开启时
            # 顺序拼接是唯一符合实际执行逻辑的选项。同步控件值可避免界面仍显示
            # “随机拼接”，同时保留用户原选择，关闭后自动恢复。
            sync_script_order_concat_mode()
            selected_concat_mode = stable_selectbox(
                tr("Video Concat Mode"),
                options=[value for _, value in video_concat_modes],
                default_value=_saved_ui_choice(
                    "video_concat_mode",
                    [value for _, value in video_concat_modes],
                    VideoConcatMode.random.value,
                ),
                key="video_concat_mode_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_concat_modes
                )[value],
                disabled=bool(st.session_state.get("match_materials_to_script", False)),
            )
            params.video_concat_mode = VideoConcatMode(selected_concat_mode)

            params.match_materials_to_script = st.checkbox(
                tr("Match Materials to Script Order"),
                help=tr("Match Materials to Script Order Help"),
                key="match_materials_to_script",
                on_change=sync_script_order_concat_mode,
            )
            _set_runtime_config(
                "app",
                "match_materials_to_script",
                params.match_materials_to_script,
            )
            # 顺序匹配开启时，sequential 是派生出的强制值，不应覆盖用户在关闭
            # 该功能时选择的拼接偏好；关闭后仍能恢复此前的 random/sequential。
            if not params.match_materials_to_script:
                _set_runtime_config(
                    "ui", "video_concat_mode", params.video_concat_mode.value
                )

            # 视频转场模式
            video_transition_modes = [
                (tr("None"), VideoTransitionMode.none.value),
                (tr("Shuffle"), VideoTransitionMode.shuffle.value),
                (tr("FadeIn"), VideoTransitionMode.fade_in.value),
                (tr("FadeOut"), VideoTransitionMode.fade_out.value),
                (tr("SlideIn"), VideoTransitionMode.slide_in.value),
                (tr("SlideOut"), VideoTransitionMode.slide_out.value),
                (tr("ZoomIn"), VideoTransitionMode.zoom_in.value),
                (tr("ZoomOut"), VideoTransitionMode.zoom_out.value),
            ]
            selected_transition_mode = stable_selectbox(
                tr("Video Transition Mode"),
                options=[value for _, value in video_transition_modes],
                default_value=_saved_ui_choice(
                    "video_transition_mode",
                    [value for _, value in video_transition_modes],
                    VideoTransitionMode.none.value,
                ),
                key="video_transition_mode_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_transition_modes
                )[value],
            )
            params.video_transition_mode = VideoTransitionMode(selected_transition_mode)
            _set_runtime_config(
                "ui",
                "video_transition_mode",
                params.video_transition_mode.value,
            )

            video_aspect_ratios = [
                (tr("Portrait"), VideoAspect.portrait.value),
                (tr("Landscape"), VideoAspect.landscape.value),
            ]
            if loomloom_video_capability is not None:
                ratio_labels = {value: label for label, value in video_aspect_ratios}
                video_aspect_ratios = [
                    (ratio_labels[value], value)
                    for value in loomloom_video_capability.aspect_ratios
                ]
            # Coverr 库 99% 是 16:9 横屏,默认竖屏会让画面被大量黑边包围。
            # 用 source-specific widget key 让每个 source 各自记忆 aspect 选择:
            #   - 首次切到 coverr → 默认 Landscape(index=1)
            #   - 其他 source 沿用 Portrait(index=0)
            #   - 用户在某 source 下手动改过 aspect,session_state 会记住,
            #     下次回到同一 source 时尊重用户选择,不会再被强制覆盖。
            default_aspect_index = 1 if params.video_source == "coverr" else 0
            video_aspect_values = [value for _, value in video_aspect_ratios]
            video_aspect_config_key = f"video_aspect_{params.video_source}"
            selected_aspect_ratio = stable_selectbox(
                tr("Video Ratio"),
                options=video_aspect_values,
                default_value=_saved_ui_choice(
                    video_aspect_config_key,
                    video_aspect_values,
                    video_aspect_ratios[default_aspect_index][1],
                ),
                key=f"video_aspect_for_{params.video_source}",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_aspect_ratios
                )[value],
            )
            params.video_aspect = VideoAspect(selected_aspect_ratio)
            _set_runtime_config(
                "ui", video_aspect_config_key, params.video_aspect.value
            )

            video_fit_modes = [
                (tr("Fill and Crop"), VideoFitMode.cover.value),
                (tr("Fit with Black Bars"), VideoFitMode.contain.value),
            ]
            selected_fit_mode = stable_selectbox(
                tr("Video Fit Mode"),
                options=[value for _, value in video_fit_modes],
                default_value=_saved_ui_choice(
                    "video_fit_mode",
                    [value for _, value in video_fit_modes],
                    VideoFitMode.cover.value,
                ),
                key="video_fit_mode_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_fit_modes
                )[value],
                help=tr("Video Fit Mode Help"),
            )
            params.video_fit_mode = VideoFitMode(selected_fit_mode)
            _set_runtime_config(
                "ui", "video_fit_mode", params.video_fit_mode.value
            )

            if params.video_source in {"local", "openai_image", "codex_image"}:
                image_motion_labels = {
                    ImageMotion.static.value: tr("Static Image"),
                    ImageMotion.zoom.value: tr("Slow Zoom"),
                }
                params.image_motion = ImageMotion(
                    stable_selectbox(
                        tr("Image Motion"),
                        options=list(image_motion_labels),
                        default_value=_saved_ui_choice(
                            "image_motion", list(image_motion_labels), ImageMotion.zoom.value,
                            # 生成任务持锁时，切换语言或素材来源会重建控件；
                            # 此时必须恢复尚未落盘的最新选择，不能写回旧配置值。
                            section=config.snapshot_config_with_pending(config.ui),
                        ),
                        key="image_motion_select",
                        format_func=image_motion_labels.get,
                        help=tr("Image Motion Help"),
                    )
                )
                _set_runtime_config("ui", "image_motion", params.image_motion.value)

            # MiniMax H3 的远端时长范围是 4～15 秒。选择秘塔时使用完整能力
            # 范围，既避免 2/3 秒被按 4 秒计费，也让 WebUI 与 CLI、服务层一致。
            if params.video_source == "infistar":
                # 每种模型先提供实测的固定规格；不把被 API 拒绝的时长发往付费接口。
                video_clip_durations = [infistar_video.MODELS.get(selected_infistar_model, {"seconds": 5})["seconds"]]
            elif params.video_source == "metaso_minimax":
                video_clip_durations = list(
                    range(
                        metaso_minimax.DEFAULT_MIN_DURATION_SECONDS,
                        metaso_minimax.DEFAULT_MAX_DURATION_SECONDS + 1,
                    )
                )
            elif params.video_source == "muapi":
                video_clip_durations = list(
                    range(
                        muapi.DEFAULT_MIN_DURATION_SECONDS,
                        muapi.DEFAULT_MAX_DURATION_SECONDS + 1,
                    )
                )
            else:
                video_clip_durations = [2, 3, 4, 5, 6, 7, 8, 9, 10]
            params.video_clip_duration = stable_selectbox(
                tr("Clip Duration"),
                options=video_clip_durations,
                default_value=_saved_ui_choice(
                    "video_clip_duration",
                    video_clip_durations,
                    5
                    if params.video_source in {"metaso_minimax", "muapi"}
                    else 3,
                ),
                key="video_clip_duration_select",
                help=tr("Clip Duration Help"),
            )
            _set_runtime_config(
                "ui", "video_clip_duration", params.video_clip_duration
            )
            clip_speed_key = localized_widget_key("video_clip_speed_slider")
            # session_state 可能来自旧任务、API 参数或旧版页面状态。控件创建前
            # 统一归一化，既保留合法选择，也确保 slider 始终收到 0.5～2.0
            # 范围内的有限浮点数。
            st.session_state[clip_speed_key] = utils.normalize_clip_speed(
                st.session_state.get(
                    clip_speed_key,
                    _saved_ui_number("video_clip_speed", 1.0, 0.5, 2.0),
                )
            )
            params.video_clip_speed = st.slider(
                tr("Clip Speed"),
                min_value=0.5,
                max_value=2.0,
                step=0.05,
                format="%.2fx",
                key=clip_speed_key,
                help=tr("Clip Speed Help"),
            )
            _set_runtime_config("ui", "video_clip_speed", params.video_clip_speed)
            video_count_options = [1, 2, 3, 4, 5]
            params.video_count = stable_selectbox(
                tr("Number of Videos Generated Simultaneously"),
                options=video_count_options,
                default_value=_saved_ui_choice(
                    "video_count", video_count_options, 1
                ),
                key="video_count_select",
            )
            _set_runtime_config("ui", "video_count", params.video_count)

            video_codec_options = [
                (tr("Default Video Encoder"), DEFAULT_VIDEO_CODEC_OPTION),
                ("libx264 (CPU)", "libx264"),
                ("NVIDIA NVENC (h264_nvenc)", "h264_nvenc"),
                ("AMD AMF (h264_amf)", "h264_amf"),
                ("Intel QSV (h264_qsv)", "h264_qsv"),
                ("Windows MediaFoundation (h264_mf)", "h264_mf"),
                ("macOS VideoToolbox (h264_videotoolbox)", "h264_videotoolbox"),
            ]
            saved_video_codec = config.app.get(
                "video_codec", DEFAULT_VIDEO_CODEC_OPTION
            )
            saved_video_codec_values = [item[1] for item in video_codec_options]
            if saved_video_codec not in saved_video_codec_values:
                # 旧版本或手工配置可能留下无效值。UI 回到“默认”而不是替用户
                # 固定某个编码器，后端仍会按稳定策略解析为 libx264。
                saved_video_codec = DEFAULT_VIDEO_CODEC_OPTION
            selected_video_codec = stable_selectbox(
                tr("Video Encoder"),
                options=saved_video_codec_values,
                default_value=saved_video_codec,
                key="video_encoder_select",
                format_func=lambda value: dict(
                    (v, label) for label, v in video_codec_options
                )[value],
                help=tr("Video Encoder Help"),
            )
            if selected_video_codec == DEFAULT_VIDEO_CODEC_OPTION:
                # 默认模式不持久化具体编码器，让配置表达“跟随项目默认值”。
                _delete_runtime_config("app", "video_codec")
            else:
                _set_runtime_config("app", "video_codec", selected_video_codec)

            concurrency_options = [1, 2, 4, 6, 8]
            if params.video_source in {"pexels", "pixabay", "coverr"}:
                selected_material_concurrency = stable_selectbox(
                    tr("Material Concurrency"),
                    options=concurrency_options,
                    default_value=_saved_ui_choice(
                        "material_concurrency", concurrency_options, 1, config.app
                    ),
                    key="material_concurrency_select",
                    help=tr("Material Concurrency Help"),
                )
                _set_runtime_config(
                    "app", "material_concurrency", selected_material_concurrency
                )

            selected_clip_concurrency = stable_selectbox(
                tr("Clip Rendering Concurrency"),
                options=concurrency_options,
                default_value=_saved_ui_choice(
                    "video_clip_concurrency", concurrency_options, 1, config.app
                ),
                key="clip_rendering_concurrency_select",
                help=tr("Clip Rendering Concurrency Help"),
            )
            _set_runtime_config(
                "app", "video_clip_concurrency", selected_clip_concurrency
            )

            if params.video_source == "loomloom":
                _render_loomloom_video_settings(params)

            if params.video_source == "wavespeed":
                _render_wavespeed_video_settings(params)
            if params.video_source == "volcengine_seedance":
                _render_seedance_video_settings(params)
            if params.video_source == "infistar":
                st.warning(tr("Infistar Video Billing Notice"))
                st.checkbox(tr("Confirm Infistar Charge"), key="infistar_confirm_charge")
            if params.video_source == "jalapeno":
                _sync_jalapeno_charge_confirmation(params)
                st.warning(tr("Jalapeno Cloud Video Billing Notice"))
                st.checkbox(tr("Confirm Jalapeno Cloud Charge"), key="jalapeno_confirm_charge")
            if params.video_source == "ofox":
                _render_ofox_video_settings(params)
            if params.video_source == "metaso_minimax":
                _render_metaso_minimax_video_settings(params)
            if params.video_source == "muapi":
                _render_muapi_video_settings(params)
    return uploaded_files


def _render_wavespeed_video_settings(params):
    """
    渲染 WaveSpeed 生成数量估算与计费确认。

    生成按条计费，提交前必须让用户看到大致会生成多少段。估算完全在本地
    完成：用配音时长估算区间除以片段时长得到需要覆盖的片段数。素材流程
    本身按需逐段生成、凑够所需时长即停，因此实际生成数以运行时为准，
    估算只用于量级提示，不参与任务执行。
    """
    clip_duration = max(int(params.video_clip_duration or 1), 1)
    video_count = max(int(params.video_count or 1), 1)
    estimated_range = _estimate_voiceover_duration_range(
        str(params.video_script or ""),
        params.voice_rate,
    )
    if estimated_range:
        min_clips = max(math.ceil(estimated_range[0] * video_count / clip_duration), 1)
        max_clips = max(
            math.ceil(estimated_range[1] * video_count / clip_duration), min_clips
        )
        st.warning(
            tr("WaveSpeed Billing Notice").format(min=min_clips, max=max_clips)
        )
    else:
        st.warning(tr("WaveSpeed Billing Notice Without Script"))
    st.checkbox(
        tr("Confirm WaveSpeed Charge"),
        key="wavespeed_confirm_charge",
        help=tr("Confirm WaveSpeed Charge Help"),
    )


def _render_seedance_video_settings(params):
    """展示预计付费任务数量，并要求用户明确确认方舟生成费用。"""
    clip_duration = max(int(params.video_clip_duration or 1), 1)
    video_count = max(int(params.video_count or 1), 1)
    estimated_range = _estimate_voiceover_duration_range(
        str(params.video_script or ""), params.voice_rate
    )
    if estimated_range:
        min_clips = max(math.ceil(estimated_range[0] * video_count / clip_duration), 1)
        max_clips = max(
            math.ceil(estimated_range[1] * video_count / clip_duration), min_clips
        )
        st.warning(
            tr("Volcano Engine Seedance Billing Notice").format(
                min=min_clips, max=max_clips
            )
        )
    else:
        st.warning(tr("Volcano Engine Seedance Billing Notice Without Script"))
    st.checkbox(
        tr("Confirm Volcano Engine Seedance Charge"),
        key="volcengine_seedance_confirm_charge",
        help=tr("Confirm Volcano Engine Seedance Charge Help"),
    )


def _render_jalapeno_provider_settings():
    """默认值直接填入；独立视频配置不依赖当前选择的大模型 Provider。"""
    snapshot = config.snapshot_config_with_pending(config.app)
    st.markdown(f"**{tr('Jalapeno Cloud AI Video')}**")
    key = st.text_input(
        tr("Jalapeno Cloud Video API Key"),
        value=str(snapshot.get("jalapeno_video_api_key", "") or ""),
        type="password",
        key="jalapeno_video_api_key_input",
        help=tr("Jalapeno Cloud Video Key Help"),
        on_change=_save_settings_change,
        args=(
            "app",
            "jalapeno_video_api_key",
            "jalapeno_video_api_key_input",
            lambda value: value.strip(),
        ),
    ).strip()
    _set_runtime_config("app", "jalapeno_video_api_key", key)
    if (
        not key
        and not os.getenv("JALAPENO_VIDEO_API_KEY", "").strip()
        and str(snapshot.get("jalapeno_api_key", "") or "").strip()
    ):
        st.caption(tr("Jalapeno Cloud Video Key Reused"))
    model = st.text_input(
        tr("Jalapeno Cloud Video Model"),
        value=str(
            snapshot.get("jalapeno_video_model", "") or jalapeno_video.DEFAULT_MODEL
        ),
        key="jalapeno_video_model_input",
        on_change=_save_settings_change,
        args=(
            "app",
            "jalapeno_video_model",
            "jalapeno_video_model_input",
            lambda value: value.strip(),
        ),
    ).strip()
    _set_runtime_config("app", "jalapeno_video_model", model)
    base = st.text_input(
        tr("Jalapeno Cloud Video Base URL"),
        value=str(
            snapshot.get("jalapeno_video_base_url", "")
            or jalapeno_video.DEFAULT_TASKS_URL
        ),
        help=tr("Jalapeno Cloud Video Base URL Help"),
        key="jalapeno_video_base_url_input",
        on_change=_save_settings_change,
        args=(
            "app",
            "jalapeno_video_base_url",
            "jalapeno_video_base_url_input",
            lambda value: value.strip(),
        ),
    ).strip()
    _set_runtime_config("app", "jalapeno_video_base_url", base)
    resolution = stable_selectbox(
        tr("Jalapeno Cloud Video Resolution"),
        options=sorted(
            volcengine_seedance.SUPPORTED_RESOLUTIONS, key=lambda value: int(value[:-1])
        ),
        default_value=str(
            snapshot.get("jalapeno_video_resolution", "")
            or jalapeno_video.DEFAULT_RESOLUTION
        ),
        key="jalapeno_video_resolution_select",
        on_change=_save_settings_change,
        args=(
            "app",
            "jalapeno_video_resolution",
            localized_widget_key("jalapeno_video_resolution_select"),
            lambda value: value,
        ),
    )
    _set_runtime_config("app", "jalapeno_video_resolution", resolution)


def _sync_jalapeno_charge_confirmation(params):
    """接入地址、模型、规格或有效 Key 变化后重新确认，避免沿用过期授权。"""
    snapshot = config.snapshot_config_with_pending(config.app)
    signature = _jalapeno_charge_signature(params, snapshot)
    if st.session_state.get("jalapeno_charge_signature") != signature:
        st.session_state["jalapeno_confirm_charge"] = False
        st.session_state["jalapeno_charge_signature"] = signature


def _jalapeno_charge_signature(params, snapshot):
    """使用同一快照检查费用确认，避免渲染与点击提交之间读取不同配置。"""
    return hashlib.sha256(json.dumps([
        jalapeno_video.get_api_key(snapshot),
        snapshot.get("jalapeno_video_base_url", ""),
        snapshot.get("jalapeno_video_model", ""),
        snapshot.get("jalapeno_video_resolution", ""),
        str(params.video_aspect), params.video_clip_duration, params.video_count,
    ]).encode()).hexdigest()


def _render_infistar_model_selector():
    """模型顺序固定为 Seedance、MiniMax、可灵；切换模型后重新确认费用。"""
    snapshot = config.snapshot_config_with_pending(config.app)
    configured = str(snapshot.get("infistar_video_model", "") or infistar_video.DEFAULT_MODEL)
    options = list(infistar_video.MODELS)
    if configured not in options:
        options.append(configured)
        st.error(tr("Infistar Video Invalid Model"))
    model = stable_selectbox(
        tr("Infistar Video Model"), options=options,
        default_value=configured, key="infistar_video_model_select",
        format_func=lambda value: infistar_video.MODELS.get(value, {"label": value})["label"],
    )
    if st.session_state.get("infistar_previous_video_model") != model:
        st.session_state["infistar_confirm_charge"] = False
        st.session_state["infistar_previous_video_model"] = model
    _set_runtime_config("app", "infistar_video_model", model)
    profile = infistar_video.MODELS.get(model)
    if profile:
        st.caption(tr("Infistar Video Specification").format(seconds=profile["seconds"], size=profile["size"]))
        if model == "MiniMax-H3":
            st.info(tr("Infistar MiniMax Landscape Help"))
    return model


def _render_ofox_video_settings(params):
    """展示预计付费任务数量，并要求用户明确确认 OFox 生成费用。"""
    clip_duration = max(int(params.video_clip_duration or 1), 1)
    video_count = max(int(params.video_count or 1), 1)
    estimated_range = _estimate_voiceover_duration_range(
        str(params.video_script or ""), params.voice_rate
    )
    if estimated_range:
        min_clips = max(math.ceil(estimated_range[0] * video_count / clip_duration), 1)
        max_clips = max(
            math.ceil(estimated_range[1] * video_count / clip_duration), min_clips
        )
        st.warning(
            tr("OFox Billing Notice").format(min=min_clips, max=max_clips)
        )
    else:
        st.warning(tr("OFox Billing Notice Without Script"))
    st.checkbox(
        tr("Confirm OFox Charge"),
        key="ofox_confirm_charge",
        help=tr("Confirm OFox Charge Help"),
    )


def _render_metaso_minimax_video_settings(params):
    """展示预计付费任务数量，并要求用户确认秘塔 MiniMax 生成费用。"""
    clip_duration = max(int(params.video_clip_duration or 1), 1)
    video_count = max(int(params.video_count or 1), 1)
    voice_mode = st.session_state.get(
        localized_widget_key("voice_mode_control"),
        config.ui.get("voice_mode"),
    )
    if voice_mode == VOICE_MODE_UPLOAD:
        # 视频设置渲染在音频设置之前，此时无法可靠读取本轮新上传文件的实际
        # 时长。上传模式不再展示按脚本文字推算的数字，避免用户误以为一个
        # 5 秒音频也会按较长文案创建多个付费任务；运行时仍以文件真实时长为准。
        st.warning(
            tr("Metaso MiniMax Billing Notice Uploaded Audio").format(
                resolution=str(
                    config.app.get(
                        "metaso_minimax_resolution",
                        metaso_minimax.DEFAULT_RESOLUTION,
                    )
                    or metaso_minimax.DEFAULT_RESOLUTION
                ),
                duration=clip_duration,
                count=video_count,
            )
        )
    elif estimated_range := _estimate_voiceover_duration_range(
        str(params.video_script or ""), params.voice_rate
    ):
        min_clips = max(math.ceil(estimated_range[0] * video_count / clip_duration), 1)
        max_clips = max(
            math.ceil(estimated_range[1] * video_count / clip_duration), min_clips
        )
        st.warning(
            tr("Metaso MiniMax Billing Notice").format(
                min=min_clips,
                max=max_clips,
                resolution=str(
                    config.app.get(
                        "metaso_minimax_resolution",
                        metaso_minimax.DEFAULT_RESOLUTION,
                    )
                    or metaso_minimax.DEFAULT_RESOLUTION
                ),
            )
        )
    else:
        st.warning(tr("Metaso MiniMax Billing Notice Without Script"))
    st.checkbox(
        tr("Confirm Metaso MiniMax Charge"),
        key="metaso_minimax_confirm_charge",
        help=tr("Confirm Metaso MiniMax Charge Help"),
    )


def _render_muapi_video_settings(params):
    """Show an estimated MuAPI task count and require explicit billing consent."""
    clip_duration = max(int(params.video_clip_duration or 1), 1)
    video_count = max(int(params.video_count or 1), 1)
    if estimated_range := _estimate_voiceover_duration_range(
        str(params.video_script or ""), params.voice_rate
    ):
        min_clips = max(math.ceil(estimated_range[0] * video_count / clip_duration), 1)
        max_clips = max(
            math.ceil(estimated_range[1] * video_count / clip_duration), min_clips
        )
        st.warning(
            tr("MuAPI Billing Notice").format(min=min_clips, max=max_clips)
        )
    else:
        st.warning(tr("MuAPI Billing Notice Without Script"))
    st.checkbox(
        tr("Confirm MuAPI Charge"),
        key="muapi_confirm_charge",
        help=tr("Confirm MuAPI Charge Help"),
    )
