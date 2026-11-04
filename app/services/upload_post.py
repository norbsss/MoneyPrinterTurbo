"""
Upload-Post API integration for cross-posting videos to TikTok, Instagram and YouTube Shorts.

Docs: https://docs.upload-post.com
"""
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional
from uuid import uuid4

import requests
from loguru import logger

from app.utils.request_logging import trace_http_call
from app.config import config


_UPLOAD_STATUS_POLL_INTERVAL_SECONDS = 10
_UPLOAD_STATUS_TIMEOUT_SECONDS = 3600
_MAX_CONSECUTIVE_STATUS_ERRORS = 3
# Upload-Post 按平台读取的发布文案字段；缺失时回退到通用 title。
# https://docs.upload-post.com/api/upload-video
_PLATFORM_TITLE_FIELDS = {"tiktok": "tiktok_title", "instagram": "instagram_title"}
_PLATFORM_TEXT_LIMIT = 2200
SCHEDULE_NOW = "now"
SCHEDULE_QUEUE = "queue"
SCHEDULE_DELAY = "delay"
SCHEDULE_MODES = (SCHEDULE_NOW, SCHEDULE_QUEUE, SCHEDULE_DELAY)
# Upload-Post 只接受未来 365 天内的 scheduled_date。
_MAX_SCHEDULE_DELAY_HOURS = 24 * 365
_DEFAULT_SCHEDULE_DELAY_HOURS = 2.0


class UploadPostService:
    API_BASE = "https://api.upload-post.com"

    def __init__(self, account: dict | None = None):
        # A background publish must keep its account across queueing and polling.
        # Only the in-memory job carries this snapshot; it is never task metadata.
        self._account = dict(account) if account is not None else None

    def snapshot_account(self) -> dict:
        settings = dict(config.app if self._account is None else self._account)
        return {
            "upload_post_api_key": settings.get("upload_post_api_key", ""),
            "upload_post_username": settings.get("upload_post_username", ""),
            "upload_post_enabled": settings.get("upload_post_enabled", False),
        }

    def _account_setting(self, key: str, default):
        settings = config.app if self._account is None else self._account
        return settings.get(key, default)

    @staticmethod
    def _with_platform_outcome(result: dict, expected_platforms: list | None = None) -> dict:
        """A successful API request can still contain failed platform publishes."""
        platform_results = result.get("results")
        if isinstance(platform_results, dict):
            entries = list(platform_results.items())
        elif isinstance(platform_results, list):
            entries = [
                (
                    entry.get("platform", "unknown")
                    if isinstance(entry, dict)
                    else "unknown",
                    entry,
                )
                for entry in platform_results
            ]
        else:
            if "results" in result:
                return {
                    **result,
                    "success": False,
                    "error": "Upload-Post returned invalid platform results",
                }
            return result

        failures = [
            str(platform)
            for platform, entry in entries
            if not isinstance(entry, dict)
            or entry.get("success") is not True
            or entry.get("skipped") is True
        ]
        reported_platforms = {platform for platform, _ in entries if isinstance(platform, str)}
        failures.extend(
            f"{platform} (missing result)"
            for platform in dict.fromkeys(expected_platforms or [])
            if platform not in reported_platforms
        )
        if not platform_results:
            failures.append("no platform results")
        if failures:
            return {
                **result,
                "success": False,
                "error": "Upload-Post failed or skipped platforms: "
                + ", ".join(failures),
            }
        return result

    def _wait_for_upload_completion(self, request_id: str, expected_platforms: list | None = None) -> dict:
        """Resolve Upload-Post's automatic sync-to-background fallback."""
        consecutive_errors = 0
        deadline = time.monotonic() + _UPLOAD_STATUS_TIMEOUT_SECONDS
        while True:
            status_result = self.check_status(request_id)
            status = status_result.get("status")
            if status == "completed":
                if not isinstance(status_result.get("results"), (dict, list)):
                    return {
                        **status_result,
                        "request_id": request_id,
                        "success": False,
                        "error": "Upload-Post completed without platform results",
                    }
                return self._with_platform_outcome(
                    {**status_result, "request_id": request_id, "success": True},
                    expected_platforms,
                )
            if status == "failed":
                return {
                    **status_result,
                    "request_id": request_id,
                    "success": False,
                    "error": status_result.get("message")
                    or "Upload-Post background upload failed",
                }
            if status in {"pending", "queued", "processing", "in_progress"}:
                consecutive_errors = 0
            else:
                consecutive_errors += 1
                if consecutive_errors >= _MAX_CONSECUTIVE_STATUS_ERRORS:
                    return {
                        "success": False,
                        "request_id": request_id,
                        "error": "Upload-Post status could not be confirmed; "
                        f"check request_id {request_id}",
                    }
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(_UPLOAD_STATUS_POLL_INTERVAL_SECONDS, remaining))
        return {
            "success": False,
            "request_id": request_id,
            "error": "Upload-Post status did not complete within 1 hour; "
            f"check request_id {request_id}",
        }

    @property
    def api_key(self) -> str:
        return self._account_setting("upload_post_api_key", "")

    @property
    def username(self) -> str:
        return self._account_setting("upload_post_username", "")

    @property
    def enabled(self) -> bool:
        return self._account_setting("upload_post_enabled", False)

    @property
    def platforms(self) -> list:
        return config.app.get("upload_post_platforms", ["tiktok", "instagram"])

    @property
    def auto_upload(self) -> bool:
        return config.app.get("upload_post_auto_upload", False)

    @property
    def youtube_privacy_status(self) -> str:
        return config.app.get("upload_post_youtube_privacy_status", "public")

    @property
    def schedule_mode(self) -> str:
        mode = str(config.app.get("upload_post_schedule", SCHEDULE_NOW) or SCHEDULE_NOW)
        return mode if mode in SCHEDULE_MODES else SCHEDULE_NOW

    @property
    def schedule_delay_hours(self) -> float:
        try:
            hours = float(
                config.app.get("upload_post_schedule_delay_hours", _DEFAULT_SCHEDULE_DELAY_HOURS)
            )
        except (TypeError, ValueError):
            hours = _DEFAULT_SCHEDULE_DELAY_HOURS
        return min(max(hours, 0.1), _MAX_SCHEDULE_DELAY_HOURS)

    def schedule_request(self, now: datetime | None = None) -> dict | None:
        """发布时间的快照，在视频排队发布时确定。

        延迟发布从排队时刻起算，之后修改设置不影响已排队的视频，与 YouTube
        隐私和受众声明的快照语义一致。返回 None 表示立即发布。
        """
        mode = self.schedule_mode
        if mode == SCHEDULE_QUEUE:
            return {"mode": SCHEDULE_QUEUE}
        if mode == SCHEDULE_DELAY:
            when = (now or datetime.now(timezone.utc)) + timedelta(hours=self.schedule_delay_hours)
            return {
                "mode": SCHEDULE_DELAY,
                "scheduled_date": when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        return None

    @property
    def ai_label(self) -> bool:
        # TikTok 与 Instagram 要求标注 AI 生成内容，与 YouTube 的
        # containsSyntheticMedia 一致默认开启。Instagram 的标注发布后无法撤销。
        return config.app.get("upload_post_ai_label", True) is True

    @property
    def youtube_made_for_kids(self) -> bool:
        # 未配置时，后续 YouTube 上传显式声明为非面向儿童；儿童内容需设置为 true。
        # 这与旧版省略声明字段不同，但不会修改已上传的视频。
        # 上传前严格校验类型，避免将 TOML 字符串 "false" 当作真值而误报受众。
        return config.app.get("upload_post_youtube_made_for_kids", False)

    def is_configured(self) -> bool:
        return bool(self.api_key and self.username and self.enabled)

    def upload_video(
        self,
        video_path: str,
        title: str,
        platforms: Optional[list] = None,
        privacy_level: str = "PUBLIC_TO_EVERYONE",
        youtube_extra: Optional[dict] = None,
        on_background_start: Callable[[str], None] | None = None,
        platform_titles: Optional[dict] = None,
        schedule: Optional[dict] = None,
    ) -> dict:
        if not self.is_configured():
            logger.warning("Upload-Post is not configured. Skipping cross-post.")
            return {"success": False, "error": "Upload-Post not configured"}

        if platforms is None:
            platforms = self.platforms

        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return {"success": False, "error": f"Video file not found: {video_path}"}

        has_youtube = any(p.startswith("youtube") for p in platforms)
        if has_youtube:
            # 后台任务传入排队时的声明快照；直接调用服务时才读取当前配置。
            made_for_kids = (youtube_extra or {}).get(
                "selfDeclaredMadeForKids", self.youtube_made_for_kids
            )
            if not isinstance(made_for_kids, bool):
                error = "YouTube made-for-kids setting must be a boolean"
                logger.error(error)
                return {"success": False, "error": error}

        logger.info(f"Cross-posting video to {', '.join(platforms)} via Upload-Post...")

        # Generate the remote handle before POST: a lost response does not
        # prove that Upload-Post stopped publishing the received video.
        client_request_id = str(uuid4())
        def unconfirmed_response(message: str) -> dict:
            return {
                "success": False,
                "request_id": client_request_id,
                "error": (
                    f"{message}; upload outcome is unconfirmed; "
                    f"check request_id {client_request_id} before submitting again"
                ),
            }

        try:
            with open(video_path, 'rb') as video_file:
                files = {'video': video_file}

                data = [
                    ('user', self.username),
                    ('request_id', client_request_id),
                    ('title', title[:2200]),
                    ('privacy_level', privacy_level),
                ]

                for platform in platforms:
                    data.append(('platform[]', platform))

                # 每个平台各自的文案（含话题标签），未提供时 Upload-Post 使用 title。
                for platform, text in (platform_titles or {}).items():
                    field = _PLATFORM_TITLE_FIELDS.get(platform)
                    if field and platform in platforms and str(text or "").strip():
                        data.append((field, str(text)[:_PLATFORM_TEXT_LIMIT]))

                if self.ai_label and any(p in _PLATFORM_TITLE_FIELDS for p in platforms):
                    data.append(('is_ai_generated', "true"))

                # Upload-Post 自己保存定时任务并按时发布，本地进程退出也不受影响。
                if schedule and schedule.get("mode") == SCHEDULE_QUEUE:
                    data.append(('add_to_queue', "true"))
                elif schedule and schedule.get("scheduled_date"):
                    data.append(('scheduled_date', schedule["scheduled_date"]))

                if has_youtube:
                    # multipart 表单使用小写布尔字符串，且不能依赖 LLM 元数据
                    # 是否存在；只要发布到 YouTube，就显式传递用户的受众声明。
                    data.append(('selfDeclaredMadeForKids', str(made_for_kids).lower()))
                    # Privacy is an account/user setting, independent of optional
                    # generated titles and descriptions. Preserve queued overrides.
                    data.append(('privacyStatus', (youtube_extra or {}).get(
                        "privacyStatus", self.youtube_privacy_status
                    )))
                    logger.info(f"YouTube audience declaration: made_for_kids={made_for_kids}")

                if youtube_extra and has_youtube:
                    if "youtube_title" in youtube_extra:
                        data.append(('youtube_title', youtube_extra["youtube_title"][:100]))
                    if "youtube_description" in youtube_extra:
                        data.append(('youtube_description', youtube_extra["youtube_description"]))
                    for tag in youtube_extra.get("tags", []):
                        data.append(('tags[]', tag))
                    data.append(('containsSyntheticMedia', "true"))

                headers = {'Authorization': f'Apikey {self.api_key}'}

                response = trace_http_call(
                    "POST", requests.post,
                    f"{self.API_BASE}/api/upload",
                    headers=headers,
                    data=data,
                    files=files,
                    timeout=300,
                    allow_redirects=False,
                )

                if 300 <= response.status_code < 400:
                    logger.error(
                        "Upload-Post upload returned an unexpected redirect: "
                        f"status={response.status_code}"
                    )
                    return unconfirmed_response(
                        "Upload-Post upload returned an unexpected redirect"
                    )

                response.raise_for_status()
                try:
                    result = response.json()
                except ValueError:
                    logger.error("Upload-Post returned invalid JSON to upload")
                    return unconfirmed_response("Upload-Post returned invalid JSON")

            # Release the source file before waiting for a remote background
            # upload, which can take much longer than the initial POST.
            if not isinstance(result, dict) or not isinstance(
                result.get("success"), bool
            ):
                logger.error("Upload-Post returned an invalid response to upload")
                return unconfirmed_response("Upload-Post returned an invalid response")

            if result["success"] and schedule:
                # 定时发布返回 202 与 job_id，没有 request_id 也没有平台结果；
                # 平台结果要到发布时间才产生，不能按后台上传轮询一小时后报失败。
                job_id = result.get("job_id")
                if not isinstance(job_id, str) or not job_id.strip():
                    return unconfirmed_response("Upload-Post scheduled the post without a job_id")
                logger.info(
                    "Upload-Post scheduled the post: "
                    f"job_id={job_id.strip()}, scheduled_date={result.get('scheduled_date')}"
                )
                return {
                    **result,
                    "success": True,
                    "scheduled": True,
                    "job_id": job_id.strip(),
                }

            if result["success"]:
                is_background = "results" not in result
                if is_background:
                    request_id = result.get("request_id")
                    if not isinstance(request_id, str) or not request_id.strip():
                        return unconfirmed_response(
                            "Upload-Post started a background upload without a request_id"
                        )
                    logger.info(
                        "Upload-Post background upload accepted: "
                        f"request_id={request_id.strip()}"
                    )
                    if on_background_start is not None:
                        try:
                            on_background_start(request_id.strip())
                        except Exception as exc:
                            # The remote upload has already started. A local
                            # status-write failure must not resubmit the video.
                            logger.warning(
                                "failed to record background upload request ID: "
                                f"{exc}"
                            )
                    result = self._wait_for_upload_completion(request_id.strip(), platforms)
                else:
                    result = self._with_platform_outcome(result, platforms)

            if result.get("success"):
                logger.info(
                    f"Video cross-posted successfully! Request ID: {result.get('request_id')}"
                )
            else:
                logger.warning(
                    f"Cross-post failed: {result.get('error') or result.get('message') or 'Unknown error'}"
                )

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to cross-post video: {str(e)}")
            uncertain_outcome = isinstance(
                e, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)
            ) or (
                e.response is not None and e.response.status_code >= 500
            )
            error = str(e)
            if uncertain_outcome:
                error += (
                    "; upload outcome is unconfirmed; "
                    f"check request_id {client_request_id} before submitting again"
                )
            return {
                "success": False,
                "request_id": client_request_id,
                "error": error,
            }

    def check_status(self, request_id: str) -> dict:
        """
        Check the status of an upload request.

        Args:
            request_id (str): The request ID from upload

        Returns:
            dict: Status information
        """
        try:
            headers = {
                'Authorization': f'Apikey {self.api_key}'
            }

            response = trace_http_call(
                "GET", requests.get,
                f"{self.API_BASE}/api/uploadposts/status",
                params={'request_id': request_id},
                headers=headers,
                timeout=30
            )

            response.raise_for_status()
            try:
                result = response.json()
            except ValueError:
                logger.error("Upload-Post returned invalid JSON to status query")
                return {
                    "success": False,
                    "error": "Upload-Post returned invalid status JSON",
                }
            if not isinstance(result, dict):
                logger.error("Upload-Post returned an invalid response to status query")
                return {
                    "success": False,
                    "error": "Upload-Post returned an invalid response",
                }
            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to check status: {str(e)}")
            return {"success": False, "error": str(e)}


# Singleton instance
upload_post_service = UploadPostService()


def cross_post_video(
    video_path: str,
    title: str,
    platforms: Optional[list] = None,
    youtube_extra: Optional[dict] = None,
    on_background_start: Callable[[str], None] | None = None,
    account: dict | None = None,
    platform_titles: Optional[dict] = None,
    schedule: Optional[dict] = None,
) -> dict:
    service = UploadPostService(account) if account is not None else upload_post_service
    return service.upload_video(
        video_path,
        title,
        platforms,
        youtube_extra=youtube_extra,
        on_background_start=on_background_start,
        platform_titles=platform_titles,
        schedule=schedule,
    )
