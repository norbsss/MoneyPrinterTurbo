"""本机 AI 生图：检测硬件、选择模型、安装运行环境、下载模型并管理 GPU 服务进程。

GPU 服务（local_ai.server）运行在独立虚拟环境中，主程序不依赖 torch。主程序
只负责安装该环境、启动进程并通过 OpenAI 兼容接口请求图片；模型加载、显存
与 Ollama 的让渡都在服务进程内完成。
"""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests
from loguru import logger

import local_ai
from app.config import config
from app.utils import utils
from local_ai import catalog, hardware, models, runtime

DEFAULT_BASE_URL = "http://127.0.0.1:7862"
# 首次请求要把模型从磁盘读入显存，大模型可能需要数分钟。
IMAGE_REQUEST_TIMEOUT = (5, 900)
SERVER_START_TIMEOUT_SECONDS = 120
_HARDWARE_CACHE_SECONDS = 60

_lock = threading.Lock()
_hardware_cache: tuple[float, hardware.Hardware] | None = None
_install_job: dict | None = None
_download_jobs: dict[str, models.DownloadJob] = {}
_server_process: subprocess.Popen | None = None


class LocalAIError(RuntimeError):
    pass


def _setting(key: str, default=""):
    return config.app.get(key, default)


def base_url() -> str:
    return str(_setting("local_ai_url", "") or DEFAULT_BASE_URL).strip().rstrip("/")


def local_ai_dir() -> str:
    return utils.storage_dir("local_ai", create=True)


def models_dir() -> str:
    configured = str(_setting("local_ai_models_dir", "") or "").strip()
    return configured or os.path.join(local_ai_dir(), "models")


def runtime_dir() -> str:
    configured = str(_setting("local_ai_runtime_dir", "") or "").strip()
    return configured or os.path.join(local_ai_dir(), "runtime")


def hf_token() -> str:
    return str(_setting("local_ai_hf_api_token", "") or os.getenv("HF_TOKEN", "")).strip()


def install_log_path() -> str:
    return os.path.join(local_ai_dir(), "install.log")


def server_log_path() -> str:
    return os.path.join(local_ai_dir(), "server.log")


def detect_hardware(refresh: bool = False) -> hardware.Hardware:
    """检测结果缓存一分钟：nvidia-smi 每次约耗时数百毫秒，页面每次 rerun 都会读取。"""
    global _hardware_cache
    with _lock:
        if not refresh and _hardware_cache and time.time() - _hardware_cache[0] < _HARDWARE_CACHE_SECONDS:
            return _hardware_cache[1]
    detected = hardware.detect()
    with _lock:
        _hardware_cache = (time.time(), detected)
    return detected


def gpu_hardware() -> hardware.Hardware:
    """只保留服务实际使用的那块 GPU，模型适配判断与服务进程保持一致。"""
    detected = detect_hardware()
    index = str(_setting("local_ai_gpu_index", "") or "").strip()
    if index.isdigit() and any(gpu.index == int(index) for gpu in detected.gpus):
        detected = hardware.Hardware(**{**detected.__dict__, "gpus": [
            gpu for gpu in detected.gpus if gpu.index == int(index)
        ]})
    return detected


def image_model() -> catalog.Model | None:
    """用户选定的生图模型；未选择时使用本机推荐模型。"""
    selected = str(_setting("local_ai_image_model", "") or "").strip()
    if selected:
        model = catalog.get(selected)
        if model is not None and model.kind == "image":
            return model
    recommended = catalog.recommend(gpu_hardware(), "image")
    return recommended.model if recommended else None


def model_installed(model: catalog.Model) -> bool:
    return models.installed_path(model, models_dir()) is not None


def is_runtime_installed() -> bool:
    return runtime.is_installed(runtime_dir())


def readiness() -> tuple[bool, str]:
    """生成前的最小检查：运行环境已安装且所选模型已下载完整。"""
    if not is_runtime_installed():
        return False, "local_ai.runtime_missing"
    model = image_model()
    if model is None:
        return False, "local_ai.no_model_fits"
    if not model_installed(model):
        return False, "local_ai.model_missing"
    return True, ""


# ---------------------------------------------------------------- 后台任务


def install_status() -> dict | None:
    with _lock:
        return dict(_install_job) if _install_job else None


def start_runtime_install() -> dict:
    """在后台线程安装 GPU 服务环境；已在安装时直接返回当前状态。"""
    global _install_job
    with _lock:
        if _install_job and _install_job["state"] == "running":
            return dict(_install_job)
        _install_job = {"state": "running", "error": "", "started_at": time.time()}
    detected = detect_hardware(refresh=True)

    def run():
        global _install_job
        try:
            runtime.install(runtime_dir(), detected, install_log_path())
            result = {"state": "done", "error": ""}
        except Exception as error:
            logger.error(f"local AI runtime install failed: {error}")
            result = {"state": "failed", "error": str(error)}
        with _lock:
            _install_job = {**_install_job, **result}

    threading.Thread(target=run, name="local-ai-install", daemon=True).start()
    return install_status()


def download_status(model_id: str) -> dict | None:
    job = _download_jobs.get(model_id)
    return job.to_dict() if job else None


def start_download(model_id: str) -> dict:
    model = catalog.get(model_id)
    if model is None:
        raise LocalAIError(f"unknown model {model_id!r}")
    with _lock:
        job = _download_jobs.get(model_id)
        if job is None or not job.active:
            job = models.DownloadJob(model, models_dir(), hf_token() or None).start()
            _download_jobs[model_id] = job
    return job.to_dict()


def delete_model(model_id: str) -> None:
    model = catalog.get(model_id)
    if model is None:
        return
    job = _download_jobs.get(model_id)
    if job is not None and job.active:
        raise LocalAIError("model is still downloading")
    stop_server()  # 服务可能正持有该模型的文件句柄
    models.delete(model, models_dir())
    _download_jobs.pop(model_id, None)


# ---------------------------------------------------------------- GPU 服务进程


def _session() -> requests.Session:
    # 本机服务不能走用户为外网配置的代理。
    session = requests.Session()
    session.trust_env = False
    return session


def server_status() -> dict | None:
    try:
        with _session() as session:
            response = session.get(f"{base_url()}/health", timeout=2)
        return response.json() if response.ok else None
    except (requests.RequestException, ValueError):
        return None


def start_server() -> None:
    global _server_process
    if not is_runtime_installed():
        raise LocalAIError(
            "the local AI runtime is not installed: open Settings > AI Image > Local AI "
            "and click Install"
        )
    model = image_model()
    target = gpu_hardware().best_gpu
    env = dict(os.environ)
    package_root = str(Path(local_ai.__file__).resolve().parent.parent)
    env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
    env["LOCAL_AI_MODELS_DIR"] = models_dir()
    env["LOCAL_AI_PORT"] = str(requests.utils.urlparse(base_url()).port or 7862)
    env["PYTHONUNBUFFERED"] = "1"
    if model is not None:
        env["LOCAL_AI_IMAGE_MODEL"] = model.id
    if target is not None and target.vendor == "nvidia":
        # 只让服务看到选定的显卡，模型不会被拆到两块卡上。
        env["CUDA_VISIBLE_DEVICES"] = str(target.index)
        env["LOCAL_AI_GPU_INDEX"] = str(target.index)
    with _lock:
        if _server_process is not None and _server_process.poll() is None:
            return
        log = open(server_log_path(), "a", encoding="utf-8", errors="replace")
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        _server_process = subprocess.Popen(
            [runtime.venv_python(runtime_dir()), "-m", "local_ai.server"],
            cwd=package_root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        log.close()
    logger.info(f"started local AI server: pid={_server_process.pid}, url={base_url()}")


def stop_server() -> None:
    """停止服务进程，包括上次运行 WebUI 时启动、仍在后台运行的服务。

    Windows 上虚拟环境的 python.exe 是启动器，真正的解释器是它的子进程，
    只结束 Popen 句柄会留下占用显存的服务，因此按服务自报的 pid 结束。
    """
    global _server_process
    status = server_status()
    with _lock:
        process, _server_process = _server_process, None
    server_pid = status.get("pid") if status else None
    if isinstance(server_pid, int) and server_pid != os.getpid():
        try:
            os.kill(server_pid, signal.SIGTERM)
        except OSError:
            pass
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def ensure_server(timeout: float = SERVER_START_TIMEOUT_SECONDS) -> dict:
    """服务未运行时按配置自动启动，并等待健康检查通过。"""
    status = server_status()
    if status:
        return status
    if not bool(_setting("local_ai_autostart", True)):
        raise LocalAIError(f"the local AI server is not running at {base_url()}")
    start_server()
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _server_process is not None and _server_process.poll() is not None:
            raise LocalAIError(f"local AI server exited, see {server_log_path()}")
        status = server_status()
        if status:
            return status
        time.sleep(1)
    raise LocalAIError(f"local AI server did not start within {timeout:g}s, see {server_log_path()}")


def generate_image(prompt: str, size: str) -> bytes:
    """请求一张图片，返回 PNG 字节。服务未运行时自动启动。"""
    import base64

    model = image_model()
    if model is None:
        raise LocalAIError("no local image model fits this computer's GPU or memory")
    ensure_server()
    payload = {"model": model.id, "prompt": prompt, "n": 1, "size": size}
    try:
        with _session() as session:
            response = session.post(
                f"{base_url()}/v1/images/generations",
                json=payload,
                timeout=IMAGE_REQUEST_TIMEOUT,
            )
    except requests.RequestException as error:
        raise LocalAIError(f"local AI server request failed: {error}") from error
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise LocalAIError(f"local AI server returned {response.status_code}: {detail}")
    try:
        return base64.b64decode(response.json()["data"][0]["b64_json"])
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise LocalAIError("local AI server returned no image") from error
