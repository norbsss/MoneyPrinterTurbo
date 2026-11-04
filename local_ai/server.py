"""OpenAI-compatible /v1/images/generations server for downloaded catalog models.

MoneyPrinterTurbo starts it in its own Python environment (see runtime.py). It
can also run by hand: ``python -m local_ai.server`` from the project root.

The GPU is shared with Ollama, whose script model may not fit beside an image
model. Before every image the server asks Ollama to unload its models, and an
idle model moves back to the CPU so Ollama gets the whole GPU for the next
script. This is the scheme from docs/local-image-server, made model-agnostic.
"""

import base64
import io
import os
import threading
import time
from contextlib import asynccontextmanager

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from local_ai import catalog, hardware, models, pipelines

HOST = os.environ.get("LOCAL_AI_HOST", "127.0.0.1")
PORT = int(os.environ.get("LOCAL_AI_PORT", "7862"))
MODELS_DIR = os.environ.get("LOCAL_AI_MODELS_DIR", os.path.join("storage", "local_ai", "models"))
DEFAULT_MODEL = os.environ.get("LOCAL_AI_IMAGE_MODEL", "")
# Index in nvidia-smi numbering; the app also sets CUDA_VISIBLE_DEVICES to it.
GPU_INDEX = os.environ.get("LOCAL_AI_GPU_INDEX", "")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_UNLOAD_TIMEOUT = float(os.environ.get("OLLAMA_UNLOAD_TIMEOUT", "30"))
IDLE_SECONDS = float(os.environ.get("LOCAL_AI_IDLE_SECONDS", "90"))
IDLE_CHECK_SECONDS = float(os.environ.get("LOCAL_AI_IDLE_CHECK_SECONDS", "10"))

STYLE = os.environ.get(
    "LOCAL_AI_STYLE",
    "cinematic photograph, natural light, highly detailed, sharp focus",
)
NEGATIVE = os.environ.get(
    "LOCAL_AI_NEGATIVE",
    "text, letters, words, watermark, logo, signature, caption, deformed, "
    "distorted, bad anatomy, extra fingers, blurry, lowres",
)
MAX_IMAGES_PER_REQUEST = 4

# One lock guards the pipeline and the state below: one image at a time per GPU.
lock = threading.Lock()
pipe = None
loaded_model: catalog.Model | None = None
mode = ""
on_gpu = False
last_used = 0.0
stop_idle = threading.Event()
idle_thread = None


class ImageRequest(BaseModel):
    prompt: str
    model: str | None = None
    n: int = 1
    size: str | None = None
    response_format: str | None = "b64_json"


def _hardware_for_server() -> hardware.Hardware:
    detected = hardware.detect()
    if GPU_INDEX.isdigit():
        detected.gpus = [gpu for gpu in detected.gpus if gpu.index == int(GPU_INDEX)]
    return detected


def ollama_models() -> list[str]:
    try:
        response = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
        return [entry["name"] for entry in response.json().get("models", [])]
    except (requests.RequestException, ValueError, KeyError):
        return []


def unload_ollama() -> list[str]:
    # Ollama can load again between requests, so this runs before every image.
    names = ollama_models()
    for name in names:
        try:
            requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": name, "keep_alive": 0},
                timeout=30,
            )
        except requests.RequestException:
            pass
    if not names:
        return []
    deadline = time.time() + OLLAMA_UNLOAD_TIMEOUT
    while (left := ollama_models()) and time.time() < deadline:
        time.sleep(0.5)
    if left:
        print(f"ollama still holds {', '.join(left)} after {OLLAMA_UNLOAD_TIMEOUT:g}s, painting anyway")
    return names


def resolve_model(model_id: str | None) -> tuple[catalog.Model, str]:
    """The catalog entry and local folder for a request; HTTP errors for the client."""
    model_id = (model_id or DEFAULT_MODEL or "").strip()
    if not model_id:
        raise HTTPException(400, "no model requested and LOCAL_AI_IMAGE_MODEL is not set")
    model = catalog.get(model_id)
    if model is None or model.kind != "image":
        raise HTTPException(404, f"unknown image model {model_id!r}")
    path = models.installed_path(model, MODELS_DIR)
    if path is None:
        raise HTTPException(409, f"model {model_id!r} is not downloaded")
    return model, path


def claim(model: catalog.Model, path: str):
    """Make ``model`` the loaded pipeline, on the GPU when it fits. Call with the lock held."""
    global pipe, loaded_model, mode, on_gpu
    if loaded_model is None or loaded_model.id != model.id:
        pipe = None
        loaded_model = None
        pipelines.empty_cache()
        mode = catalog.fit(model, _hardware_for_server()).mode
        if mode == catalog.FIT_NONE:
            raise HTTPException(507, f"{model.name} does not fit this machine's GPU or memory")
        print(f"loading {model.id} ({mode})")
        pipe = pipelines.load(model, path, mode)
        loaded_model = model
        on_gpu = False
    if mode == catalog.FIT_GPU:
        unload_ollama()
        if not on_gpu:
            pipe.to(pipelines.device())
            on_gpu = True
    elif mode == catalog.FIT_OFFLOAD:
        # Offloaded models still peak at their largest part on the GPU.
        unload_ollama()
    return pipe


def offload_if_idle() -> bool:
    """Move an idle model back to the CPU so Ollama has the GPU."""
    global on_gpu
    with lock:
        if not on_gpu or time.time() - last_used < IDLE_SECONDS:
            return False
        pipe.to("cpu")
        pipelines.empty_cache()
        on_gpu = False
    print("idle: moved pipeline to cpu")
    return True


def idle_worker():
    while not stop_idle.wait(IDLE_CHECK_SECONDS):
        offload_if_idle()


@asynccontextmanager
async def lifespan(app):
    global idle_thread
    stop_idle.clear()
    idle_thread = threading.Thread(target=idle_worker, name="idle-offload", daemon=True)
    idle_thread.start()
    try:
        yield
    finally:
        stop_idle.set()
        idle_thread.join(timeout=5)


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "ok": True,
        "model": loaded_model.id if loaded_model else None,
        "mode": mode or None,
        "on_gpu": on_gpu,
        "pid": os.getpid(),
    }


@app.get("/v1/models")
def list_models():
    installed = [
        model for model in catalog.load()
        if model.kind == "image" and models.installed_path(model, MODELS_DIR)
    ]
    return {"object": "list", "data": [{"id": model.id, "object": "model"} for model in installed]}


@app.post("/v1/images/generations")
def generate(req: ImageRequest):
    global last_used
    if not req.prompt.strip():
        raise HTTPException(400, "prompt is empty")
    model, path = resolve_model(req.model)
    width, height = pipelines.pick_size(model.family, req.size)
    prompt = f"{req.prompt.strip()}, {STYLE}" if STYLE else req.prompt.strip()
    data = []
    with lock:
        try:
            for _ in range(max(1, min(req.n, MAX_IMAGES_PER_REQUEST))):
                # Reclaim per image: Ollama may have loaded while the last one painted.
                current = claim(model, path)
                image = pipelines.paint(current, model, prompt, NEGATIVE, width, height)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                data.append({"b64_json": base64.b64encode(buffer.getvalue()).decode()})
        finally:
            # torch keeps gigabytes of decode buffers cached; give them back for Ollama.
            pipelines.empty_cache()
            last_used = time.time()
    return {"created": int(time.time()), "data": data}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
