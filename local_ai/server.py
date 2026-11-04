"""OpenAI-compatible image server, plus clip generation, for downloaded catalog models.

MoneyPrinterTurbo starts it in its own Python environment (see runtime.py). It
can also run by hand: ``python -m local_ai.server`` from the project root.

One image model and one video model can be loaded at a time, but only one of
them sits on the GPU: a scene first gets its image, then the clip animates it.

The GPU is shared with Ollama, whose script model may not fit beside these
models. Before every request the server asks Ollama to unload its models, and
an idle model moves back to the CPU so Ollama gets the whole GPU for the next
script. This is the scheme from docs/local-image-server, made model-agnostic.
"""

import base64
import binascii
import gc
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from local_ai import catalog, hardware, models, pipelines

HOST = os.environ.get("LOCAL_AI_HOST", "127.0.0.1")
PORT = int(os.environ.get("LOCAL_AI_PORT", "7862"))
MODELS_DIR = os.environ.get("LOCAL_AI_MODELS_DIR", os.path.join("storage", "local_ai", "models"))
DEFAULT_MODELS = {
    "image": os.environ.get("LOCAL_AI_IMAGE_MODEL", ""),
    "video": os.environ.get("LOCAL_AI_VIDEO_MODEL", ""),
}
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
VIDEO_NEGATIVE = os.environ.get(
    "LOCAL_AI_VIDEO_NEGATIVE",
    "text, subtitles, watermark, logo, static frame, still picture, blurry, "
    "low quality, jpeg artifacts, deformed, distorted, extra limbs, flickering",
)
MAX_IMAGES_PER_REQUEST = 4


@dataclass
class Slot:
    """The pipeline loaded for one kind of model."""

    model: catalog.Model | None = None
    pipe: object = None
    mode: str = ""
    on_gpu: bool = False


# One lock guards the slots: one request at a time per GPU.
lock = threading.Lock()
slots = {"image": Slot(), "video": Slot()}
last_used = 0.0
stop_idle = threading.Event()
idle_thread = None


class ImageRequest(BaseModel):
    prompt: str
    model: str | None = None
    n: int = 1
    size: str | None = None
    response_format: str | None = "b64_json"


class VideoRequest(BaseModel):
    prompt: str
    model: str | None = None
    size: str | None = None
    # Base64 PNG or JPEG start frame; required by image-to-video models.
    image: str | None = None


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
    # Ollama can load again between requests, so this runs before every one.
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
        print(f"ollama still holds {', '.join(left)} after {OLLAMA_UNLOAD_TIMEOUT:g}s, continuing anyway")
    return names


def resolve_model(kind: str, model_id: str | None) -> tuple[catalog.Model, str]:
    """The catalog entry and local folder for a request; HTTP errors for the client."""
    model_id = (model_id or DEFAULT_MODELS[kind] or "").strip()
    if not model_id:
        raise HTTPException(400, f"no {kind} model requested and no default is set")
    model = catalog.get(model_id)
    if model is None or model.kind != kind:
        raise HTTPException(404, f"unknown {kind} model {model_id!r}")
    path = models.installed_path(model, MODELS_DIR)
    if path is None:
        raise HTTPException(409, f"model {model_id!r} is not downloaded")
    return model, path


def _unload(slot: Slot) -> None:
    slot.pipe, slot.model, slot.mode, slot.on_gpu = None, None, "", False
    # Large pipelines hold reference cycles; collect now so their memory is
    # free before the next model loads, not whenever Python next collects.
    gc.collect()
    pipelines.empty_cache()


def _move_to_cpu(slot: Slot) -> None:
    """Free the GPU. 4-bit (bitsandbytes) weights cannot move, so they unload."""
    if not slot.on_gpu:
        return
    if slot.model is not None and slot.model.quantize:
        _unload(slot)
        return
    slot.pipe.to("cpu")
    slot.on_gpu = False
    pipelines.empty_cache()


def claim(kind: str, model: catalog.Model, path: str):
    """Make ``model`` the loaded pipeline of its kind and give it the GPU.

    Call with the lock held. The other kind's model leaves the GPU first, and
    before loading a new model it unloads completely: a video model needs most
    of an ordinary computer's free memory while it loads.
    """
    slot = slots[kind]
    loading = slot.model is None or slot.model.id != model.id
    for other_kind, other in slots.items():
        if other_kind == kind:
            continue
        if loading:
            _unload(other)
        else:
            _move_to_cpu(other)
    if loading:
        _unload(slot)
        slot.mode = catalog.fit(model, _hardware_for_server()).mode
        if slot.mode == catalog.FIT_NONE:
            raise HTTPException(507, f"{model.name} does not fit this machine's GPU or memory")
        print(f"loading {model.id} ({slot.mode})")
        loader = pipelines.load_video if kind == "video" else pipelines.load
        try:
            slot.pipe = loader(model, path, slot.mode)
        except (MemoryError, OSError) as error:
            # A half-loaded model would keep its memory and block the next try.
            _unload(slot)
            raise HTTPException(
                507, f"not enough memory to load {model.name}: {error}"
            ) from error
        slot.model = model
    if slot.mode == catalog.FIT_GPU:
        unload_ollama()
        if not slot.on_gpu:
            slot.pipe.to(pipelines.device())
            slot.on_gpu = True
    elif slot.mode == catalog.FIT_OFFLOAD:
        # Offloaded models still peak at their largest part on the GPU.
        unload_ollama()
    return slot.pipe


def offload_if_idle() -> bool:
    """Move idle models back to the CPU so Ollama has the GPU."""
    with lock:
        busy = [slot for slot in slots.values() if slot.on_gpu]
        if not busy or time.time() - last_used < IDLE_SECONDS:
            return False
        for slot in busy:
            _move_to_cpu(slot)
    print("idle: moved pipelines to cpu")
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
    image, video = slots["image"], slots["video"]
    return {
        "ok": True,
        "model": image.model.id if image.model else None,
        "mode": image.mode or None,
        "on_gpu": image.on_gpu,
        "video_model": video.model.id if video.model else None,
        "video_mode": video.mode or None,
        "video_on_gpu": video.on_gpu,
        "pid": os.getpid(),
    }


@app.get("/v1/models")
def list_models():
    installed = [model for model in catalog.load() if models.installed_path(model, MODELS_DIR)]
    return {
        "object": "list",
        "data": [{"id": model.id, "object": "model", "kind": model.kind} for model in installed],
    }


@app.post("/v1/images/generations")
def generate(req: ImageRequest):
    global last_used
    if not req.prompt.strip():
        raise HTTPException(400, "prompt is empty")
    model, path = resolve_model("image", req.model)
    width, height = pipelines.pick_size(model.family, req.size)
    prompt = f"{req.prompt.strip()}, {STYLE}" if STYLE else req.prompt.strip()
    data = []
    with lock:
        try:
            for _ in range(max(1, min(req.n, MAX_IMAGES_PER_REQUEST))):
                # Reclaim per image: Ollama may have loaded while the last one painted.
                current = claim("image", model, path)
                image = pipelines.paint(current, model, prompt, NEGATIVE, width, height)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                data.append({"b64_json": base64.b64encode(buffer.getvalue()).decode()})
        finally:
            # torch keeps gigabytes of decode buffers cached; give them back for Ollama.
            pipelines.empty_cache()
            last_used = time.time()
    return {"created": int(time.time()), "data": data}


def _decode_image(encoded: str | None):
    if not encoded:
        return None
    from PIL import Image, UnidentifiedImageError

    try:
        return Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))).convert("RGB")
    except (binascii.Error, ValueError, UnidentifiedImageError) as error:
        raise HTTPException(400, f"image is not a base64 PNG or JPEG: {error}") from error


@app.post("/v1/videos/generations")
def generate_video(req: VideoRequest):
    global last_used
    if not req.prompt.strip():
        raise HTTPException(400, "prompt is empty")
    model, path = resolve_model("video", req.model)
    image = _decode_image(req.image)
    if model.image_to_video and image is None:
        raise HTTPException(400, f"{model.name} animates a start image: send it as image")
    width, height = pipelines.pick_video_size(model.family, req.size)
    with lock:
        try:
            current = claim("video", model, path)
            clip = pipelines.animate(
                current, model, req.prompt.strip(), VIDEO_NEGATIVE, width, height, image
            )
        finally:
            pipelines.empty_cache()
            last_used = time.time()
    return {
        "created": int(time.time()),
        "data": [{
            "b64_mp4": base64.b64encode(clip).decode(),
            "seconds": round(model.frames / model.fps, 2),
            "width": width,
            "height": height,
        }],
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
