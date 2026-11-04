"""Load catalog models with diffusers and paint images. Imports torch lazily."""

import os
import random
import time

from local_ai.catalog import FIT_CPU, FIT_OFFLOAD, Model

# Sizes each family was trained on; a request snaps to the closest aspect ratio.
_BUCKETS = {
    # about one megapixel
    "sdxl": [
        (1024, 1024), (896, 1152), (832, 1216), (768, 1344), (640, 1536),
        (1152, 896), (1216, 832), (1344, 768), (1536, 640),
    ],
    "flux": [
        (1024, 1024), (896, 1152), (832, 1216), (768, 1344), (640, 1536),
        (1152, 896), (1216, 832), (1344, 768), (1536, 640),
    ],
    # SD 1.5 drifts into duplicated subjects much above 512 px on the short side
    "sd15": [(512, 512), (512, 768), (512, 896), (768, 512), (896, 512)],
}
_DEFAULT_ASPECT = (9, 16)


def pick_size(family: str, size: str | None) -> tuple[int, int]:
    width, height = _DEFAULT_ASPECT
    if size:
        w, _, h = size.lower().partition("x")
        if w.isdigit() and h.isdigit() and int(w) > 0 and int(h) > 0:
            width, height = int(w), int(h)
    return min(_BUCKETS[family], key=lambda bucket: abs(bucket[0] / bucket[1] - width / height))


def device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load(model: Model, path: str, mode: str):
    """Load ``model`` from its local folder and place it for ``mode``."""
    import torch
    from diffusers import (
        DPMSolverMultistepScheduler,
        FluxPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
    )

    pipeline_class = {
        "sdxl": StableDiffusionXLPipeline,
        "sd15": StableDiffusionPipeline,
        "flux": FluxPipeline,
    }[model.family]
    if mode == FIT_CPU or device() == "cpu":
        dtype = torch.float32  # half precision is slow or unsupported on most CPUs
    elif model.family == "flux":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    kwargs = {"torch_dtype": dtype, "local_files_only": True}
    if model.variant:
        kwargs["variant"] = model.variant
    if model.family == "sd15":
        kwargs.update(safety_checker=None, requires_safety_checker=False)
    pipe = pipeline_class.from_pretrained(path, **kwargs)
    if model.family in ("sdxl", "sd15"):
        # Lightning and SD 1.5 checkpoints are tuned for DPM++ SDE with Karras sigmas.
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )
    if mode == FIT_OFFLOAD:
        pipe.enable_model_cpu_offload()
    return pipe


def paint(pipe, model: Model, prompt: str, negative: str, width: int, height: int):
    import torch

    generator = torch.Generator("cpu").manual_seed(random.randint(0, 2**31 - 1))
    kwargs = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": model.steps,
        "guidance_scale": model.guidance,
        "generator": generator,
    }
    if model.family == "flux":
        kwargs["max_sequence_length"] = 256  # schnell was trained with 256 T5 tokens
    else:
        kwargs["negative_prompt"] = negative
    return pipe(**kwargs).images[0]


# Wan clips at about 480p: the sizes take minutes instead of tens of minutes on
# 12-16 GB cards. Sides are multiples of 32, as the Wan 2.2 VAE and patching need.
_VIDEO_BUCKETS = {
    "wan22": [(480, 832), (832, 480), (640, 640)],
    "wan21": [(480, 832), (832, 480), (640, 640)],
}


def pick_video_size(family: str, size: str | None) -> tuple[int, int]:
    width, height = _DEFAULT_ASPECT
    if size:
        w, _, h = size.lower().partition("x")
        if w.isdigit() and h.isdigit() and int(w) > 0 and int(h) > 0:
            width, height = int(w), int(h)
    return min(_VIDEO_BUCKETS[family], key=lambda bucket: abs(bucket[0] / bucket[1] - width / height))


BF16_MARKER = ".mpt-bf16"
_MAX_SHARD_BYTES = 1024**3


def convert_to_bf16(folder: str) -> None:
    """Rewrite a component as bfloat16 shards of at most 1 GB, once.

    diffusers reads a whole shard into memory before loading it. Wan publishes
    5 GB float32 shards, and an ordinary Windows computer with WSL or Docker
    running may have less memory than that left to commit (os error 1455).
    Writing tensor by tensor into small shards keeps the peak near 1 GB here
    and when the server later loads the model. The new shards and index are
    written first and the old files removed last, so an interrupted run leaves
    a loadable model and continues next time.
    """
    import json
    from pathlib import Path

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    component = Path(folder)
    if (component / BF16_MARKER).exists():
        return
    indexes = list(component.glob("*.safetensors.index.json"))
    if len(indexes) != 1:
        return  # single-file components are small enough as they are
    index_path = indexes[0]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    old_shards = sorted(set(index["weight_map"].values()))

    def needs_rewrite(name):
        shard = component / name
        if shard.stat().st_size > _MAX_SHARD_BYTES:
            return True
        with safe_open(str(shard), framework="pt") as source:
            return any(source.get_slice(key).get_dtype() == "F32" for key in source.keys())

    if not any(needs_rewrite(name) for name in old_shards):
        (component / BF16_MARKER).write_text("bfloat16", encoding="utf-8")
        return

    prefix = index_path.name.removesuffix(".safetensors.index.json") + "-mpt"
    weight_map, written, batch, batch_bytes = {}, [], {}, 0

    def flush():
        nonlocal batch, batch_bytes
        if not batch:
            return
        name = f"{prefix}-{len(written) + 1:05d}.safetensors"
        save_file(batch, str(component / name), metadata={"format": "pt"})
        for key in batch:
            weight_map[key] = name
        written.append(name)
        batch, batch_bytes = {}, 0

    for name in old_shards:
        print(f"converting {name} to bfloat16")
        with safe_open(str(component / name), framework="pt") as source:
            for key in source.keys():
                tensor = source.get_tensor(key)
                if tensor.dtype == torch.float32:
                    tensor = tensor.to(torch.bfloat16)
                size = tensor.numel() * tensor.element_size()
                if batch and batch_bytes + size > _MAX_SHARD_BYTES:
                    flush()
                batch[key] = tensor
                batch_bytes += size
    flush()
    total = sum((component / name).stat().st_size for name in written)
    partial = index_path.with_suffix(".partial")
    partial.write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2),
        encoding="utf-8",
    )
    partial.replace(index_path)
    for name in old_shards:
        if name not in written:
            (component / name).unlink(missing_ok=True)
    (component / BF16_MARKER).write_text("bfloat16", encoding="utf-8")


def load_video(model: Model, path: str, mode: str):
    """Load a Wan video model. The VAE stays in float32, as Wan recommends."""
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline

    for component in model.bf16:
        convert_to_bf16(os.path.join(path, component))
    vae = AutoencoderKLWan.from_pretrained(
        path, subfolder="vae", torch_dtype=torch.float32, local_files_only=True
    )
    pipeline_class = WanImageToVideoPipeline if model.image_to_video else WanPipeline
    kwargs = {"vae": vae, "torch_dtype": torch.bfloat16, "local_files_only": True}
    if model.quantize:
        from diffusers.quantizers import PipelineQuantizationConfig

        kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=list(model.quantize),
        )
    pipe = pipeline_class.from_pretrained(path, **kwargs)
    if mode == FIT_OFFLOAD:
        pipe.enable_model_cpu_offload()
    # Decoding every frame at once in float32 is the peak. Past the card's
    # memory, Windows spills into shared system memory and a clip takes ten
    # times longer; tiles keep the decode on the card.
    pipe.vae.enable_tiling()
    return pipe


def animate(pipe, model: Model, prompt: str, negative: str, width: int, height: int, image=None) -> bytes:
    """Render one clip and return it as MP4 bytes."""
    import tempfile

    import torch
    from diffusers.utils import export_to_video
    from PIL import ImageOps

    kwargs = {
        "prompt": prompt,
        "negative_prompt": negative,
        "width": width,
        "height": height,
        "num_frames": model.frames,
        "num_inference_steps": model.steps,
        "guidance_scale": model.guidance,
        "generator": torch.Generator("cpu").manual_seed(random.randint(0, 2**31 - 1)),
    }
    if model.image_to_video:
        if image is None:
            raise ValueError(f"{model.name} animates a start image, but none was sent")
        # Crop the scene image to the clip's aspect ratio instead of stretching it.
        kwargs["image"] = ImageOps.fit(image.convert("RGB"), (width, height))
    started = time.time()
    stamps = {}

    def on_step_end(_pipe, step, _timestep, callback_kwargs):
        stamps.setdefault("first_step", time.time())
        stamps["last_step"] = time.time()
        return callback_kwargs

    frames = pipe(**kwargs, callback_on_step_end=on_step_end).frames[0]
    finished = time.time()
    if stamps:
        print(
            f"clip {model.id} {width}x{height}x{model.frames}: "
            f"prepare {stamps['first_step'] - started:.0f}s, "
            f"denoise {stamps['last_step'] - stamps['first_step']:.0f}s, "
            f"decode {finished - stamps['last_step']:.0f}s"
        )
    with tempfile.TemporaryDirectory() as folder:
        path = f"{folder}/clip.mp4"
        export_to_video(frames, path, fps=model.fps)
        with open(path, "rb") as clip:
            return clip.read()


def empty_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
