"""Load catalog models with diffusers and paint images. Imports torch lazily."""

import random

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


def empty_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
