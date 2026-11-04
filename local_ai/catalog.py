"""Model catalog and the choice of the best model for this machine's hardware."""

import json
from dataclasses import dataclass
from pathlib import Path

from local_ai.hardware import Gpu, Hardware

CATALOG_PATH = Path(__file__).with_name("catalog.json")

# How a model would run here, best first.
FIT_GPU = "gpu"  # weights stay on the GPU: fastest
FIT_OFFLOAD = "offload"  # parts move between RAM and GPU per step: slower
FIT_CPU = "cpu"  # no usable GPU: much slower
FIT_NONE = "none"
_FIT_RANK = {FIT_GPU: 0, FIT_OFFLOAD: 1, FIT_CPU: 2, FIT_NONE: 3}


@dataclass(frozen=True)
class Model:
    id: str
    kind: str
    name: str
    repo: str
    family: str
    license: str
    gated: bool
    download_gb: float
    quality: int
    vram_gpu_gb: float
    vram_offload_gb: float
    ram_offload_gb: float
    cpu: bool
    allow_patterns: tuple[str, ...]
    ignore_patterns: tuple[str, ...]
    variant: str | None
    steps: int
    guidance: float
    notes: str = ""
    # Video models only: output frame rate, frames per clip, and whether the
    # model animates a start image (the scene's generated image) or text only.
    fps: int = 0
    frames: int = 0
    image_to_video: bool = False
    # Components loaded in 4-bit (bitsandbytes NF4): the memory figures above
    # assume it. Wan's T5 text encoder alone is 11 GB in bf16.
    quantize: tuple[str, ...] = ()
    # Components rewritten on first load as bfloat16 shards of at most 1 GB:
    # less disk, and loading never needs a whole multi-GB shard in memory.
    bf16: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict) -> "Model":
        fields = dict(data)
        fields["allow_patterns"] = tuple(fields.get("allow_patterns") or ())
        fields["ignore_patterns"] = tuple(fields.get("ignore_patterns") or ())
        fields["quantize"] = tuple(fields.get("quantize") or ())
        fields["bf16"] = tuple(fields.get("bf16") or ())
        return cls(**fields)


@dataclass(frozen=True)
class Fit:
    model: Model
    mode: str
    gpu_index: int | None

    def to_dict(self) -> dict:
        return {"model_id": self.model.id, "mode": self.mode, "gpu_index": self.gpu_index}


def load(path: Path = CATALOG_PATH) -> list[Model]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Model.from_dict(entry) for entry in data["models"]]


def get(model_id: str, models: list[Model] | None = None) -> Model | None:
    return next((model for model in (models or load()) if model.id == model_id), None)


def fit(model: Model, hardware: Hardware) -> Fit:
    """How ``model`` would run on the GPU with the most memory, if at all."""
    gpu: Gpu | None = hardware.best_gpu
    if gpu is not None:
        if gpu.vram_gb >= model.vram_gpu_gb:
            return Fit(model, FIT_GPU, gpu.index)
        if gpu.vram_gb >= model.vram_offload_gb and hardware.ram_gb >= model.ram_offload_gb:
            return Fit(model, FIT_OFFLOAD, gpu.index)
    if model.cpu and hardware.ram_gb >= model.ram_offload_gb:
        return Fit(model, FIT_CPU, None)
    return Fit(model, FIT_NONE, None)


def rank(hardware: Hardware, kind: str = "image", models: list[Model] | None = None) -> list[Fit]:
    """Every model of ``kind``, best choice first.

    Offloading only moves whole components (text encoders, VAE) between RAM
    and GPU, so it costs seconds per image and a better model is worth it.
    The CPU costs minutes per image, so any GPU option beats it.
    """
    fits = [fit(model, hardware) for model in (models or load()) if model.kind == kind]

    def key(item: Fit):
        tier = 0 if item.mode in (FIT_GPU, FIT_OFFLOAD) else _FIT_RANK[item.mode]
        return (tier, -item.model.quality, _FIT_RANK[item.mode])

    return sorted(fits, key=key)


def recommend(hardware: Hardware, kind: str = "image", models: list[Model] | None = None) -> Fit | None:
    """The best model that can run here; gated models only when nothing else fits."""
    usable = [item for item in rank(hardware, kind, models) if item.mode != FIT_NONE]
    open_models = [item for item in usable if not item.model.gated]
    return (open_models or usable or [None])[0]
