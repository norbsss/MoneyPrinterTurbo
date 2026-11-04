"""Detect GPUs, system memory and free disk without importing torch."""

import ctypes
import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field

_NVIDIA_SMI_TIMEOUT_SECONDS = 10
# Older drivers print "CUDA Version: 12.4", newer ones "CUDA UMD Version: 13.3".
_CUDA_VERSION_PATTERN = re.compile(r"CUDA (?:UMD )?Version:\s*(\d+)\.(\d+)")


@dataclass
class Gpu:
    index: int
    name: str
    vendor: str  # "nvidia" or "apple"
    vram_gb: float
    free_vram_gb: float | None = None


@dataclass
class Hardware:
    os: str
    machine: str
    ram_gb: float
    gpus: list[Gpu] = field(default_factory=list)
    nvidia_driver: str = ""
    cuda_version: tuple[int, int] | None = None

    @property
    def best_gpu(self) -> Gpu | None:
        """The GPU with the most memory; models run on one card at a time."""
        return max(self.gpus, key=lambda gpu: gpu.vram_gb, default=None)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["cuda_version"] = (
            ".".join(str(part) for part in self.cuda_version) if self.cuda_version else ""
        )
        best = self.best_gpu
        data["best_gpu_index"] = best.index if best else None
        return data


def _run(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def _mib_to_gb(value: str) -> float | None:
    try:
        return round(float(value) / 1024, 1)
    except ValueError:
        return None


def parse_nvidia_gpus(csv_output: str) -> tuple[list[Gpu], str]:
    """Parse ``nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version``."""
    gpus, driver = [], ""
    for line in csv_output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5 or not parts[0].isdigit():
            continue
        total = _mib_to_gb(parts[2])
        if total is None:
            continue
        gpus.append(
            Gpu(
                index=int(parts[0]),
                name=parts[1],
                vendor="nvidia",
                vram_gb=total,
                free_vram_gb=_mib_to_gb(parts[3]),
            )
        )
        driver = driver or parts[4]
    return gpus, driver


def parse_cuda_version(nvidia_smi_output: str) -> tuple[int, int] | None:
    match = _CUDA_VERSION_PATTERN.search(nvidia_smi_output)
    return (int(match.group(1)), int(match.group(2))) if match else None


def _nvidia() -> tuple[list[Gpu], str, tuple[int, int] | None]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return [], "", None
    gpus, driver = parse_nvidia_gpus(
        _run(
            [
                executable,
                "--query-gpu=index,name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ]
        )
    )
    cuda = parse_cuda_version(_run([executable])) if gpus else None
    return gpus, driver, cuda


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _ram_bytes() -> int:
    system = platform.system()
    if system == "Windows":
        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys)
        return 0
    if system == "Darwin":
        output = _run(["sysctl", "-n", "hw.memsize"]).strip()
        return int(output) if output.isdigit() else 0
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def _apple_gpu(ram_gb: float) -> list[Gpu]:
    # Apple Silicon shares memory between CPU and GPU; macOS lets the GPU use
    # about three quarters of it.
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return [Gpu(index=0, name="Apple Silicon", vendor="apple", vram_gb=round(ram_gb * 0.75, 1))]
    return []


def free_disk_gb(path: str) -> float:
    """Free space on the drive that holds ``path`` (or its nearest existing parent)."""
    path = os.path.abspath(path)
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    try:
        return round(shutil.disk_usage(path).free / 1024**3, 1)
    except OSError:
        return 0.0


def detect() -> Hardware:
    ram_gb = round(_ram_bytes() / 1024**3, 1)
    gpus, driver, cuda = _nvidia()
    if not gpus:
        gpus = _apple_gpu(ram_gb)
    return Hardware(
        os=platform.system(),
        machine=platform.machine(),
        ram_gb=ram_gb,
        gpus=gpus,
        nvidia_driver=driver,
        cuda_version=cuda,
    )
