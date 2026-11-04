"""Install the GPU server's Python environment: torch for this GPU, then diffusers.

The server runs in its own virtual environment so the main app never depends on
torch, and so a CUDA build of torch can match the installed GPU driver.
"""

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from local_ai.hardware import Hardware

PYTHON_VERSION = "3.11"
REQUIREMENTS = Path(__file__).with_name("requirements.txt")
TORCH_INDEX = "https://download.pytorch.org/whl/{tag}"

# Lowest NVIDIA driver for each CUDA build of torch, newest first.
# https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/ (minimum driver per toolkit)
_CUDA_BUILDS = ((570, "cu128"), (560, "cu126"))


def torch_index(hardware: Hardware) -> str | None:
    """The PyTorch wheel index for this machine; None means the default PyPI build."""
    nvidia = any(gpu.vendor == "nvidia" for gpu in hardware.gpus)
    if nvidia:
        try:
            driver_major = int(hardware.nvidia_driver.split(".")[0])
        except ValueError:
            driver_major = 0
        for minimum, tag in _CUDA_BUILDS:
            if driver_major >= minimum:
                return TORCH_INDEX.format(tag=tag)
        # Too old for any current CUDA build: run on the CPU rather than fail.
        return TORCH_INDEX.format(tag="cpu")
    if hardware.os == "Darwin":
        return None  # PyPI's macOS wheels include Apple's Metal (MPS) backend
    return TORCH_INDEX.format(tag="cpu")


def venv_python(runtime_dir: str) -> str:
    scripts = "Scripts" if os.name == "nt" else "bin"
    name = "python.exe" if os.name == "nt" else "python"
    return str(Path(runtime_dir) / "venv" / scripts / name)


def _requirements_digest() -> str:
    # Git may check the file out with CRLF or LF; the requirements are the same.
    text = REQUIREMENTS.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_installed(runtime_dir: str) -> bool:
    """True when the runtime finished installing the current requirements.

    A runtime from an older release lacks packages newer models need, so it
    counts as missing and the next install only adds what changed.
    """
    try:
        marker = Path(runtime_dir, "venv", ".mpt-installed").read_text(encoding="utf-8")
    except OSError:
        return False
    return _requirements_digest() in marker.split()


def install_commands(runtime_dir: str, hardware: Hardware) -> list[list[str]]:
    venv = str(Path(runtime_dir) / "venv")
    python = venv_python(runtime_dir)
    index = torch_index(hardware)
    torch_args = ["torch"] + (["--index-url", index] if index else [])
    uv = shutil.which("uv")
    if uv:
        return [
            [uv, "venv", "--python", PYTHON_VERSION, "--allow-existing", venv],
            [uv, "pip", "install", "--python", python, *torch_args],
            [uv, "pip", "install", "--python", python, "-r", str(REQUIREMENTS)],
        ]
    return [
        [sys.executable, "-m", "venv", venv],
        [python, "-m", "pip", "install", "--upgrade", "pip"],
        [python, "-m", "pip", "install", *torch_args],
        [python, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
    ]


def install(runtime_dir: str, hardware: Hardware, log_path: str) -> None:
    """Run the install commands, appending their output to ``log_path``.

    Raises RuntimeError naming the failed command; the log has the details.
    """
    Path(runtime_dir).mkdir(parents=True, exist_ok=True)
    marker = Path(runtime_dir, "venv", ".mpt-installed")
    marker.unlink(missing_ok=True)
    with open(log_path, "a", encoding="utf-8", errors="replace") as log:
        for command in install_commands(runtime_dir, hardware):
            log.write(f"$ {subprocess.list2cmdline(command)}\n")
            log.flush()
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"{Path(command[0]).name} exited with {result.returncode}")
    marker.write_text(f"{torch_index(hardware) or 'pypi'} {_requirements_digest()}", encoding="utf-8")
