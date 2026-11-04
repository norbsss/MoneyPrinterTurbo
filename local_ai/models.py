"""Download catalog models from Hugging Face and track which ones are complete.

Each model downloads into its own folder under the models directory, as plain
files: the Hugging Face cache relies on symlinks, which Windows refuses without
Developer Mode. A finished download writes a marker file, and only folders with
a marker count as installed, so an interrupted download is never loaded.
"""

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from local_ai.catalog import Model
from local_ai.hardware import free_disk_gb

STATE_QUEUED = "queued"
STATE_DOWNLOADING = "downloading"
STATE_DONE = "done"
STATE_FAILED = "failed"

MARKER_NAME = ".mpt-complete.json"
# Keep this much disk free after a download so the system and renders keep working.
_DISK_HEADROOM_GB = 5.0


class DownloadError(RuntimeError):
    pass


def model_dir(model: Model, models_dir: str) -> Path:
    return Path(models_dir) / model.id


def installed_path(model: Model, models_dir: str) -> str | None:
    """The folder of a completely downloaded model, else None."""
    folder = model_dir(model, models_dir)
    try:
        data = json.loads((folder / MARKER_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return str(folder) if data.get("repo") == model.repo else None


def delete(model: Model, models_dir: str) -> None:
    folder = model_dir(model, models_dir)
    # Drop the marker first: a half-deleted folder must never look installed.
    (folder / MARKER_NAME).unlink(missing_ok=True)
    shutil.rmtree(folder, ignore_errors=True)


def folder_size(path: Path) -> int:
    """Bytes under ``path``, including the partial files of a running download."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def remote_size(model: Model, token: str | None = None) -> int:
    """Bytes of the files this model downloads, from the Hugging Face API."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import filter_repo_objects

    info = HfApi().model_info(model.repo, files_metadata=True, token=token or None)
    siblings = filter_repo_objects(
        info.siblings or [],
        allow_patterns=list(model.allow_patterns) or None,
        ignore_patterns=list(model.ignore_patterns) or None,
        key=lambda sibling: sibling.rfilename,
    )
    return sum(sibling.size or 0 for sibling in siblings)


@dataclass
class DownloadJob:
    model: Model
    models_dir: str
    token: str | None = None
    state: str = STATE_QUEUED
    bytes_total: int = 0
    error: str = ""
    path: str = ""
    started_at: float = field(default_factory=time.time)
    _thread: threading.Thread | None = field(default=None, repr=False)

    @property
    def bytes_done(self) -> int:
        if self.state == STATE_DONE:
            return self.bytes_total
        # Files left by an earlier, interrupted download count: they are not fetched again.
        done = folder_size(model_dir(self.model, self.models_dir))
        return min(done, self.bytes_total) if self.bytes_total else done

    @property
    def active(self) -> bool:
        return self.state in (STATE_QUEUED, STATE_DOWNLOADING)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model.id,
            "state": self.state,
            "bytes_done": self.bytes_done,
            "bytes_total": self.bytes_total,
            "error": self.error,
            "path": self.path,
        }

    def start(self) -> "DownloadJob":
        self._thread = threading.Thread(
            target=self._run, name=f"download-{self.model.id}", daemon=True
        )
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            self.path = download(self.model, self.models_dir, self.token, job=self)
            self.state = STATE_DONE
        except Exception as error:  # report every failure to the UI instead of a dead thread
            self.error = f"{type(error).__name__}: {error}"
            self.state = STATE_FAILED


def download(
    model: Model,
    models_dir: str,
    token: str | None = None,
    job: DownloadJob | None = None,
) -> str:
    """Download ``model`` under ``models_dir`` and mark it complete. Returns its folder."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    folder = model_dir(model, models_dir)
    total = remote_size(model, token)
    needed_gb = max(0, total - folder_size(folder)) / 1024**3
    free_gb = free_disk_gb(str(folder))
    if needed_gb + _DISK_HEADROOM_GB > free_gb:
        raise DownloadError(
            f"{model.name} needs {needed_gb:.1f} GB but only {free_gb:.1f} GB is free for {folder}"
        )
    if job is not None:
        job.bytes_total = total
        job.state = STATE_DOWNLOADING
    try:
        snapshot_download(
            model.repo,
            local_dir=str(folder),
            allow_patterns=list(model.allow_patterns) or None,
            ignore_patterns=list(model.ignore_patterns) or None,
            token=token or None,
        )
    except GatedRepoError as error:
        raise DownloadError(
            f"{model.name} is gated: accept its terms at https://huggingface.co/{model.repo} "
            "and set a Hugging Face token"
        ) from error
    (folder / MARKER_NAME).write_text(
        json.dumps({"repo": model.repo, "downloaded_at": time.time()}), encoding="utf-8"
    )
    return str(folder)
