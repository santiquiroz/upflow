from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import numpy as np

from app.config import Settings
from app.exceptions import ModelNotFoundError, ModelProtectedError
from app.services.engines.onnx_upscaler import (
    _build_providers,
    _detect_scale,
    _from_nchw_float,
    _to_nchw_float,
)
from app.services.hf_client import HfClient, pick_weight_file
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SP1 Task 5 - model_installer: installs .onnx models straight from a
# Hugging Face repo_id into ModelRegistry, running on its own single-worker
# asyncio queue (deliberately NOT the GPU job queue -- downloads/validation
# are CPU/network-bound and must never compete with GPU-gated upscale jobs).
#
# .pth/.safetensors weights are NOT converted here -- that lands in Task 6.
# Picking a repo without an .onnx file completes the install job with
# status=error and a clear "not available yet" message instead of silently
# hanging or attempting a conversion that doesn't exist; T6 replaces this
# branch with a real conversion step.
#
# ONNX validation reuses onnx_upscaler's `_build_providers` (cpu-only here),
# `_detect_scale`, `_to_nchw_float` and `_from_nchw_float` instead of
# duplicating the tiling/scale-inference math. Only the trivial
# `InferenceSession(...)` constructor call is repeated -- OnnxUpscaler's own
# `_create_session` is keyed on an already-registered ModelEntry +
# DevicesService, neither of which exists yet for a file still being
# validated (and this task's constructor is `(settings, registry,
# hf_client)`, with no DevicesService).
# ---------------------------------------------------------------------------

# owner/name: letters, digits, '-', '_', '.' only, each segment 1-97 chars,
# exactly one '/' separator. Rejects extra path segments ("a/b/c"), parent
# traversal ("../../etc"), spaces, and any other non-allowlisted byte
# (including a literal "%00") by construction -- nothing outside the
# character class survives re.fullmatch.
REPO_ID_PATTERN = re.compile(
    r"^(?P<owner>[A-Za-z0-9][A-Za-z0-9_.-]{0,96})/(?P<name>[A-Za-z0-9][A-Za-z0-9_.-]{0,96})$"
)

VALIDATION_TILE_SIZE = 32
ONNX_SUFFIX = ".onnx"
NON_ONNX_ERROR_MESSAGE = (
    "The selected weight file is not .onnx; conversion of .pth/.safetensors "
    "weights is not available yet (coming in a later step)."
)


class InstallStatus(str, Enum):
    downloading = "downloading"
    validating = "validating"
    converting = "converting"
    installed = "installed"
    error = "error"


@dataclass(slots=True, kw_only=True)
class InstallJob:
    id: str
    repo_id: str
    status: InstallStatus = InstallStatus.downloading
    progress_pct: float | None = None
    model_id: str | None = None
    error: str | None = None


def _reject_control_characters(repo_id: str) -> None:
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in repo_id):
        raise ValueError(f"repo_id contains control characters: {repo_id!r}")


def _reject_parent_traversal(owner: str, name: str) -> None:
    if ".." in owner or ".." in name:
        raise ValueError(f"repo_id must not contain '..': {owner}/{name}")


def _validate_repo_id(repo_id: str) -> str:
    """Validates a user-supplied Hugging Face repo_id and returns a
    URL-quoted 'owner/name' safe to interpolate into HfClient request URLs.

    Binding requirement (SP1 Task 3 review carryover): repo_id comes from
    free user input and must be validated/sanitized before it ever reaches
    an HfClient URL -- strict 'owner/name' shape, no '..', no extra path
    segments, no control characters, each segment URL-quoted.
    """
    if not repo_id or not isinstance(repo_id, str):
        raise ValueError("repo_id is required")
    _reject_control_characters(repo_id)
    match = REPO_ID_PATTERN.fullmatch(repo_id)
    if not match:
        raise ValueError(
            f"repo_id must look like 'owner/name' (letters, digits, '-', '_', '.' only): {repo_id!r}"
        )
    owner, name = match.group("owner"), match.group("name")
    _reject_parent_traversal(owner, name)
    return f"{quote(owner, safe='')}/{quote(name, safe='')}"


def _model_id_from_repo_id(repo_id: str) -> str:
    return repo_id.lower().replace("/", "--")


def _progress_percent(downloaded: int, total: int | None) -> float | None:
    if not total:
        return None
    return round(min(downloaded, total) / total * 100, 1)


def _make_validation_tile() -> np.ndarray:
    return np.zeros((VALIDATION_TILE_SIZE, VALIDATION_TILE_SIZE, 3), dtype=np.uint8)


def _require_single_input(inputs: list[Any]) -> Any:
    if len(inputs) != 1:
        raise ValueError(f"ONNX model must have exactly 1 input, found {len(inputs)}")
    return inputs[0]


def _require_4d_float_input(input_info: Any) -> None:
    shape = getattr(input_info, "shape", None)
    if shape is None or len(shape) != 4:
        raise ValueError(f"ONNX model input must be 4D (NCHW), got shape {shape!r}")
    dtype = str(getattr(input_info, "type", ""))
    if "float" not in dtype.lower():
        raise ValueError(f"ONNX model input must be float, got type {dtype!r}")


class ModelInstaller:
    def __init__(self, settings: Settings, registry: ModelRegistry, hf_client: HfClient) -> None:
        self.settings = settings
        self.registry = registry
        self.hf_client = hf_client
        self._jobs: dict[str, InstallJob] = {}
        self._queue: asyncio.Queue[InstallJob] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._worker_task is not None:
            return
        self._worker_task = asyncio.create_task(self._worker(), name="model-install-worker")

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None

    async def install_from_hf(self, repo_id: str) -> str:
        sanitized_repo_id = _validate_repo_id(repo_id)
        job = InstallJob(id=uuid4().hex, repo_id=sanitized_repo_id)
        self._jobs[job.id] = job
        await self._queue.put(job)
        return job.id

    def status(self, install_id: str) -> InstallJob | None:
        return self._jobs.get(install_id)

    async def delete(self, model_id: str) -> None:
        entry = self.registry.get(model_id)
        if entry is None:
            raise ModelNotFoundError(f"Unknown model id: {model_id!r}")
        if entry.kind == ModelKind.builtin_ncnn:
            raise ModelProtectedError(f"Cannot remove builtin model: {model_id!r}")
        await asyncio.to_thread(self.registry.remove, model_id)

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                await self._run_install(job)
            finally:
                self._queue.task_done()

    async def _process_next(self) -> bool:
        """Test seam: processes exactly one queued job without a background
        worker task, so tests can assert on the finished InstallJob state
        deterministically instead of polling a live asyncio.Task.
        """
        try:
            job = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return False
        try:
            await self._run_install(job)
        finally:
            self._queue.task_done()
        return True

    async def _run_install(self, job: InstallJob) -> None:
        try:
            await self._download_and_register(job)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as job.error, never raised
            job.status = InstallStatus.error
            job.error = str(exc)
            logger.warning("Model install failed for repo_id=%r: %s", job.repo_id, exc)

    async def _download_and_register(self, job: InstallJob) -> None:
        job.status = InstallStatus.downloading
        files = await self.hf_client.repo_files(job.repo_id)
        weight_file = pick_weight_file(files)

        if Path(weight_file.path).suffix.lower() != ONNX_SUFFIX:
            job.status = InstallStatus.error
            job.error = NON_ONNX_ERROR_MESSAGE
            return

        model_id = _model_id_from_repo_id(job.repo_id)
        final_dest = self._onnx_dest_path(model_id)
        # Downloaded/validated under a staging name, not final_dest directly:
        # reinstalling an already-installed repo whose new download fails
        # validation must not delete the previously-working file at
        # final_dest. Only a successful validation promotes staging_dest to
        # final_dest (atomic rename), mirroring the tmp-then-replace pattern
        # already used by ModelRegistry._persist and HfClient.download.
        staging_dest = final_dest.with_name(f"{final_dest.name}.validating")

        def progress_cb(downloaded: int, total: int | None) -> None:
            job.progress_pct = _progress_percent(downloaded, total)

        await self.hf_client.download(job.repo_id, weight_file.path, staging_dest, progress_cb=progress_cb)

        job.status = InstallStatus.validating
        try:
            scale = await asyncio.to_thread(self._validate_onnx_file, staging_dest)
        except Exception:
            staging_dest.unlink(missing_ok=True)
            raise

        staging_dest.replace(final_dest)

        entry = ModelEntry(
            id=model_id,
            name=job.repo_id,
            kind=ModelKind.onnx,
            source=f"https://huggingface.co/{job.repo_id}",
            size_bytes=weight_file.size,
            scale=scale,
            arch=Path(weight_file.path).stem,
            file_path=self._relative_onnx_path(model_id),
            status=ModelStatus.installed,
        )
        await asyncio.to_thread(self.registry.register, entry)
        job.model_id = model_id
        job.status = InstallStatus.installed

    def _onnx_dest_path(self, model_id: str) -> Path:
        return self.settings.models_path / "onnx" / f"{model_id}{ONNX_SUFFIX}"

    @staticmethod
    def _relative_onnx_path(model_id: str) -> str:
        return f"onnx/{model_id}{ONNX_SUFFIX}"

    def _validate_onnx_file(self, path: Path) -> int:
        session = self._create_validation_session(path)
        input_info = _require_single_input(session.get_inputs())
        _require_4d_float_input(input_info)
        output_info = session.get_outputs()[0]
        batch = _to_nchw_float(_make_validation_tile())
        result = session.run([output_info.name], {input_info.name: batch})[0]
        output_hwc = _from_nchw_float(result)
        return _detect_scale(VALIDATION_TILE_SIZE, VALIDATION_TILE_SIZE, output_hwc)

    def _create_validation_session(self, path: Path) -> Any:
        # Monkeypatchable seam (mirrors OnnxUpscaler._create_session): unit
        # tests inject a fake numpy-based session and never touch real
        # onnxruntime.
        import onnxruntime as ort

        return ort.InferenceSession(str(path), providers=_build_providers("cpu"))
