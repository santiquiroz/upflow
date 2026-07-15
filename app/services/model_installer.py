from __future__ import annotations

import asyncio
import gc
import logging
import re
from collections.abc import Callable
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
from app.services.hf_client import HfClient, HfFile, ProgressCallback, pick_weight_file
from app.services.model_converter import ConversionResult, convert_to_onnx
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SP1 Task 5 - model_installer: installs .onnx models straight from a
# Hugging Face repo_id into ModelRegistry, running on its own single-worker
# asyncio queue (deliberately NOT the GPU job queue -- downloads/validation
# are CPU/network-bound and must never compete with GPU-gated upscale jobs).
#
# SP1 Task 6 - .pth/.safetensors weights are converted to ONNX via
# model_converter.convert_to_onnx (Spandrel + torch.onnx.export) before
# entering the SAME staging/validate/promote pipeline used for a natively
# published .onnx file, through an extra `converting` job status between
# `downloading` and `validating`. The raw downloaded checkpoint is staged
# under settings.temp_path (NOT the final models/onnx dir -- it is never a
# registered artifact) and is always deleted once conversion finishes,
# success or failure: only the resulting .onnx is kept, mirroring "kind=onnx
# has exactly one file on disk" everywhere else in this module. `arch` and
# `scale` on the registered entry come from Spandrel's ConversionResult (the
# architecture's own declared metadata), not from the filename or from
# re-inferring scale via a black-box ONNX forward pass -- the latter still
# runs (via `_validate_onnx_file`, unchanged) as a structural sanity check
# on the exported graph, but its returned scale is discarded in favor of the
# authoritative one for converted models.
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

# Windows can keep a brief handle on the just-validated .onnx file (ORT's
# validation session, or a warm cached session from a previous install of
# the same model_id) after the InferenceSession object is logically dead --
# _validate_onnx_file forces `del` + `gc.collect()` to drop it eagerly, and
# this backoff absorbs whatever race remains before failing the install.
PROMOTE_RETRY_DELAYS_SECONDS = (0.1, 0.2, 0.4)


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
        # Capture the entry (and its file_path) BEFORE removing it: the
        # builtin/unknown guards raise before touching disk, and the on-disk
        # file is only unlinked after registry.remove() succeeds, so a failed
        # remove never leaves the registry and disk inconsistent.
        entry = self.registry.get(model_id)
        if entry is None:
            raise ModelNotFoundError(f"Unknown model id: {model_id!r}")
        if entry.kind == ModelKind.builtin_ncnn:
            raise ModelProtectedError(f"Cannot remove builtin model: {model_id!r}")
        await asyncio.to_thread(self.registry.remove, model_id)
        await asyncio.to_thread(self._delete_model_file, entry)

    def _delete_model_file(self, entry: ModelEntry) -> None:
        # Only ever deletes an onnx entry's own file, and only when it
        # resolves INSIDE settings.models_path -- guards a manipulated/corrupt
        # file_path (traversal, absolute path) from unlinking arbitrary files.
        if entry.kind != ModelKind.onnx or entry.file_path is None:
            return
        models_root = self.settings.models_path.resolve()
        target = (self.settings.models_path / entry.file_path).resolve()
        if not target.is_relative_to(models_root):
            logger.warning("Refusing to delete model file outside models dir: %s", target)
            return
        try:
            target.unlink(missing_ok=True)
        except OSError:
            # Windows can hold a lock on a file still mapped by a warm ORT
            # session; log and move on rather than failing the delete.
            logger.exception("Failed to delete model file %s", target)

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

        weight_suffix = Path(weight_file.path).suffix.lower()
        if weight_suffix == ONNX_SUFFIX:
            await self.hf_client.download(job.repo_id, weight_file.path, staging_dest, progress_cb=progress_cb)
            conversion_result = None
        else:
            conversion_result = await self._download_and_convert(
                job, weight_file, weight_suffix, model_id, staging_dest, progress_cb
            )

        job.status = InstallStatus.validating
        try:
            detected_scale = await asyncio.to_thread(self._validate_onnx_file, staging_dest)
        except Exception:
            staging_dest.unlink(missing_ok=True)
            raise

        # A converted model's arch/scale/size come from Spandrel's own
        # metadata + the resulting .onnx (authoritative), not from the
        # filename stem, the HF-declared checkpoint size, or the
        # runtime-detected scale of a zero-tile forward pass -- that
        # detection still runs above as a structural sanity check on the
        # exported graph (1 input, 4D, float), its returned scale is just not
        # the one that gets stored for converted models.
        if conversion_result is not None:
            arch = conversion_result.arch
            scale = conversion_result.scale
            size_bytes = staging_dest.stat().st_size
        else:
            arch = Path(weight_file.path).stem
            scale = detected_scale
            size_bytes = weight_file.size

        await self._promote_staging_file(staging_dest, final_dest)

        entry = ModelEntry(
            id=model_id,
            name=job.repo_id,
            kind=ModelKind.onnx,
            source=f"https://huggingface.co/{job.repo_id}",
            size_bytes=size_bytes,
            scale=scale,
            arch=arch,
            file_path=self._relative_onnx_path(model_id),
            status=ModelStatus.installed,
        )
        await asyncio.to_thread(self.registry.register, entry)
        job.model_id = model_id
        job.status = InstallStatus.installed

    async def _download_and_convert(
        self,
        job: InstallJob,
        weight_file: HfFile,
        weight_suffix: str,
        model_id: str,
        staging_dest: Path,
        progress_cb: ProgressCallback,
    ) -> ConversionResult:
        # The raw .pth/.safetensors checkpoint is staged OUTSIDE the models
        # dir (settings.temp_path): it is never a registered artifact, only
        # ever an input to conversion, and is always removed below -- success
        # or failure -- so a `converting` job never leaves it behind.
        source_weight_path = self._weight_source_path(model_id, weight_suffix)
        await self.hf_client.download(job.repo_id, weight_file.path, source_weight_path, progress_cb=progress_cb)

        job.status = InstallStatus.converting
        try:
            return await asyncio.to_thread(
                convert_to_onnx,
                source_weight_path,
                staging_dest,
                self._conversion_progress_logger(job),
            )
        except Exception:
            staging_dest.unlink(missing_ok=True)
            raise
        finally:
            source_weight_path.unlink(missing_ok=True)

    def _weight_source_path(self, model_id: str, suffix: str) -> Path:
        return self.settings.temp_path / f"{model_id}{suffix}"

    def _conversion_progress_logger(self, job: InstallJob) -> Callable[[str], None]:
        def _log_stage(stage: str) -> None:
            logger.debug("Converting %s: %s", job.repo_id, stage)

        return _log_stage

    def _onnx_dest_path(self, model_id: str) -> Path:
        return self.settings.models_path / "onnx" / f"{model_id}{ONNX_SUFFIX}"

    @staticmethod
    def _relative_onnx_path(model_id: str) -> str:
        return f"onnx/{model_id}{ONNX_SUFFIX}"

    async def _promote_staging_file(self, staging_dest: Path, final_dest: Path) -> None:
        """Renames the validated staging file onto final_dest, retrying a
        transient Windows PermissionError with backoff before giving up.

        A validation InferenceSession (or, on reinstall, a warm cached
        session still holding the previous final_dest open) can keep a file
        handle alive for a moment after it is logically unreferenced --
        `_validate_onnx_file` already forces `del` + `gc.collect()` to drop
        it eagerly, this retry absorbs whatever race remains.
        """
        attempts = len(PROMOTE_RETRY_DELAYS_SECONDS) + 1
        for attempt in range(attempts):
            try:
                staging_dest.replace(final_dest)
                return
            except PermissionError as exc:
                is_last_attempt = attempt == attempts - 1
                if is_last_attempt:
                    staging_dest.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Could not install model: {final_dest.name} is locked by another "
                        "process (Windows file handle not yet released). Try again."
                    ) from exc
                logger.warning(
                    "PermissionError promoting %s -> %s (attempt %d/%d), retrying: %s",
                    staging_dest,
                    final_dest,
                    attempt + 1,
                    attempts,
                    exc,
                )
                await asyncio.sleep(PROMOTE_RETRY_DELAYS_SECONDS[attempt])

    def _validate_onnx_file(self, path: Path) -> int:
        session = self._create_validation_session(path)
        try:
            input_info = _require_single_input(session.get_inputs())
            _require_4d_float_input(input_info)
            output_info = session.get_outputs()[0]
            batch = _to_nchw_float(_make_validation_tile())
            result = session.run([output_info.name], {input_info.name: batch})[0]
            output_hwc = _from_nchw_float(result)
            return _detect_scale(VALIDATION_TILE_SIZE, VALIDATION_TILE_SIZE, output_hwc)
        finally:
            # Drop the session eagerly instead of relying on refcounting at
            # function exit: an ORT session can be entangled in an internal
            # reference cycle that defers its __del__ (and the file handle it
            # holds) to the next gc pass, which on Windows can still be
            # holding final_dest.replace()'s target open.
            del session
            gc.collect()

    def _create_validation_session(self, path: Path) -> Any:
        # Monkeypatchable seam (mirrors OnnxUpscaler._create_session): unit
        # tests inject a fake numpy-based session and never touch real
        # onnxruntime.
        import onnxruntime as ort

        return ort.InferenceSession(str(path), providers=_build_providers("cpu"))
