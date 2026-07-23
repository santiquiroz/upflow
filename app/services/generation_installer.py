from __future__ import annotations

import asyncio
import contextlib
import gc
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from app.config import Settings
from app.services.engines.generation_onnx import (
    _build_providers_for_validation,
    _load_pipeline_class,
    _wrap_generation_error,
    generation_dependencies_available,
)
from app.services.hf_client import HfClient, HfFile
from app.services.model_installer import (
    InstallJob,
    InstallStatus,
    PROMOTE_RETRY_DELAYS_SECONDS,
    _validate_repo_id,
)
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus

# ---------------------------------------------------------------------------
# SP1 generation module Task 5 - GenerationModelInstaller: installs a
# multi-file diffusers ONNX pipeline (model_index.json + one subdirectory per
# declared component) from a Hugging Face repo_id, into
# models_path/generation/<model_id>/, parallel to the single-file
# ModelInstaller in model_installer.py (which stays untouched -- kind=onnx
# has exactly one file on disk, kind=diffusion_onnx has a directory tree).
#
# Two-phase download, deliberate (spike findings,
# docs/superpowers/specs/2026-07-22-optimum-spike-findings.md): the `amd/`
# legacy repos ship a diffusers pipeline ALONGSIDE unrelated multi-GB
# artifacts (a duplicate torch .ckpt, vendor-specific MIGraphX `MXR/`
# binaries). model_index.json is a few KBs and is the only reliable source of
# "which components does this pipeline actually declare" -- so it downloads
# first, and only the files under its declared component directories (plus
# small top-level metadata) are downloaded afterwards. This also lets the
# size cap reject an oversized repo BEFORE any weight bytes move.
#
# Legacy config patch, deliberate (same findings doc, blocking finding): repos
# whose model_index.json declares `_class_name: OnnxStableDiffusionPipeline`
# do not ship a config.json per component, but optimum-onnx's
# ORTStableDiffusionPipeline.from_pretrained requires one to know each
# component's architecture. The SD1.5 configs vendored under
# app/assets/generation/sd15_legacy_configs/ (same architectures: UNet2D-
# ConditionModel, CLIPTextModel, AutoencoderKL, safety checker) are copied in
# for any declared component missing its own config.json. Only for that
# legacy class: any other pipeline class either ships its own configs or
# fails functional validation below with an actionable message, rather than
# silently getting an unrelated config grafted onto it.
# ---------------------------------------------------------------------------

MODEL_INDEX_FILENAME = "model_index.json"
GENERATION_MODELS_SUBDIR = "generation"
SKIP_WEIGHT_SUFFIXES = (".ckpt", ".pth", ".safetensors", ".bin", ".msgpack", ".h5")
LEGACY_PIPELINE_CLASS = "OnnxStableDiffusionPipeline"
LEGACY_CONFIGS_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "generation" / "sd15_legacy_configs"
VALIDATION_PROMPT = "validation"
VALIDATION_SIZE = 64
VALIDATION_STEPS = 1


def _generation_model_id(repo_id: str) -> str:
    return "gen--" + repo_id.lower().replace("/", "--")


def _select_files(files: list[HfFile]) -> list[HfFile]:
    return [f for f in files if not f.path.lower().endswith(SKIP_WEIGHT_SUFFIXES)]


def _read_declared_components(staging_root: Path) -> list[str]:
    index = json.loads((staging_root / MODEL_INDEX_FILENAME).read_text(encoding="utf-8"))
    return [
        name
        for name, value in index.items()
        if not name.startswith("_") and isinstance(value, list)
    ]


def _filter_to_declared(files: list[HfFile], declared: list[str]) -> list[HfFile]:
    # Solo componentes declarados en model_index + metadata chica top-level.
    # Evita bajar carpetas ajenas al pipeline (ej. MXR/ binarios MIGraphX ~GBs,
    # controlnet/ no declarado) presentes en los repos amd/ (findings, repo id real).
    kept: list[HfFile] = []
    for hf_file in files:
        if hf_file.path == MODEL_INDEX_FILENAME:
            continue  # se descarga aparte, antes que el resto
        top_segment = hf_file.path.split("/", 1)[0]
        if "/" in hf_file.path:
            if top_segment in declared:
                kept.append(hf_file)
        elif hf_file.path.lower().endswith((".json", ".txt")):
            kept.append(hf_file)
    return kept


def _is_inside(candidate: Path, root: Path) -> bool:
    return candidate.resolve().is_relative_to(root.resolve())


def _safe_staging_dest(staging_root: Path, relative_path: str) -> Path:
    # model_index.json declares its own component names (an attacker-
    # controlled repo file), and repo_files() lists whatever the repo
    # actually contains -- both feed _filter_to_declared, so a malicious repo
    # could otherwise smuggle a "declared component" like "../../etc" plus a
    # matching file path and have it written outside staging_root.
    dest = staging_root / relative_path
    if not _is_inside(dest, staging_root):
        raise ValueError(f"Archivo del repo escapa el directorio de staging: {relative_path!r}")
    return dest


def _patch_legacy_component_configs(staging_root: Path) -> None:
    # Los repos amd/ legacy (_class_name: OnnxStableDiffusionPipeline) no traen
    # config.json por componente y optimum-onnx lo exige (findings, hallazgo
    # bloqueante). Se completan desde los configs SD1.5 vendorizados en
    # app/assets/generation/sd15_legacy_configs/. Solo para esa clase legacy:
    # otros layouts o traen sus configs o fallan la validacion funcional con
    # mensaje accionable.
    index = json.loads((staging_root / MODEL_INDEX_FILENAME).read_text(encoding="utf-8"))
    if index.get("_class_name") != LEGACY_PIPELINE_CLASS:
        return
    for component in _read_declared_components(staging_root):
        component_dir = staging_root / component
        if not _is_inside(component_dir, staging_root):
            # Nombre de componente (atacante-controlado via model_index.json)
            # intenta escapar staging_root -- salteo silencioso: la validacion
            # estructural de mas adelante ya falla con su propio mensaje.
            continue
        config_path = component_dir / "config.json"
        vendored = LEGACY_CONFIGS_ASSETS_DIR / component / "config.json"
        if component_dir.is_dir() and not config_path.exists() and vendored.is_file():
            shutil.copyfile(vendored, config_path)


def _ensure_model_index_listed(files: list[HfFile], repo_id: str) -> None:
    if not any(f.path == MODEL_INDEX_FILENAME for f in files):
        raise ValueError(
            f"El repo {repo_id!r} no parece un pipeline diffusers ONNX: falta {MODEL_INDEX_FILENAME}."
        )


def _ensure_size_cap(files: list[HfFile], cap_mb: int) -> None:
    total = sum(f.size for f in files)
    if total > cap_mb * 1024 * 1024:
        raise ValueError(
            f"La descarga ({total // (1024 * 1024)} MB) supera el límite de {cap_mb} MB "
            "(MAX_GENERATION_MODEL_DOWNLOAD_MB)."
        )


def _validate_structure(staging_root: Path) -> None:
    index_path = staging_root / MODEL_INDEX_FILENAME
    if not index_path.is_file():
        raise ValueError(f"Descarga incompleta: falta {MODEL_INDEX_FILENAME}.")
    declared = _read_declared_components(staging_root)
    missing = sorted(name for name in declared if not (staging_root / name).is_dir())
    if missing:
        raise ValueError(f"Faltan componentes del pipeline en el repo: {', '.join(missing)}.")


class GenerationModelInstaller:
    def __init__(self, settings: Settings, registry: ModelRegistry, hf_client: HfClient) -> None:
        self.settings = settings
        self.registry = registry
        self.hf_client = hf_client
        self._queue: asyncio.Queue[InstallJob] = asyncio.Queue()
        self._jobs: dict[str, InstallJob] = {}
        self._worker_task: asyncio.Task | None = None
        self._model_locks: dict[str, asyncio.Lock] = {}

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(
                self._worker(), name="generation-install-worker"
            )

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None

    async def install_from_hf(self, repo_id: str) -> str:
        available, reason = generation_dependencies_available()
        if not available:
            raise ValueError(reason or "Generation dependencies missing")
        validated = _validate_repo_id(repo_id)
        job = InstallJob(id=uuid.uuid4().hex, repo_id=validated)
        self._jobs[job.id] = job
        await self._queue.put(job)
        return job.id

    def status(self, install_id: str) -> InstallJob | None:
        return self._jobs.get(install_id)

    def _lock_for(self, model_id: str) -> asyncio.Lock:
        lock = self._model_locks.get(model_id)
        if lock is None:
            lock = asyncio.Lock()
            self._model_locks[model_id] = lock
        return lock

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            await self._run_install(job)
            self._queue.task_done()

    async def _process_next(self) -> bool:
        try:
            job = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return False
        await self._run_install(job)
        self._queue.task_done()
        return True

    async def _run_install(self, job: InstallJob) -> None:
        try:
            await self._download_and_register(job)
        except Exception as exc:  # noqa: BLE001 - el job reporta cualquier fallo
            job.status = InstallStatus.error
            job.error = str(exc)

    async def _download_and_register(self, job: InstallJob) -> None:
        files = await self.hf_client.repo_files(job.repo_id)
        _ensure_model_index_listed(files, job.repo_id)

        model_id = _generation_model_id(job.repo_id)
        staging_root = self.settings.temp_path / f"gen-staging-{model_id}"
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        staging_root.mkdir(parents=True, exist_ok=True)

        max_file_bytes = self.settings.max_generation_model_download_mb * 1024 * 1024
        job.status = InstallStatus.downloading
        try:
            # Fase 1: model_index.json primero (KBs) para conocer los componentes
            # declarados y filtrar la descarga a lo que el pipeline realmente usa.
            await self.hf_client.download(
                job.repo_id,
                MODEL_INDEX_FILENAME,
                _safe_staging_dest(staging_root, MODEL_INDEX_FILENAME),
                max_bytes=max_file_bytes,
            )
            declared = _read_declared_components(staging_root)
            selected = _filter_to_declared(_select_files(files), declared)
            _ensure_size_cap(selected, self.settings.max_generation_model_download_mb)

            total_bytes = sum(f.size for f in selected) or 1
            downloaded_bytes = 0
            for hf_file in selected:
                dest = _safe_staging_dest(staging_root, hf_file.path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                await self.hf_client.download(
                    job.repo_id, hf_file.path, dest, max_bytes=max_file_bytes
                )
                downloaded_bytes += hf_file.size
                job.progress_pct = round(downloaded_bytes / total_bytes * 100, 1)

            _validate_structure(staging_root)
            _patch_legacy_component_configs(staging_root)
            job.status = InstallStatus.validating
            await asyncio.to_thread(self._validate_pipeline, staging_root)

            final_dir = (
                self.settings.models_path / GENERATION_MODELS_SUBDIR / model_id
            )
            async with self._lock_for(model_id):
                await self._promote_staging_dir(staging_root, final_dir)
                entry = ModelEntry(
                    id=model_id,
                    name=job.repo_id,
                    kind=ModelKind.diffusion_onnx,
                    source=f"hf:{job.repo_id}",
                    size_bytes=sum(f.size for f in selected),
                    scale=None,
                    file_path=f"{GENERATION_MODELS_SUBDIR}/{model_id}",
                    status=ModelStatus.installed,
                )
                self.registry.register(entry)
            job.model_id = model_id
            job.status = InstallStatus.installed
        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root, ignore_errors=True)

    async def _promote_staging_dir(self, staging_root: Path, final_dir: Path) -> None:
        # Move-aside + rollback, not delete-then-replace: deleting final_dir
        # up front means a permanently-locked staging->final replace (a real
        # Windows file-lock case in this repo, see PROMOTE_RETRY_DELAYS_SECONDS)
        # loses BOTH the previous working install and the new staging build.
        # The previous install is parked at final_dir + ".old" until the
        # replace actually succeeds, and restored on any failure.
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        backup_dir = final_dir.with_name(final_dir.name + ".old")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        had_previous = final_dir.exists()
        if had_previous:
            final_dir.replace(backup_dir)
        try:
            await self._replace_with_retries(staging_root, final_dir)
        except Exception:
            if had_previous and not final_dir.exists():
                backup_dir.replace(final_dir)
            raise
        if had_previous:
            shutil.rmtree(backup_dir, ignore_errors=True)

    async def _replace_with_retries(self, staging_root: Path, final_dir: Path) -> None:
        last_error: Exception | None = None
        for delay in (0.0, *PROMOTE_RETRY_DELAYS_SECONDS):
            if delay:
                await asyncio.sleep(delay)
            try:
                staging_root.replace(final_dir)
                return
            except PermissionError as exc:
                last_error = exc
        raise RuntimeError(f"Could not promote generation model into place: {last_error}")

    def _validate_pipeline(self, pipeline_dir: Path) -> None:
        pipeline = None
        try:
            pipeline = self._create_validation_pipeline(pipeline_dir)
            pipeline(
                prompt=VALIDATION_PROMPT,
                num_inference_steps=VALIDATION_STEPS,
                width=VALIDATION_SIZE,
                height=VALIDATION_SIZE,
            )
        except Exception as exc:
            raise _wrap_generation_error(exc) from exc
        finally:
            del pipeline
            gc.collect()

    def _create_validation_pipeline(self, pipeline_dir: Path) -> Any:
        pipeline_cls = _load_pipeline_class()
        kwargs = _build_providers_for_validation(self.settings.default_device)
        return pipeline_cls.from_pretrained(str(pipeline_dir), **kwargs)
