from __future__ import annotations

import asyncio
import gc
import json
import shutil
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from app.config import Settings
from app.models import InstallJob
from app.services.device_semaphores import DeviceSemaphores
from app.services.engines.generation_onnx import (
    _build_providers_for_validation,
    _load_pipeline_class,
    _read_declared_class_name,
    _wrap_generation_error,
    generation_dependencies_available,
)
from app.services.generation_compat import covered_by_onnx
from app.services.generation_staging import (
    MODEL_INDEX_FILENAME,
    ensure_installable_layout as _ensure_installable_layout,
    ensure_model_index_listed as _ensure_model_index_listed,
    generation_model_id as _generation_model_id,
    is_inside as _is_inside,
    map_disk_full,
    read_declared_components as _read_declared_components,
    root_checkpoint_paths as _root_checkpoint_paths,
    safe_staging_dest as _safe_staging_dest,
)
from app.services.generation_variants import Precision
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.hf_client import HfClient, HfFile
from app.services.install_queue_base import SingleWorkerJobQueue
from app.services.model_installer import (
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
# installer discover the exact pipeline layout before any weight bytes move.
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

GENERATION_MODELS_SUBDIR = "generation"
SKIP_WEIGHT_SUFFIXES = (".ckpt", ".pth", ".safetensors", ".bin", ".msgpack", ".h5")
LEGACY_PIPELINE_CLASS = "OnnxStableDiffusionPipeline"
LEGACY_CONFIGS_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "generation" / "sd15_legacy_configs"
VALIDATION_PROMPT = "validation"
VALIDATION_SIZE = 64
VALIDATION_STEPS = 1


class CheckpointNotFoundError(ValueError):
    pass


def _select_files(files: list[HfFile]) -> list[HfFile]:
    return [f for f in files if not f.path.lower().endswith(SKIP_WEIGHT_SUFFIXES)]


def _top_level_dirs(files: list[HfFile]) -> set[str]:
    return {f.path.split("/", 1)[0] for f in files if "/" in f.path}


def _dirs_with_suffix(
    files: list[HfFile], suffixes: tuple[str, ...]
) -> set[str]:
    return _top_level_dirs(
        [f for f in files if f.path.lower().endswith(suffixes)]
    )


def _needs_conversion(files: list[HfFile]) -> bool:
    # La decisión es por componente, no global: repos mixtos reales como
    # stabilityai/sdxl-turbo publican ONNX para unet pero sólo pesos PyTorch
    # para vae (findings doc sección (a):
    # docs/superpowers/specs/2026-07-25-third-party-spike-findings.md).
    # Descargar sólo el payload ONNX deja un pipeline parcial; convertir desde
    # los pesos torch produce todos los componentes. En un repo ONNX completo,
    # cada directorio que contiene pesos torch también tiene su propio ONNX.
    torch_dirs = _dirs_with_suffix(files, (".safetensors", ".bin"))
    onnx_dirs = _dirs_with_suffix(files, (".onnx",))
    return any(not covered_by_onnx(name, onnx_dirs) for name in torch_dirs)


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


def _ensure_checkpoint_listed(
    files: list[HfFile],
    repo_id: str,
    checkpoint_path: str,
) -> None:
    candidates = _root_checkpoint_paths(files)
    if checkpoint_path not in candidates:
        available = ", ".join(candidates) if candidates else "(ninguno)"
        raise CheckpointNotFoundError(
            f"El checkpoint {checkpoint_path!r} no existe en la raiz del repo "
            f"{repo_id!r}. Candidatos: {available}."
        )


# Un componente declarado por el pipeline de ORIGEN que en la exportacion ONNX se
# materializa con OTROS nombres. El VAE es el caso: optimum lo parte en encoder y
# decoder (sus propias constantes DIFFUSION_MODEL_VAE_{ENCODER,DECODER}_SUBFOLDER), asi
# que una carpeta `vae` no existe ni va a existir en un pipeline convertido.
#
# Sin esto, un SDXL exportaba sus cinco componentes correctamente y la validacion lo
# rechazaba con "Faltan componentes del pipeline en el repo: vae" -- estaba comparando
# la salida convertida contra los nombres del formato de entrada. Medido con
# John6666/hassaku-xl-illustrious-v31-sdxl.
_COMPONENT_ALIASES: dict[str, tuple[str, ...]] = {
    "vae": ("vae_encoder", "vae_decoder"),
    # SDXL publica ademas un VAE alternativo; el export produce un solo par.
    "vae_1_0": ("vae_encoder", "vae_decoder"),
}


def _component_is_present(staging_root: Path, name: str) -> bool:
    if (staging_root / name).is_dir():
        return True
    aliases = _COMPONENT_ALIASES.get(name)
    # Todas y no alguna: media exportacion del VAE es una exportacion rota.
    return bool(aliases) and all((staging_root / alias).is_dir() for alias in aliases)


def _validate_structure(staging_root: Path) -> None:
    index_path = staging_root / MODEL_INDEX_FILENAME
    if not index_path.is_file():
        raise ValueError(f"Descarga incompleta: falta {MODEL_INDEX_FILENAME}.")
    declared = _read_declared_components(staging_root)
    missing = sorted(
        name for name in declared if not _component_is_present(staging_root, name)
    )
    if missing:
        raise ValueError(f"Faltan componentes del pipeline en el repo: {', '.join(missing)}.")


class _ValidationSessionOwner:
    def release_device(self, device: str) -> None:
        # La sesion de validacion se descarta apenas termina _validate_pipeline
        # (ver finally: del pipeline) -- no hay nada que evacuar, pero el
        # protocolo GpuSessionOwner exige el metodo.
        pass


class GenerationModelInstaller(SingleWorkerJobQueue[InstallJob]):
    _worker_name = "generation-install-worker"
    _error_status = InstallStatus.error

    def __init__(
        self,
        settings: Settings,
        registry: ModelRegistry,
        hf_client: HfClient,
        gpu_coordinator: GpuSessionCoordinator,
        device_semaphores: DeviceSemaphores,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.registry = registry
        self.hf_client = hf_client
        self.gpu_coordinator = gpu_coordinator
        self.device_semaphores = device_semaphores
        self.enqueue_conversion: (
            Callable[[str, Precision, str | None], Awaitable[str]] | None
        ) = None

    async def install_from_hf(
        self,
        repo_id: str,
        precision: Precision = "fp16",
        checkpoint_path: str | None = None,
    ) -> str:
        available, reason = generation_dependencies_available()
        if not available:
            raise ValueError(reason or "Generation dependencies missing")
        validated = _validate_repo_id(repo_id)
        if checkpoint_path is not None:
            files = await self.hf_client.repo_files(validated)
            _ensure_checkpoint_listed(files, validated, checkpoint_path)
        return await self._enqueue(
            InstallJob(
                id=uuid.uuid4().hex,
                repo_id=validated,
                precision=precision,
                checkpoint_path=checkpoint_path,
                status=InstallStatus.downloading,
            )
        )

    async def _run(self, job: InstallJob) -> None:
        try:
            await self._download_and_register(job)
        except OSError as exc:
            job.status = InstallStatus.error
            job.error = map_disk_full(exc) or str(exc)
        except Exception as exc:  # noqa: BLE001 - el job reporta cualquier fallo
            self._fail_job(job, exc)

    async def _download_and_register(self, job: InstallJob) -> None:
        files = await self.hf_client.repo_files(job.repo_id)
        if job.checkpoint_path is None:
            _ensure_model_index_listed(files, job.repo_id)
        else:
            _ensure_checkpoint_listed(files, job.repo_id, job.checkpoint_path)
        _ensure_installable_layout(files, job.repo_id, job.checkpoint_path)
        if job.checkpoint_path is not None or _needs_conversion(files):
            # Pipeline con al menos un componente publicado sólo en PyTorch:
            # se auto-rutea a un job de conversión separado y visible.
            if self.enqueue_conversion is None:
                raise ValueError(
                    f"El repo {job.repo_id!r} publica al menos un componente "
                    "con pesos PyTorch pero sin ONNX propio y la conversión "
                    "no está disponible."
                )
            if job.checkpoint_path is None:
                job.conversion_id = await self.enqueue_conversion(
                    job.repo_id,
                    job.precision,
                )
            else:
                job.conversion_id = await self.enqueue_conversion(
                    job.repo_id,
                    job.precision,
                    job.checkpoint_path,
                )
            # Estado terminal del install job: el progreso real vive en el
            # conversion job y el frontend hace el hand-off con conversion_id.
            job.status = InstallStatus.converting
            return

        model_id = _generation_model_id(job.repo_id)
        staging_root = self.settings.temp_path / f"gen-staging-{model_id}"
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        staging_root.mkdir(parents=True, exist_ok=True)

        job.status = InstallStatus.downloading
        try:
            # Fase 1: model_index.json primero (KBs) para conocer los componentes
            # declarados y filtrar la descarga a lo que el pipeline realmente usa.
            await self.hf_client.download(
                job.repo_id,
                MODEL_INDEX_FILENAME,
                _safe_staging_dest(staging_root, MODEL_INDEX_FILENAME),
                unlimited=True,
            )
            declared = _read_declared_components(staging_root)
            selected = _filter_to_declared(_select_files(files), declared)

            total_bytes = sum(f.size for f in selected) or 1
            downloaded_bytes = 0
            for hf_file in selected:
                dest = _safe_staging_dest(staging_root, hf_file.path)
                dest.parent.mkdir(parents=True, exist_ok=True)
                await self.hf_client.download(
                    job.repo_id,
                    hf_file.path,
                    dest,
                    unlimited=True,
                )
                downloaded_bytes += hf_file.size
                job.progress_pct = round(downloaded_bytes / total_bytes * 100, 1)

            job.status = InstallStatus.validating
            job.model_id = await self.validate_and_promote(
                staging_root,
                job.repo_id,
                sum(f.size for f in selected),
            )
            job.status = InstallStatus.installed
        finally:
            if staging_root.exists():
                shutil.rmtree(staging_root, ignore_errors=True)

    async def validate_and_promote(
        self,
        staging_root: Path,
        repo_id: str,
        size_bytes: int,
        checkpoint_path: str | None = None,
        model_id: str | None = None,
        display_name: str | None = None,
    ) -> str:
        _validate_structure(staging_root)
        _patch_legacy_component_configs(staging_root)
        async with self.device_semaphores.acquire(self.settings.default_device):
            await asyncio.to_thread(self._validate_pipeline, staging_root)

        # model_id/display_name explícitos: los merges de inpainting registran
        # "<repo> (inpainting)" con id propio para no pisar el modelo original.
        model_id = model_id or _generation_model_id(repo_id, checkpoint_path)
        final_dir = self.settings.models_path / GENERATION_MODELS_SUBDIR / model_id
        async with self._lock_for(model_id):
            await self._promote_staging_dir(staging_root, final_dir)
            entry = ModelEntry(
                id=model_id,
                name=display_name or repo_id,
                kind=ModelKind.diffusion_onnx,
                source=f"hf:{repo_id}",
                size_bytes=size_bytes,
                scale=None,
                file_path=f"{GENERATION_MODELS_SUBDIR}/{model_id}",
                checkpoint_path=checkpoint_path,
                status=ModelStatus.installed,
            )
            self.registry.register(entry)
        return model_id

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
            self.gpu_coordinator.acquire(self.settings.default_device, _ValidationSessionOwner())
            pipeline = self._create_validation_pipeline(pipeline_dir)
            pipeline(**self._validation_call_kwargs(pipeline_dir))
        except Exception as exc:
            raise _wrap_generation_error(exc) from exc
        finally:
            del pipeline
            gc.collect()

    def _validation_call_kwargs(self, pipeline_dir: Path) -> dict[str, Any]:
        from app.services.generation_pipeline_modes import is_dedicated_inpaint_class

        kwargs: dict[str, Any] = {
            "prompt": VALIDATION_PROMPT,
            "num_inference_steps": VALIDATION_STEPS,
            "width": VALIDATION_SIZE,
            "height": VALIDATION_SIZE,
        }
        if is_dedicated_inpaint_class(_read_declared_class_name(pipeline_dir)):
            # Un unet 9ch exige imagen+máscara: validar sin ellas fallaría
            # SIEMPRE y el checkpoint dedicado quedaría ininstalable.
            from PIL import Image

            kwargs["image"] = Image.new("RGB", (VALIDATION_SIZE, VALIDATION_SIZE), (128, 128, 128))
            kwargs["mask_image"] = Image.new("L", (VALIDATION_SIZE, VALIDATION_SIZE), 255)
            # strength EXPLÍCITO: el default del pipeline es 0.9999 y con el
            # único paso de validación int(1*0.9999)=0 pasos — el merge entero
            # moría acá con "number of pipeline steps is 0" (visto real).
            kwargs["strength"] = 1.0
        return kwargs

    def _create_validation_pipeline(self, pipeline_dir: Path) -> Any:
        from app.services.generation_pipeline_modes import (
            is_dedicated_inpaint_class,
            load_inpaint_class,
        )

        declared = _read_declared_class_name(pipeline_dir)
        if is_dedicated_inpaint_class(declared):
            pipeline_cls = load_inpaint_class(declared)
        else:
            pipeline_cls = _load_pipeline_class(declared)
        kwargs = _build_providers_for_validation(self.settings.default_device)
        return pipeline_cls.from_pretrained(str(pipeline_dir), **kwargs)
