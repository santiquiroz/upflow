from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch as _patch

from app.config import Settings
from app.models import ConversionJob, JobStatus, utc_now
from app.services.engines.generation_onnx import generation_dependencies_available
from app.services.generation_installer import (
    MODEL_INDEX_FILENAME,
    GenerationModelInstaller,
    _ensure_installable_layout,
    _ensure_model_index_listed,
    _generation_model_id,
    _read_declared_components,
    _safe_staging_dest,
    map_disk_full,
)
from app.services.generation_single_file import (
    CheckpointVerdict,
    classify_checkpoint,
    materialize,
    supported_architecture,
)
from app.services.generation_variants import (
    CONVERSION_SKIP_SUFFIXES,
    Precision,
    available_precisions,
    canonical_weight_name,
    select_for_precision,
)
from app.services.hf_client import HfClient, HfFile
from app.services.model_installer import _validate_repo_id
from app.services.progress import (
    advance_conversion_stage,
    complete_conversion_stages,
)

_SUBMODEL_LINE = re.compile(
    r"^\*{5}\s+Exporting submodel (?P<index>\d+)/\d+:\s+"
    r"(?P<class_name>\S+)\s+\*{5}$"
)

# Medido en el smoke del 2026-07-28: fp16 necesita una tolerancia mayor por
# redondeo esperado. La validacion sigue activa; solo se relaja su atol.
FP16_EXPORT_ATOL = 1e-2
_ALLOCATION_ERROR_MARKERS = (
    "out of memory",
    "not enough memory",
    "cannot allocate",
    "can't allocate",
    "allocation",
    "bad alloc",
)


def _parse_submodel_line(line: str) -> str | None:
    match = _SUBMODEL_LINE.fullmatch(line.strip())
    if match is None:
        return None
    return f"{match.group('index')}-{match.group('class_name')}"


def _selected_checkpoint(
    files: list[HfFile],
    repo_id: str,
    checkpoint_path: str,
) -> HfFile:
    candidates = [
        file
        for file in files
        if "/" not in file.path and file.path.lower().endswith(".safetensors")
    ]
    for file in candidates:
        if file.path == checkpoint_path:
            return file
    available = ", ".join(file.path for file in candidates) if candidates else "(ninguno)"
    raise ValueError(
        f"El checkpoint {checkpoint_path!r} no existe en la raiz del repo "
        f"{repo_id!r}. Candidatos: {available}."
    )


def _architecture_from_verdict(verdict: CheckpointVerdict) -> str:
    if (
        verdict.installable
        and verdict.architecture is not None
        and supported_architecture(verdict.architecture) is not None
    ):
        return verdict.architecture
    raise ValueError(verdict.diagnostic())


async def _read_checkpoint_architecture(
    hf_client: HfClient,
    repo_id: str,
    checkpoint_path: str,
) -> str | None:
    try:
        header, _header_length = await hf_client.read_safetensors_header(
            repo_id,
            checkpoint_path,
        )
    except Exception:  # noqa: BLE001 - install sigue habilitado si falla el preflight
        return None
    return _architecture_from_verdict(classify_checkpoint(header))


def _read_local_checkpoint_architecture(checkpoint_path: Path) -> str:
    with checkpoint_path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError("El checkpoint no contiene un prefijo safetensors valido.")
        header_length = int.from_bytes(length_bytes, byteorder="little", signed=False)
        header_bytes = handle.read(header_length)
    if len(header_bytes) != header_length:
        raise ValueError("El header safetensors local esta truncado.")
    header = json.loads(header_bytes)
    if not isinstance(header, dict):
        raise ValueError("El header safetensors local no es un objeto JSON.")
    return _architecture_from_verdict(classify_checkpoint(header))


def _advance_materializing_stage(job: ConversionJob) -> None:
    job.metadata["stage"] = "materializing"
    job.metadata["stages"] = [
        {
            "key": "downloading",
            "label": "Downloading checkpoint",
            "weight": 0.1,
            "status": "done",
        },
        {
            "key": "materializing",
            "label": "Materializing checkpoint",
            "weight": 0.05,
            "status": "active",
        },
        {
            "key": "exporting",
            "label": "Exporting to ONNX",
            "weight": 0.7,
            "status": "pending",
        },
        {
            "key": "validating",
            "label": "Validating pipeline",
            "weight": 0.15,
            "status": "pending",
        },
    ]
    job.metadata["progress"] = 0.1
    job.metadata["stageStartedAt"] = utc_now().isoformat()


def _allocation_message(checkpoint: HfFile) -> str:
    gib = checkpoint.size / (1024 ** 3)
    return (
        "No hay RAM suficiente para materializar el checkpoint "
        f"{checkpoint.path!r} de {checkpoint.size} bytes ({gib:.1f} GiB). "
        "Cerrá otras aplicaciones o elegí un checkpoint más pequeño y volvé a intentar."
    )


def _is_allocation_runtime_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _ALLOCATION_ERROR_MARKERS)


class _SubmodelProgressHandler(logging.Handler):
    def __init__(
        self,
        on_component: Callable[[str], None],
        seen: list[str],
    ) -> None:
        super().__init__(level=logging.INFO)
        self._on_component = on_component
        self._seen = seen

    def emit(self, record: logging.LogRecord) -> None:
        key = _parse_submodel_line(record.getMessage())
        if key is None or key in self._seen:
            return
        self._seen.append(key)
        self._on_component(key)


def _export_with_optimum(
    src_dir: Path,
    out_dir: Path,
    on_component: Callable[[str], None],
    dtype: str | None = None,
    atol: float | None = None,
) -> list[str]:
    # Imports perezosos: optimum (y su dependencia torch) nunca se cargan al
    # importar app.services.
    from optimum.exporters.onnx import main_export
    from optimum.utils import logging as optimum_logging

    seen: list[str] = []
    handler = _SubmodelProgressHandler(on_component, seen)
    optimum_logger = logging.getLogger("optimum")
    previous_verbosity = optimum_logging.get_verbosity()
    optimum_logger.addHandler(handler)
    extra: dict[str, Any] = {}
    if dtype is not None:
        extra["dtype"] = dtype
    if atol is not None:
        extra["atol"] = atol
    try:
        optimum_logging.set_verbosity_info()
        # onnxruntime-directml no es detectado como "onnxruntime" por Optimum.
        # El spike verificó que este guard debe parchearse sólo durante export:
        # docs/superpowers/specs/2026-07-25-third-party-spike-findings.md.
        with _patch(
            "optimum.exporters.onnx.base.is_onnxruntime_available",
            return_value=True,
        ):
            main_export(
                str(src_dir),
                str(out_dir),
                task="text-to-image",
                device="cpu",
                **extra,
            )
    finally:
        optimum_logger.removeHandler(handler)
        optimum_logging.set_verbosity(previous_verbosity)
    return seen or ["pipeline"]


ExportFn = Callable[
    [
        Path,
        Path,
        Callable[[str], None],
        str | None,
        float | None,
    ],
    list[str],
]


class GenerationModelConverter:
    """Cola single-worker de conversión, paralela al installer de generación.

    No reutiliza model_converter.py (Spandrel: un solo conv-net, segundos):
    exporta un pipeline diffusers completo multi-componente (minutos, RAM
    alta). El resultado pasa por installer.validate_and_promote, con la misma
    validación funcional y promoción atómica que un install ONNX nativo.

    Riesgo MVP aceptado: el export corre en CPU/RAM sin admisión por capacidad
    hasta que aterrice el subproyecto B. La única fase GPU es la validación
    funcional, ya protegida por device_semaphores en validate_and_promote.
    """

    def __init__(
        self,
        settings: Settings,
        installer: GenerationModelInstaller,
        hf_client: HfClient,
        export_fn: ExportFn | None = None,
    ) -> None:
        self.settings = settings
        self.installer = installer
        self.hf_client = hf_client
        self.export_fn = export_fn or _export_with_optimum
        self._queue: asyncio.Queue[ConversionJob] = asyncio.Queue()
        self._jobs: dict[str, ConversionJob] = {}
        self._worker_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(
                self._worker(),
                name="generation-convert-worker",
            )

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None

    async def convert_from_hf(
        self,
        repo_id: str,
        precision: Precision = "fp16",
        checkpoint_path: str | None = None,
    ) -> str:
        available, reason = generation_dependencies_available()
        if not available:
            raise ValueError(reason or "Generation dependencies missing")
        validated = _validate_repo_id(repo_id)
        job = ConversionJob(
            repo_id=validated,
            precision=precision,
            checkpoint_path=checkpoint_path,
        )
        self._jobs[job.id] = job
        await self._queue.put(job)
        return job.id

    def status(self, conversion_id: str) -> ConversionJob | None:
        return self._jobs.get(conversion_id)

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            await self._run_conversion(job)
            self._queue.task_done()

    async def _process_next(self) -> bool:
        try:
            job = self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return False
        await self._run_conversion(job)
        self._queue.task_done()
        return True

    async def _run_conversion(self, job: ConversionJob) -> None:
        job.status = JobStatus.running
        job.started_at = utc_now()
        try:
            await self._convert_and_register(job)
            job.status = JobStatus.completed
        except OSError as exc:
            job.status = JobStatus.failed
            job.error = map_disk_full(exc) or str(exc)
        except Exception as exc:  # noqa: BLE001 - el job reporta cualquier fallo
            job.status = JobStatus.failed
            job.error = str(exc)
        finally:
            job.finished_at = utc_now()

    async def _convert_and_register(self, job: ConversionJob) -> None:
        files = await self.hf_client.repo_files(job.repo_id)
        checkpoint: HfFile | None = None
        architecture: str | None = None
        if job.checkpoint_path is None:
            _ensure_model_index_listed(files, job.repo_id)
            _ensure_installable_layout(files, job.repo_id)
        else:
            checkpoint = _selected_checkpoint(
                files,
                job.repo_id,
                job.checkpoint_path,
            )
            _ensure_installable_layout(files, job.repo_id, job.checkpoint_path)
            architecture = await _read_checkpoint_architecture(
                self.hf_client,
                job.repo_id,
                job.checkpoint_path,
            )
        model_id = _generation_model_id(job.repo_id, job.checkpoint_path)
        src_root = self.settings.temp_path / f"genconv-src-{model_id}"
        out_root = self.settings.temp_path / f"genconv-onnx-{model_id}"
        for root in (src_root, out_root):
            if root.exists():
                shutil.rmtree(root, ignore_errors=True)
            root.mkdir(parents=True, exist_ok=True)

        component_keys: list[str] = []
        try:
            advance_conversion_stage(job, component_keys, "downloading")
            selected: list[HfFile]
            if checkpoint is not None:
                checkpoint_dest = _safe_staging_dest(src_root, checkpoint.path)
                checkpoint_dest.parent.mkdir(parents=True, exist_ok=True)
                await self.hf_client.download(
                    job.repo_id,
                    checkpoint.path,
                    checkpoint_dest,
                    unlimited=True,
                )
                _advance_materializing_stage(job)
                if architecture is None:
                    architecture = await asyncio.to_thread(
                        _read_local_checkpoint_architecture,
                        checkpoint_dest,
                    )
                try:
                    await asyncio.to_thread(
                        materialize,
                        checkpoint_dest,
                        src_root,
                        architecture,
                    )
                except MemoryError as exc:
                    raise RuntimeError(_allocation_message(checkpoint)) from exc
                except RuntimeError as exc:
                    if _is_allocation_runtime_error(exc):
                        raise RuntimeError(_allocation_message(checkpoint)) from exc
                    raise
                selected = [checkpoint]
                precision = job.precision
            else:
                await self.hf_client.download(
                    job.repo_id,
                    MODEL_INDEX_FILENAME,
                    _safe_staging_dest(src_root, MODEL_INDEX_FILENAME),
                    unlimited=True,
                )
                declared = _read_declared_components(src_root)
                offered = available_precisions(files)
                precision = (
                    job.precision
                    if job.precision in offered
                    else (offered[0] if offered else "fp32")
                )
                selected = select_for_precision(files, declared, precision)
                for hf_file in selected:
                    dest = _safe_staging_dest(
                        src_root,
                        canonical_weight_name(hf_file.path),
                    )
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    await self.hf_client.download(
                        job.repo_id,
                        hf_file.path,
                        dest,
                        unlimited=True,
                    )

            def on_component(name: str) -> None:
                if name not in component_keys:
                    component_keys.append(name)
                advance_conversion_stage(
                    job,
                    component_keys,
                    f"exporting:{name}",
                )

            exported = await asyncio.to_thread(
                self.export_fn,
                src_root,
                out_root,
                on_component,
                "fp16" if precision == "fp16" else None,
                FP16_EXPORT_ATOL if precision == "fp16" else None,
            )
            advance_conversion_stage(job, exported, "validating")
            size_bytes = sum(hf_file.size for hf_file in selected)
            job.model_id = await self.installer.validate_and_promote(
                out_root,
                job.repo_id,
                size_bytes,
                checkpoint_path=job.checkpoint_path,
            )
            complete_conversion_stages(job, exported)
        finally:
            for root in (src_root, out_root):
                if root.exists():
                    shutil.rmtree(root, ignore_errors=True)
