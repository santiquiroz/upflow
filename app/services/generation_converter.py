from __future__ import annotations

import asyncio
import contextlib
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
from app.services.generation_variants import (
    CONVERSION_SKIP_SUFFIXES,
    Precision,
    available_precisions,
    canonical_weight_name,
    select_for_precision,
)
from app.services.hf_client import HfClient
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


def _parse_submodel_line(line: str) -> str | None:
    match = _SUBMODEL_LINE.fullmatch(line.strip())
    if match is None:
        return None
    return f"{match.group('index')}-{match.group('class_name')}"


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
    ) -> str:
        available, reason = generation_dependencies_available()
        if not available:
            raise ValueError(reason or "Generation dependencies missing")
        validated = _validate_repo_id(repo_id)
        job = ConversionJob(repo_id=validated, precision=precision)
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
        _ensure_model_index_listed(files, job.repo_id)
        _ensure_installable_layout(files, job.repo_id)
        model_id = _generation_model_id(job.repo_id)
        src_root = self.settings.temp_path / f"genconv-src-{model_id}"
        out_root = self.settings.temp_path / f"genconv-onnx-{model_id}"
        for root in (src_root, out_root):
            if root.exists():
                shutil.rmtree(root, ignore_errors=True)
            root.mkdir(parents=True, exist_ok=True)

        component_keys: list[str] = []
        try:
            advance_conversion_stage(job, component_keys, "downloading")
            await self.hf_client.download(
                job.repo_id,
                MODEL_INDEX_FILENAME,
                _safe_staging_dest(src_root, MODEL_INDEX_FILENAME),
                unlimited=True,
            )
            declared = _read_declared_components(src_root)
            offered = available_precisions(files)
            precision: Precision = (
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
            )
            complete_conversion_stages(job, exported)
        finally:
            for root in (src_root, out_root):
                if root.exists():
                    shutil.rmtree(root, ignore_errors=True)
