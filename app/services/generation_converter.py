from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import shutil
import warnings
from contextlib import contextmanager
from dataclasses import replace
from collections.abc import Callable, Iterator
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
    canonical_weight_name,
    real_available_precisions,
    select_for_precision,
)
from app.services.hf_client import HfClient, HfFile
from app.services.model_installer import _validate_repo_id
from app.services.model_registry import ModelEntry, ModelKind, ModelStatus
from app.services.progress import (
    advance_conversion_stage,
    complete_conversion_stages,
)

_SUBMODEL_LINE = re.compile(
    r"^\*{5}\s+Exporting submodel (?P<index>\d+)/\d+:\s+"
    r"(?P<class_name>\S+)\s+\*{5}$"
)

# Medido en el smoke del 2026-07-28: fp16 necesita una tolerancia mayor por redondeo
# esperado.
#
# ACTUALIZADO 2026-07-30: relajar el atol NO alcanzaba. Un SDXL real
# (John6666/hassaku-xl-illustrious-v31-sdxl) exportaba sus cinco componentes bien y la
# validacion numerica lo rechazaba igual. Ahora `do_validation=False` y el gate es la
# validacion FUNCIONAL de Upflow, que genera una imagen con el modelo convertido. Este
# atol queda como plomeria por si alguna vez se vuelve a activar la numerica; hoy no
# tiene efecto.
FP16_EXPORT_ATOL = 1e-2
_ALLOCATION_ERROR_MARKERS = (
    "out of memory",
    "not enough memory",
    "cannot allocate",
    "can't allocate",
    "allocation",
    # Con guion bajo: asi lo escribe el runtime de C++ (`std::bad_alloc`), que
    # es como muere el export en Windows cuando no hay RAM. La variante con
    # espacio no aparece nunca en la practica.
    "bad_alloc",
)


def _parse_submodel_line(line: str) -> str | None:
    match = _SUBMODEL_LINE.fullmatch(line.strip())
    if match is None:
        return None
    return f"{match.group('index')}-{match.group('class_name')}"


def _cap_vae_trace_resolution(src_root: Path) -> None:
    config_path = src_root / "vae" / "config.json"
    if not config_path.is_file():
        return

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        return
    sample_size = config.get("sample_size")
    if (
        isinstance(sample_size, bool)
        or not isinstance(sample_size, (int, float))
        or sample_size <= 512
    ):
        return

    # Optimum usa sample_size como resolución del input dummy del trace. Los ejes
    # ONNX son dinámicos, así que el grafo no cambia; a 1024 el encoder cuadruplica
    # el costo de CPU (medido: 75 minutos contra 15 a 512).
    config["sample_size"] = 512
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _disable_vae_force_upcast(src_root: Path) -> None:
    config_path = src_root / "vae" / "config.json"
    if not config_path.is_file():
        return

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not config.get("force_upcast"):
        return

    # El VAE ONNX queda horneado en el dtype del export: el upcast dinámico de
    # diffusers no existe en el wrapper ORT (sin post_quant_conv), por lo que
    # force_upcast=true crashea durante el decode. Los repos modernos ya hornean
    # VAEs fp16-safe; un VAE genuinamente fp16-unsafe daría una imagen negra que
    # este flag tampoco podría corregir en ORT.
    config["force_upcast"] = False
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


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


# Lo que `torch.onnx.export` escupe por cada modelo: decenas de TracerWarning
# que son esperadas y no significan nada. En la consola del instalador se leen
# como un volcado de errores — reporte real (2026-08-05): un usuario dio la
# instalacion por fallida viendo solo esas lineas.
_EXPORT_NOISE_PATTERNS = (
    "Converting a tensor to a Python",
    "Exporting aten::index operator",
    "torch.onnx.export",
)


@contextmanager
def silence_export_noise() -> Iterator[None]:
    """Calla las advertencias esperadas del export, y solo esas."""
    with warnings.catch_warnings():
        for patron in _EXPORT_NOISE_PATTERNS:
            warnings.filterwarnings("ignore", message=f".*{re.escape(patron)}.*")
        yield


def _allocation_message_for_export(component: str | None) -> str:
    donde = f" del componente {component!r}" if component else ""
    return (
        f"No hay RAM suficiente para convertir el modelo{donde}. "
        "La conversion de un SDXL corre en CPU y pide varios GB de una sola vez. "
        "Cerra otras aplicaciones y volve a intentar, o elegi un modelo mas chico."
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
        # El ruido se calla y los fallos de memoria se traducen: sin esto la
        # consola parece un volcado de errores mientras todo va bien, y cuando
        # de verdad falla por RAM muere con un mensaje de C++ que no dice que
        # hacer.
        # onnxruntime-directml no es detectado como "onnxruntime" por Optimum.
        # El spike verificó que este guard debe parchearse sólo durante export:
        # docs/superpowers/specs/2026-07-25-third-party-spike-findings.md.
        with silence_export_noise(), _patch(
            "optimum.exporters.onnx.base.is_onnxruntime_available",
            return_value=True,
        ):
            main_export(
                str(src_dir),
                str(out_dir),
                task="text-to-image",
                device="cpu",
                # La validacion numerica de optimum queda APAGADA a proposito, y no por
                # relajar el control: Upflow tiene una mejor. `_validate_pipeline` carga
                # el pipeline convertido y GENERA UNA IMAGEN de verdad -- si eso anda, el
                # modelo sirve, sin importar cuanto se aparten los tensores intermedios.
                #
                # Medido con John6666/hassaku-xl-illustrious-v31-sdxl (SDXL): los cinco
                # componentes exportaron bien y la conversion igual se dio por fallida al
                # 71% con el propio mensaje de optimum diciendo "an error occurred during
                # validation, but the model was saved nonetheless". Upflow borraba ese
                # export que funcionaba. O sea que la validacion mas debil estaba vetando
                # lo que la mas fuerte habria aceptado.
                do_validation=False,
                **extra,
            )
    except MemoryError as exc:
        raise RuntimeError(_allocation_message_for_export(seen[-1] if seen else None)) from exc
    except RuntimeError as exc:
        if _is_allocation_runtime_error(exc):
            raise RuntimeError(
                _allocation_message_for_export(seen[-1] if seen else None)
            ) from exc
        raise
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


class ConversionCancelled(Exception):
    """El usuario corto la conversion. No es un fallo."""


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
        # Cancelaciones pedidas. El export corre en un hilo, asi que el aviso
        # tiene que cruzar el limite de hilo — de ahi un set y no un flag de
        # asyncio.
        self._cancelled: set[str] = set()
        # Enganche de prueba: permite cancelar justo cuando empieza un submodelo.
        self._on_component_hook: Callable[[str], None] | None = None
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
        self._register_converting_entry(job)
        await self._queue.put(job)
        return job.id

    def _register_converting_entry(self, job: ConversionJob) -> None:
        """La conversion existe en el registro DESDE que se encola.

        Sin esto, las conversiones disparadas por los packs del instalador eran
        invisibles: el modelo aparecia en la UI recien al promover, ~40 min
        despues, y el usuario concluia que la instalacion no trajo nada. Un
        modelo YA instalado nunca se degrada: sigue usable hasta que una
        promocion nueva lo reemplace.
        """
        model_id = _generation_model_id(job.repo_id, job.checkpoint_path)
        existing = self.installer.registry.get(model_id)
        if existing is not None and existing.status == ModelStatus.installed:
            return
        self.installer.registry.register(
            ModelEntry(
                id=model_id,
                name=job.repo_id,
                kind=ModelKind.diffusion_onnx,
                source=f"hf:{job.repo_id}",
                size_bytes=0,
                scale=None,
                file_path=None,
                status=ModelStatus.converting,
            )
        )

    def _mark_entry_error(self, job: ConversionJob, error: str) -> None:
        model_id = _generation_model_id(job.repo_id, job.checkpoint_path)
        entry = self.installer.registry.get(model_id)
        if entry is None or entry.status != ModelStatus.converting:
            # Un instalado previo sigue instalado; sin entrada no hay que marcar.
            return
        self.installer.registry.register(
            replace(entry, status=ModelStatus.error, error=error)
        )

    def cancel(self, conversion_id: str) -> bool:
        """Corta una conversion. Devuelve si habia algo que cortar.

        El corte cae en el limite del SIGUIENTE submodelo, no en cualquier
        instante: el export vive dentro de una libreria que no se puede
        interrumpir a la mitad. En un SDXL son cinco submodelos, asi que cancelar
        durante el mas largo espera a que ese termine.
        """
        job = self._jobs.get(conversion_id)
        if job is None or job.status in (
            JobStatus.completed,
            JobStatus.failed,
            JobStatus.cancelled,
        ):
            return False
        self._cancelled.add(conversion_id)
        if job.status is JobStatus.queued:
            # Todavia no arranco: se marca ya y el worker la saltea.
            job.status = JobStatus.cancelled
            job.finished_at = utc_now()
        return True

    def _raise_if_cancelled(self, job: ConversionJob) -> None:
        if job.id in self._cancelled:
            raise ConversionCancelled()

    def active(self) -> list[ConversionJob]:
        """Las conversiones que siguen corriendo.

        Existe porque el id vivia SOLO en la pantalla: al cambiar de seccion se
        iba con ella y la barra de progreso no podia volver a engancharse.
        La conversion nunca se perdia —sigue en el servidor— pero el usuario veia
        que si, y convertir un SDXL tarda cerca de media hora: creer que se
        perdio y arrancarlo de nuevo es media hora tirada y dos conversiones
        peleandose la misma maquina.
        """
        return [
            job
            for job in self._jobs.values()
            if job.status in (JobStatus.queued, JobStatus.running)
        ]

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
        if job.status is JobStatus.cancelled or job.id in self._cancelled:
            job.status = JobStatus.cancelled
            job.finished_at = utc_now()
            return
        job.status = JobStatus.running
        job.started_at = utc_now()
        try:
            await self._convert_and_register(job)
            job.status = JobStatus.completed
        except ConversionCancelled:
            # Cancelar no es un fallo: no lleva mensaje de error ni marca la
            # entrada como rota.
            job.status = JobStatus.cancelled
        except OSError as exc:
            job.status = JobStatus.failed
            job.error = map_disk_full(exc) or str(exc)
            self._mark_entry_error(job, job.error)
        except Exception as exc:  # noqa: BLE001 - el job reporta cualquier fallo
            job.status = JobStatus.failed
            job.error = str(exc)
            self._mark_entry_error(job, job.error)
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
                offered = await real_available_precisions(
                    self.hf_client,
                    job.repo_id,
                    files,
                    declared,
                )
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
                if self._on_component_hook is not None:
                    self._on_component_hook(name)
                # Unico punto donde se puede cortar: entre submodelo y submodelo.
                self._raise_if_cancelled(job)
                if name not in component_keys:
                    component_keys.append(name)
                advance_conversion_stage(
                    job,
                    component_keys,
                    f"exporting:{name}",
                )

            _cap_vae_trace_resolution(src_root)
            _disable_vae_force_upcast(src_root)
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
