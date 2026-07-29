from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.exceptions import HfAuthError
from app.services.generation_compat import CompatVerdict, classify
from app.services.generation_single_file import (
    CheckpointVerdict,
    classify_checkpoint,
)
from app.services.generation_variants import (
    MODEL_INDEX_FILENAME,
    Precision,
    available_precisions,
    select_for_precision,
)
from app.services.vram_estimate import estimate_peak_bytes

MB = 1024 * 1024


@dataclass(slots=True, frozen=True)
class PrecisionCost:
    precision: Precision
    download_bytes: int
    estimated_peak_bytes: int


@dataclass(slots=True, frozen=True)
class DeviceCapacity:
    id: str
    name: str
    kind: str
    free_vram_bytes: int | None


@dataclass(slots=True, frozen=True)
class DiskCapacity:
    target_path: str
    free_bytes: int


@dataclass(slots=True, frozen=True)
class CheckpointCandidate:
    path: str
    size_bytes: int
    architecture: str | None
    installable: bool | None
    reason: str


@dataclass(slots=True, frozen=True)
class PreflightReport:
    repo_id: str
    compat: CompatVerdict | None
    compat_reason: str | None
    degraded: bool
    reference_width: int
    reference_height: int
    precisions: list[PrecisionCost] = field(default_factory=list)
    devices: list[DeviceCapacity] = field(default_factory=list)
    disk: DiskCapacity | None = None
    checkpoints: list[CheckpointCandidate] = field(default_factory=list)
    free_ram_bytes: int | None = None


def _measure_disk(target: Path) -> DiskCapacity | None:
    # El directorio puede no existir todavia en una instalacion nueva: se sube
    # al primer ancestro que exista antes de medir.
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return DiskCapacity(target_path=str(target), free_bytes=shutil.disk_usage(probe).free)
    except OSError:
        return None


def _measure_devices(devices_service: Any, probes: dict[str, Any]) -> list[DeviceCapacity]:
    rows: list[DeviceCapacity] = []
    for info in devices_service.list_devices():
        probe = probes.get(info["kind"])
        free_mb = probe.free_capacity_mb(info["id"]) if probe is not None else None
        rows.append(
            DeviceCapacity(
                id=info["id"],
                name=info.get("name") or info["id"],
                kind=info["kind"],
                free_vram_bytes=None if free_mb is None else free_mb * MB,
            )
        )
    return rows


def _measure_free_ram(probes: dict[str, Any]) -> int | None:
    probe = probes.get("cpu")
    if probe is None:
        return None
    free_mb = probe.free_capacity_mb("cpu")
    return None if free_mb is None else free_mb * MB


def _candidate_from_verdict(
    path: str,
    size_bytes: int,
    verdict: CheckpointVerdict,
) -> CheckpointCandidate:
    return CheckpointCandidate(
        path=path,
        size_bytes=size_bytes,
        architecture=verdict.architecture,
        installable=verdict.installable,
        reason=verdict.reason,
    )


async def _read_declared(hf_client: Any, repo_id: str) -> list[str]:
    scratch = Path(tempfile.mkdtemp(prefix="upflow-preflight-"))
    try:
        dest = scratch / MODEL_INDEX_FILENAME
        await hf_client.download(repo_id, MODEL_INDEX_FILENAME, dest, unlimited=True)
        index = json.loads(dest.read_text(encoding="utf-8"))
        return [
            name
            for name, value in index.items()
            if not name.startswith("_") and isinstance(value, list)
        ]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


async def preflight(
    hf_client: Any,
    devices_service: Any,
    settings: Any,
    probes: dict[str, Any],
    repo_id: str,
    width: int = 512,
    height: int = 512,
) -> PreflightReport:
    # Los dispositivos y el disco no dependen de Hugging Face, asi que se miden
    # aunque la parte de red falle: un reporte degradado sigue siendo util.
    devices = _measure_devices(devices_service, probes)
    disk = _measure_disk(Path(settings.temp_path))
    free_ram_bytes = _measure_free_ram(probes)

    def build(
        compat: CompatVerdict | None,
        reason: str | None,
        degraded: bool,
        precisions: list[PrecisionCost] | None = None,
        checkpoints: list[CheckpointCandidate] | None = None,
    ) -> PreflightReport:
        return PreflightReport(
            repo_id=repo_id,
            compat=compat,
            compat_reason=reason,
            degraded=degraded,
            reference_width=width,
            reference_height=height,
            precisions=precisions or [],
            devices=devices,
            disk=disk,
            checkpoints=checkpoints or [],
            free_ram_bytes=free_ram_bytes,
        )

    try:
        files = await hf_client.repo_files(repo_id)
    except HfAuthError as exc:
        # 401/403 es un VEREDICTO, no una falla de medicion: el repo es gated y
        # eso es exactamente lo que hay que decirle al usuario. `classify` no
        # puede detectarlo desde aca porque repo_files no devuelve el flag
        # `gated` -- el error de auth ES la senal. Si el usuario tiene un token
        # valido, repo_files funciona y el repo se clasifica por su contenido.
        return build("gated", str(exc), False)
    except Exception:  # noqa: BLE001 - el pre-flight es diagnostico: nunca propaga
        return build(None, None, True)

    verdict, reason = classify(tuple(f.path for f in files), None)
    if verdict == "single_file":
        candidates: list[CheckpointCandidate] = []
        for file in files:
            if "/" in file.path or not file.path.lower().endswith(".safetensors"):
                continue
            try:
                header, _header_length = await hf_client.read_safetensors_header(
                    repo_id,
                    file.path,
                )
                candidates.append(
                    _candidate_from_verdict(
                        file.path,
                        file.size,
                        classify_checkpoint(header),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - falla aislada por candidato
                detail = str(exc).strip() or type(exc).__name__
                candidates.append(
                    CheckpointCandidate(
                        path=file.path,
                        size_bytes=file.size,
                        architecture=None,
                        installable=None,
                        reason=f"No se pudo evaluar el header: {detail}",
                    )
                )
        return build(verdict, reason, False, checkpoints=candidates)

    # Clasificar ANTES de leer model_index.json: un repo `incompatible` es
    # justamente el que no lo tiene, y pedirlo daria un 404 que degradaria el
    # reporte y taparia el veredicto que ya conocemos. `ready_onnx` tampoco lo
    # necesita: no ofrece eleccion de precision (ver el spec, alcance de B).
    if verdict != "needs_conversion":
        return build(verdict, reason, False)

    try:
        declared = await _read_declared(hf_client, repo_id)
    except HfAuthError as exc:
        return build("gated", str(exc), False)
    except Exception:  # noqa: BLE001
        # El veredicto se conserva: se sabe que necesita conversion, solo no se
        # pudo poner precio a cada precision.
        return build(verdict, reason, True)

    costs = [
        PrecisionCost(
            precision=precision,
            download_bytes=(total := sum(f.size for f in select_for_precision(files, declared, precision))),
            estimated_peak_bytes=estimate_peak_bytes(total, width, height),
        )
        for precision in available_precisions(files)
    ]
    return build(verdict, reason, False, costs)
