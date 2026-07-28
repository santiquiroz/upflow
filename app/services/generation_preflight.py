from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.exceptions import HfAuthError
from app.services.generation_compat import CompatVerdict, classify
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

    def build(
        compat: CompatVerdict | None,
        reason: str | None,
        degraded: bool,
        precisions: list[PrecisionCost] | None = None,
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
