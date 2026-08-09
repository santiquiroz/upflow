from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.exceptions import HfAuthError
from app.services.compat_strategy import CompatStrategy
from app.services.generation_compat import CompatReason, CompatVerdict
from app.services.hf_client import pick_weight_file

MB = 1024 * 1024

# Las mediciones de capacidad no tienen nada de especifico de un dominio: un
# gigabyte libre es un gigabyte libre. Lo que cambia por dominio es la
# CLASIFICACION del repo, que entra por la estrategia.


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


def measure_disk(target: Path) -> DiskCapacity | None:
    # El directorio puede no existir todavia en una instalacion nueva: se sube
    # al primer ancestro que exista antes de medir.
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return DiskCapacity(target_path=str(target), free_bytes=shutil.disk_usage(probe).free)
    except OSError:
        return None


def measure_devices(devices_service: Any, probes: dict[str, Any]) -> list[DeviceCapacity]:
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


def measure_free_ram(probes: dict[str, Any]) -> int | None:
    probe = probes.get("cpu")
    if probe is None:
        return None
    free_mb = probe.free_capacity_mb("cpu")
    return None if free_mb is None else free_mb * MB


# ---------------------------------------------------------------------------
# Esqueleto comun de los pre-flights
#
# Todos miden lo mismo (dispositivos, disco, RAM) y clasifican el repo con la
# estrategia de su dominio; lo unico que cambia es el reporte que arman despues.
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class CapacitySnapshot:
    devices: list[DeviceCapacity]
    disk: DiskCapacity | None
    free_ram_bytes: int | None


def measure_capacity(
    devices_service: Any, probes: dict[str, Any], disk_target: Path
) -> CapacitySnapshot:
    # Los dispositivos y el disco no dependen de Hugging Face, asi que se miden
    # aunque la parte de red falle: un reporte degradado sigue siendo util.
    return CapacitySnapshot(
        devices=measure_devices(devices_service, probes),
        disk=measure_disk(disk_target),
        free_ram_bytes=measure_free_ram(probes),
    )


@dataclass(slots=True, frozen=True)
class RepoClassification:
    verdict: CompatVerdict | None
    reason: CompatReason | None
    degraded: bool
    # None cuando ni el listado del repo se pudo leer (gated o red caida): no
    # hay nada que enriquecer por dominio.
    files: list[Any] | None


async def classify_repo(
    hf_client: Any, repo_id: str, strategy: CompatStrategy
) -> RepoClassification:
    try:
        files = await hf_client.repo_files(repo_id)
    except HfAuthError as exc:
        # 401/403 es un VEREDICTO, no una falla de medicion: el repo es gated y
        # eso es exactamente lo que hay que decirle al usuario. `classify` no
        # puede detectarlo desde aca porque repo_files no devuelve el flag
        # `gated` -- el error de auth ES la senal. Si el usuario tiene un token
        # valido, repo_files funciona y el repo se clasifica por su contenido.
        # El detalle es el texto del 401/403: dato, no copia.
        return RepoClassification(
            "gated", CompatReason("compat.gatedAuth", {"detail": str(exc)}), False, None
        )
    except Exception:  # noqa: BLE001 - el pre-flight es diagnostico: nunca propaga
        return RepoClassification(None, None, True, None)

    verdict, reason = strategy.classify(tuple(f.path for f in files), None)
    return RepoClassification(verdict, reason, False, files)


# ---------------------------------------------------------------------------
# Pre-flight de upscalers
#
# A proposito NO estima el pico de VRAM. El factor de vram_estimate asume que las
# activaciones crecen con la resolucion de salida, y para un upscaler que hace
# tiling esa premisa es falsa: el pico lo acota el tamaño del tile, no la imagen.
# Reusar ese numero aca daria un aviso confiado y equivocado, que es exactamente
# lo que el resto de la app evita. Se reporta la VRAM libre medida como DATO y no
# se emite veredicto de "no entra".
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class UpscalerPreflightReport:
    repo_id: str
    compat: CompatVerdict | None
    compat_reason_key: str | None
    degraded: bool
    compat_reason_params: dict[str, str] = field(default_factory=dict)
    download_bytes: int | None = None
    devices: list[DeviceCapacity] = field(default_factory=list)
    disk: DiskCapacity | None = None
    free_ram_bytes: int | None = None


def _download_bytes(files: list[Any]) -> int | None:
    try:
        return pick_weight_file(files).size
    except ValueError:
        # Repo sin ningun archivo de pesos: la clasificacion ya lo llama
        # incompatible, asi que no hay tamaño que reportar.
        return None


async def preflight_upscaler(
    hf_client: Any,
    devices_service: Any,
    settings: Any,
    probes: dict[str, Any],
    repo_id: str,
    strategy: CompatStrategy,
) -> UpscalerPreflightReport:
    capacity = measure_capacity(devices_service, probes, Path(settings.models_path))

    def build(
        compat: CompatVerdict | None,
        reason: CompatReason | None,
        degraded: bool,
        download_bytes: int | None = None,
    ) -> UpscalerPreflightReport:
        return UpscalerPreflightReport(
            repo_id=repo_id,
            compat=compat,
            compat_reason_key=None if reason is None else reason.key,
            compat_reason_params={} if reason is None else dict(reason.params),
            degraded=degraded,
            download_bytes=download_bytes,
            devices=capacity.devices,
            disk=capacity.disk,
            free_ram_bytes=capacity.free_ram_bytes,
        )

    repo = await classify_repo(hf_client, repo_id, strategy)
    if repo.files is None:
        return build(repo.verdict, repo.reason, repo.degraded)
    return build(repo.verdict, repo.reason, False, download_bytes=_download_bytes(repo.files))
