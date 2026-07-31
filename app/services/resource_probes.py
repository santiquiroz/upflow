from __future__ import annotations

from typing import Protocol

from app.services import devices_service


class ResourceProbe(Protocol):
    """A read-only probe of free capacity (in MB) for one device kind.

    `None` means "can't measure this right now" -- callers (DeviceSemaphores)
    treat that as fail-open: the resource check is skipped entirely for that
    call, never blocking a job that today's job-count-only gate would admit.

    `own_usage_mb` es lo que ESTE proceso retiene del recurso: permite
    distinguir "el faltante es nuestro propio cache ocioso, reciclable" de
    "lo come un proceso externo" en la admision. None = no medible -> el
    bypass por cache propio no se habilita (fail-safe al comportamiento
    de solo-espera).
    """

    def free_capacity_mb(self, device_id: str) -> int | None: ...

    def own_usage_mb(self, device_id: str) -> int | None: ...


class NullProbe:
    """Extension point for device kinds with no real probe yet (npu today,
    and any future kind on a platform this project doesn't target yet).
    Always fail-open -- DeviceSemaphores behaves exactly as if no probe were
    registered for that kind at all."""

    def free_capacity_mb(self, device_id: str) -> int | None:
        return None

    def own_usage_mb(self, device_id: str) -> int | None:
        return None


def _probe_psutil_available_mb() -> int | None:
    # Lazy + tolerant import, same pattern as devices_service._probe_onnxruntime:
    # keeps the app/tests working even if psutil is somehow absent (fresh
    # checkout not yet pip install-ed, trimmed embedded-Python build).
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return None
    return int(psutil.virtual_memory().available // (1024 * 1024))


class SystemRamProbe:
    """Real system RAM probe for cpu-kind devices."""

    def free_capacity_mb(self, device_id: str) -> int | None:
        return _probe_psutil_available_mb()

    def own_usage_mb(self, device_id: str) -> int | None:
        try:
            import psutil  # type: ignore[import-not-found]
        except ImportError:
            return None
        return int(psutil.Process().memory_info().rss // (1024 * 1024))


_DML_DEVICE_PREFIX = "dml:"


def _dml_adapter_index(device_id: str) -> int | None:
    if not device_id.startswith(_DML_DEVICE_PREFIX):
        return None
    try:
        return int(device_id[len(_DML_DEVICE_PREFIX):])
    except ValueError:
        return None


class DxgiVramProbe:
    """Real VRAM probe for gpu-kind devices (dml:N), Windows/DirectML only.
    Thin adapter: parses the device_id and delegates the actual DXGI query
    to devices_service, which owns the ctypes/COM bindings."""

    def free_capacity_mb(self, device_id: str) -> int | None:
        adapter_index = _dml_adapter_index(device_id)
        if adapter_index is None:
            return None
        info = devices_service._query_adapter_vram_info_mb(adapter_index)
        return None if info is None else info[0]

    def own_usage_mb(self, device_id: str) -> int | None:
        adapter_index = _dml_adapter_index(device_id)
        if adapter_index is None:
            return None
        info = devices_service._query_adapter_vram_info_mb(adapter_index)
        return None if info is None else info[1]
