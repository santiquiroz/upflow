from __future__ import annotations

from typing import Protocol


class ResourceProbe(Protocol):
    """A read-only probe of free capacity (in MB) for one device kind.

    `None` means "can't measure this right now" -- callers (DeviceSemaphores)
    treat that as fail-open: the resource check is skipped entirely for that
    call, never blocking a job that today's job-count-only gate would admit.
    """

    def free_capacity_mb(self, device_id: str) -> int | None: ...


class NullProbe:
    """Extension point for device kinds with no real probe yet (npu today,
    and any future kind on a platform this project doesn't target yet).
    Always fail-open -- DeviceSemaphores behaves exactly as if no probe were
    registered for that kind at all."""

    def free_capacity_mb(self, device_id: str) -> int | None:
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
