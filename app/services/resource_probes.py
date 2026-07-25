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
