from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.config import Settings
from app.services.devices_service import CPU_DEVICE_ID


class DeviceSemaphores:
    """Per-device concurrency gate backed by a single shared condition.

    Each device_id (e.g. "cpu", "dml:0", "dml:1") has a configured capacity
    (`capacity_for`) and a live `in_flight` count. A job for a device only
    runs while `in_flight < capacity` for that device; jobs on different
    devices never gate each other. All accounting lives under ONE shared
    `asyncio.Condition`, and every release does `notify_all()`, so a waiter
    blocked on capacity for device X wakes the instant ANY device frees a
    permit -- crucially including a job releasing a device the waiter is not
    even interested in. That single release signal is what lets the
    auto-router (app/services/device_router.py) re-select the first device
    that frees instead of staying parked on a stale pick (no idle-device
    head-of-line blocking, no busy-wait).

    Implemented with manual counting + a condition rather than one
    asyncio.Semaphore per device precisely so that non-blocking "reserve iff
    free" (needed by the router's atomic pick) and the shared release signal
    share the exact same in_flight ledger and capacity limit as the blocking
    `acquire()` used by pinned-device jobs.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._in_flight: dict[str | None, int] = {}
        self._condition = asyncio.Condition()

    def capacity_for(self, device_id: str | None) -> int:
        if device_id == CPU_DEVICE_ID:
            return self._settings.cpu_concurrency
        return self._settings.per_device_gpu_concurrency

    def in_flight(self, device_id: str | None) -> int:
        return self._in_flight.get(device_id, 0)

    def free_capacity(self, device_id: str | None) -> int:
        return self.capacity_for(device_id) - self.in_flight(device_id)

    @property
    def release_condition(self) -> asyncio.Condition:
        """The shared condition the auto-router waits on for a freed permit.

        A caller reserving through it MUST hold it while checking
        `free_capacity` and calling `reserve`, so the check-and-take is
        atomic against every other acquirer under asyncio's cooperative
        scheduling.
        """
        return self._condition

    def reserve(self, device_id: str | None) -> None:
        """Take one permit unconditionally. Caller MUST hold release_condition
        AND have already confirmed `free_capacity(device_id) > 0` under it."""
        self._in_flight[device_id] = self.in_flight(device_id) + 1

    def _reserve_if_free(self, device_id: str | None) -> bool:
        if self.free_capacity(device_id) <= 0:
            return False
        self.reserve(device_id)
        return True

    async def release(self, device_id: str | None) -> None:
        async with self._condition:
            self._in_flight[device_id] = self.in_flight(device_id) - 1
            # Wake every waiter (pinned acquirers AND auto-router pickers): the
            # freed device may be exactly what an otherwise-idle picker was
            # waiting for, even if it's not the device it originally eyed.
            self._condition.notify_all()

    @asynccontextmanager
    async def acquire(self, device_id: str | None) -> AsyncIterator[None]:
        # Blocking reserve for a pinned device: wait until this specific
        # device has a free permit, take it, then run the job WITHOUT holding
        # the condition (it's a leaf lock, held only during reserve/release).
        async with self._condition:
            await self._condition.wait_for(lambda: self._reserve_if_free(device_id))
        try:
            yield
        finally:
            await self.release(device_id)
