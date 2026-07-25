from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.config import Settings
from app.services.devices_service import CPU_DEVICE_ID
from app.services.resource_probes import ResourceProbe

logger = logging.getLogger(__name__)

_GPU_DEVICE_PREFIX = "dml:"
_NPU_DEVICE_PREFIX = "npu:"


def _device_kind(device_id: str | None) -> str | None:
    if device_id == CPU_DEVICE_ID:
        return "cpu"
    if device_id is not None and device_id.startswith(_GPU_DEVICE_PREFIX):
        return "gpu"
    if device_id is not None and device_id.startswith(_NPU_DEVICE_PREFIX):
        return "npu"
    return None


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

    Subproyecto B (ver docs/superpowers/specs/2026-07-24-gpu-capacity-
    admission-design.md): admission ALSO consults an optional
    `resource_probes` registry (device kind -> ResourceProbe), checking real
    free VRAM/RAM before granting a permit, on top of the job-count check
    above. A probe returning None (or no probe registered for a kind) is
    fail-open -- identical behavior to not having this feature at all. The
    wait loop polls periodically (not just on notify_all()) because a
    resource freed by a process OUTSIDE this app never triggers our own
    release()'s notify.
    """

    def __init__(
        self, settings: Settings, resource_probes: dict[str, ResourceProbe] | None = None
    ) -> None:
        self._settings = settings
        self._in_flight: dict[str | None, int] = {}
        self._condition = asyncio.Condition()
        self._resource_probes = resource_probes or {}
        self._min_free_mb: dict[str, int] = {
            "gpu": settings.min_free_vram_mb,
            "cpu": settings.min_free_ram_mb,
        }

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
        `has_capacity` and calling `reserve`, so the check-and-take is
        atomic against every other acquirer under asyncio's cooperative
        scheduling.
        """
        return self._condition

    @property
    def resource_poll_interval_seconds(self) -> float:
        """How often a waiter should re-check admission when nothing calls
        notify_all() -- needed because resource pressure (VRAM/RAM freed by a
        process outside this app) never triggers our own release(). Shared by
        `acquire()` and DeviceRouter._reserve_least_loaded so both wait loops
        poll at the same cadence."""
        return self._settings.resource_poll_interval_seconds

    def reserve(self, device_id: str | None) -> None:
        """Take one permit unconditionally. Caller MUST hold release_condition
        AND have already confirmed `has_capacity(device_id)` under it."""
        self._in_flight[device_id] = self.in_flight(device_id) + 1

    def _has_enough_resources(self, device_id: str | None) -> bool:
        kind = _device_kind(device_id)
        if kind is None:
            return True
        probe = self._resource_probes.get(kind)
        if probe is None:
            return True
        free_mb = probe.free_capacity_mb(device_id)
        if free_mb is None:
            return True
        threshold = self._min_free_mb.get(kind)
        if threshold is None:
            return True
        return free_mb >= threshold

    def has_capacity(self, device_id: str | None) -> bool:
        """Non-reserving half of the admission predicate: would a reserve for
        `device_id` succeed right now (job-count slot free AND, if a probe
        applies, enough free VRAM/RAM)? Single source of truth shared by
        `_reserve_if_free` (pinned-device acquire) and DeviceRouter's
        auto-route selection, so the two admission paths can never diverge."""
        return self.free_capacity(device_id) > 0 and self._has_enough_resources(device_id)

    def _reserve_if_free(self, device_id: str | None) -> bool:
        if not self.has_capacity(device_id):
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
        # device has a free permit AND (if a resource probe applies) enough
        # free VRAM/RAM, take it, then run the job WITHOUT holding the
        # condition (it's a leaf lock, held only during reserve/release).
        #
        # The wait uses a bounded asyncio.wait_for(...) around each
        # condition.wait() instead of one plain wait_for(predicate) so the
        # predicate gets re-evaluated periodically even when nothing calls
        # notify_all() -- required for resource pressure that clears for
        # reasons outside this app. This relies on asyncio.Condition.wait()
        # re-acquiring its lock even when cancelled by the outer timeout,
        # guaranteed since Python 3.11 (this project's floor).
        async with self._condition:
            logged_wait = False
            while not self._reserve_if_free(device_id):
                if not logged_wait:
                    logger.info("device %s: waiting for capacity (job-count or resource threshold)", device_id)
                    logged_wait = True
                try:
                    await asyncio.wait_for(
                        self._condition.wait(), timeout=self._settings.resource_poll_interval_seconds
                    )
                except TimeoutError:
                    pass
        try:
            yield
        finally:
            await self.release(device_id)
