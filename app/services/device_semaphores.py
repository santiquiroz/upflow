from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.config import Settings
from app.services.devices_service import CPU_DEVICE_ID


class DeviceSemaphores:
    """Per-device concurrency gate.

    Each device_id (e.g. "cpu", "dml:0", "dml:1") gets its own
    asyncio.Semaphore, created lazily on first use. A job that acquires the
    semaphore for its own device never blocks on jobs running on a different
    device -- only jobs sharing the same device_id serialize against each
    other, up to that device's configured capacity.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._semaphores: dict[str | None, asyncio.Semaphore] = {}
        self._in_flight: dict[str | None, int] = {}
        self._create_lock = asyncio.Lock()

    async def _get_or_create(self, device_id: str | None) -> asyncio.Semaphore:
        semaphore = self._semaphores.get(device_id)
        if semaphore is not None:
            return semaphore
        # Double-checked locking: the lock only guards the create-and-store
        # step, so concurrent first-time acquires for the same device_id
        # never race into creating two independent semaphores (which would
        # silently double the effective capacity for that device).
        async with self._create_lock:
            semaphore = self._semaphores.get(device_id)
            if semaphore is None:
                semaphore = asyncio.Semaphore(self._capacity_for(device_id))
                self._semaphores[device_id] = semaphore
                self._in_flight[device_id] = 0
            return semaphore

    def _capacity_for(self, device_id: str | None) -> int:
        if device_id == CPU_DEVICE_ID:
            return self._settings.cpu_concurrency
        return self._settings.per_device_gpu_concurrency

    @asynccontextmanager
    async def acquire(self, device_id: str | None) -> AsyncIterator[None]:
        semaphore = await self._get_or_create(device_id)
        async with semaphore:
            # No `await` between the increment/decrement and their matching
            # read in in_flight(), so this stays race-free under asyncio's
            # single-threaded cooperative scheduling without a lock.
            self._in_flight[device_id] += 1
            try:
                yield
            finally:
                self._in_flight[device_id] -= 1

    def in_flight(self, device_id: str | None) -> int:
        return self._in_flight.get(device_id, 0)
