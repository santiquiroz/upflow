from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.services.device_semaphores import DeviceSemaphores

# ---------------------------------------------------------------------------
# SP7 Task 1 - DeviceSemaphores: a per-device_id registry of asyncio.Semaphore
# objects, created lazily on first use. This file unit-tests the registry in
# isolation (capacity per device kind, in_flight accounting, lazy
# get-or-create race-safety). Cross-device parallelism / same-device
# serialization through the real JobManager/VideoJobManager workers is
# covered end-to-end in tests/test_multigpu_concurrency.py.
# ---------------------------------------------------------------------------

HOLD_SECONDS = 0.08


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {"RUNTIME_DIR": str(tmp_path)}
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


def intervals_overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> bool:
    return start_a < end_b and start_b < end_a


async def acquire_and_record(
    semaphores: DeviceSemaphores, device_id: str, intervals: list[tuple[float, float]]
) -> None:
    async with semaphores.acquire(device_id):
        start = time.monotonic()
        await asyncio.sleep(HOLD_SECONDS)
        intervals.append((start, time.monotonic()))


async def test_same_device_capacity_one_serializes_two_acquires(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1)
    semaphores = DeviceSemaphores(settings)
    intervals: list[tuple[float, float]] = []

    await asyncio.gather(
        acquire_and_record(semaphores, "dml:0", intervals),
        acquire_and_record(semaphores, "dml:0", intervals),
    )

    assert len(intervals) == 2
    (start_a, end_a), (start_b, end_b) = intervals
    assert not intervals_overlap(start_a, end_a, start_b, end_b), (
        "two acquires on the same device with per_device capacity=1 overlapped"
    )


async def test_cpu_device_gets_cpu_concurrency_capacity_independent_of_gpu(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1, CPU_CONCURRENCY=2)
    semaphores = DeviceSemaphores(settings)

    async def hold(device_id: str, hold_seconds: float) -> None:
        async with semaphores.acquire(device_id):
            await asyncio.sleep(hold_seconds)

    # Two cpu jobs at once must both fit under CPU_CONCURRENCY=2, even though
    # PER_DEVICE_GPU_CONCURRENCY=1 would only allow one GPU job at a time.
    await asyncio.wait_for(
        asyncio.gather(hold("cpu", 0.05), hold("cpu", 0.05)),
        timeout=1.0,
    )
    assert semaphores.in_flight("cpu") == 0


async def test_distinct_devices_run_concurrently_without_blocking_each_other(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1)
    semaphores = DeviceSemaphores(settings)
    intervals: dict[str, tuple[float, float]] = {}

    async def hold(device_id: str) -> None:
        async with semaphores.acquire(device_id):
            start = time.monotonic()
            await asyncio.sleep(HOLD_SECONDS)
            intervals[device_id] = (start, time.monotonic())

    await asyncio.gather(hold("dml:0"), hold("dml:1"))

    start_a, end_a = intervals["dml:0"]
    start_b, end_b = intervals["dml:1"]
    assert intervals_overlap(start_a, end_a, start_b, end_b), (
        "dml:0 and dml:1 never overlapped despite being distinct devices"
    )


async def test_in_flight_reports_zero_before_any_acquire(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    semaphores = DeviceSemaphores(settings)

    assert semaphores.in_flight("dml:0") == 0
    assert semaphores.in_flight("cpu") == 0


async def test_in_flight_tracks_held_permits_and_resets_after_release(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=2)
    semaphores = DeviceSemaphores(settings)

    async with semaphores.acquire("dml:0"):
        assert semaphores.in_flight("dml:0") == 1
        async with semaphores.acquire("dml:0"):
            assert semaphores.in_flight("dml:0") == 2

    assert semaphores.in_flight("dml:0") == 0


async def test_concurrent_first_acquires_for_new_device_never_create_two_semaphores(tmp_path: Path) -> None:
    """Regression guard for the lazy get-or-create race: many coroutines
    hitting a brand-new device_id at once must all end up sharing exactly one
    semaphore, not silently doubling the effective capacity for that device.
    """
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=3)
    semaphores = DeviceSemaphores(settings)
    max_observed_in_flight = 0

    async def acquire_and_observe() -> None:
        nonlocal max_observed_in_flight
        async with semaphores.acquire("dml:0"):
            await asyncio.sleep(0.02)
            max_observed_in_flight = max(max_observed_in_flight, semaphores.in_flight("dml:0"))

    await asyncio.gather(*(acquire_and_observe() for _ in range(10)))

    assert max_observed_in_flight == 3, (
        f"expected the shared capacity of 3 to be the ceiling, observed {max_observed_in_flight}"
    )


@pytest.mark.parametrize(
    "field_alias", ["PER_DEVICE_GPU_CONCURRENCY", "CPU_CONCURRENCY", "MAX_CONCURRENT_JOBS"]
)
def test_concurrency_settings_reject_values_below_one(tmp_path: Path, field_alias: str) -> None:
    with pytest.raises(ValidationError):
        make_settings(tmp_path, **{field_alias: 0})


async def test_cpu_device_id_constant_matches_devices_service(tmp_path: Path) -> None:
    from app.services.devices_service import CPU_DEVICE_ID

    settings = make_settings(tmp_path, CPU_CONCURRENCY=5, PER_DEVICE_GPU_CONCURRENCY=1)
    semaphores = DeviceSemaphores(settings)

    async with semaphores.acquire(CPU_DEVICE_ID):
        async with semaphores.acquire(CPU_DEVICE_ID):
            async with semaphores.acquire(CPU_DEVICE_ID):
                assert semaphores.in_flight(CPU_DEVICE_ID) == 3, "cpu concurrency must use CPU_CONCURRENCY, not GPU"
