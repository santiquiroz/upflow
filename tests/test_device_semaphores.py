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


@pytest.mark.parametrize("field_alias", ["MIN_FREE_VRAM_MB", "MIN_FREE_RAM_MB"])
def test_min_free_mb_settings_reject_negative_values(tmp_path: Path, field_alias: str) -> None:
    with pytest.raises(ValidationError):
        make_settings(tmp_path, **{field_alias: -1})


def test_min_free_mb_settings_accept_zero_as_no_floor(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, MIN_FREE_VRAM_MB=0, MIN_FREE_RAM_MB=0)

    assert settings.min_free_vram_mb == 0
    assert settings.min_free_ram_mb == 0


def test_min_free_mb_settings_have_documented_defaults(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    assert settings.min_free_vram_mb == 768
    assert settings.min_free_ram_mb == 1024


def test_resource_poll_interval_rejects_non_positive_values(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        make_settings(tmp_path, RESOURCE_POLL_INTERVAL_SECONDS=0)


def test_resource_poll_interval_default(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    assert settings.resource_poll_interval_seconds == 5.0


class ConstantProbe:
    def __init__(self, value: int | None) -> None:
        self._value = value

    def free_capacity_mb(self, device_id: str) -> int | None:
        return self._value


class SequenceProbe:
    """Returns each value in `values` in order, repeating the last one once
    exhausted -- simulates external pressure resolving over time without any
    job of ours ever calling release()."""

    def __init__(self, values: list[int | None]) -> None:
        self._values = values
        self._calls = 0

    def free_capacity_mb(self, device_id: str) -> int | None:
        value = self._values[min(self._calls, len(self._values) - 1)]
        self._calls += 1
        return value


async def test_acquire_waits_when_capacity_ok_but_vram_below_threshold(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path, PER_DEVICE_GPU_CONCURRENCY=2, MIN_FREE_VRAM_MB=1024,
        RESOURCE_POLL_INTERVAL_SECONDS=0.05,
    )
    semaphores = DeviceSemaphores(settings, resource_probes={"gpu": ConstantProbe(512)})

    async def try_acquire() -> None:
        async with semaphores.acquire("dml:0"):
            pass

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(try_acquire(), timeout=0.3)


async def test_acquire_admits_once_external_vram_pressure_clears(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path, PER_DEVICE_GPU_CONCURRENCY=2, MIN_FREE_VRAM_MB=1024,
        RESOURCE_POLL_INTERVAL_SECONDS=0.05,
    )
    probe = SequenceProbe([256, 256, 256, 2048])
    semaphores = DeviceSemaphores(settings, resource_probes={"gpu": probe})

    async def acquire_and_check() -> None:
        async with semaphores.acquire("dml:0"):
            assert semaphores.in_flight("dml:0") == 1

    await asyncio.wait_for(acquire_and_check(), timeout=1.0)


async def test_acquire_ignores_resource_probes_for_devices_outside_their_kind(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1, MIN_FREE_VRAM_MB=999_999)
    semaphores = DeviceSemaphores(settings, resource_probes={"gpu": ConstantProbe(0)})

    async def acquire_and_check() -> None:
        async with semaphores.acquire("cpu"):
            assert semaphores.in_flight("cpu") == 1

    await asyncio.wait_for(acquire_and_check(), timeout=0.5)


async def test_acquire_behavior_unchanged_when_no_resource_probes_injected(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1)
    semaphores = DeviceSemaphores(settings)

    async def acquire_and_check() -> None:
        async with semaphores.acquire("dml:0"):
            assert semaphores.in_flight("dml:0") == 1

    await asyncio.wait_for(acquire_and_check(), timeout=0.5)


async def test_acquire_treats_probe_none_as_fail_open(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1, MIN_FREE_VRAM_MB=999_999)
    semaphores = DeviceSemaphores(settings, resource_probes={"gpu": ConstantProbe(None)})

    async def acquire_and_check() -> None:
        async with semaphores.acquire("dml:0"):
            assert semaphores.in_flight("dml:0") == 1

    await asyncio.wait_for(acquire_and_check(), timeout=0.5)


# ---------------------------------------------------------------------------
# Deadlock por cache propio (bug real 2026-07-31: jobs de generacion quedaban
# `queued` para siempre con el GPU ocioso, porque las sesiones ONNX cacheadas
# de NUESTRO propio proceso dejaban el "libre" bajo el umbral y nada lo iba a
# liberar jamas. El proximo job habria REUSADO ese cache sin alocar nada.)
# ---------------------------------------------------------------------------


class CacheHeavyProbe:
    """Libre bajo el umbral, pero NUESTRO proceso retiene de sobra: la firma
    del cache de sesiones ocioso (Budget-CurrentUsage chico, CurrentUsage
    grande). Un juego externo es lo contrario: libre chico y uso propio ~0."""

    def __init__(self, free_mb: int, own_mb: int | None) -> None:
        self._free = free_mb
        self._own = own_mb

    def free_capacity_mb(self, device_id: str) -> int | None:
        return self._free

    def own_usage_mb(self, device_id: str) -> int | None:
        return self._own


async def test_an_idle_device_admits_when_the_shortfall_is_our_own_cache(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1, MIN_FREE_VRAM_MB=1024)
    semaphores = DeviceSemaphores(
        settings, resource_probes={"gpu": CacheHeavyProbe(free_mb=100, own_mb=8000)}
    )

    async def acquire_once() -> None:
        async with semaphores.acquire("dml:0"):
            pass

    # Sin el bypass esto espera para siempre; el timeout lo vuelve un fallo claro.
    await asyncio.wait_for(acquire_once(), timeout=1.0)


async def test_an_idle_device_still_waits_under_external_pressure(tmp_path: Path) -> None:
    # Uso propio ~0 y libre bajo = un proceso EXTERNO come la VRAM (el caso que
    # el subproyecto B queria: esperar a que el juego la suelte).
    settings = make_settings(
        tmp_path, PER_DEVICE_GPU_CONCURRENCY=1, MIN_FREE_VRAM_MB=1024,
        RESOURCE_POLL_INTERVAL_SECONDS=0.05,
    )
    semaphores = DeviceSemaphores(
        settings, resource_probes={"gpu": CacheHeavyProbe(free_mb=100, own_mb=0)}
    )

    async def acquire_once() -> None:
        async with semaphores.acquire("dml:0"):
            pass

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(acquire_once(), timeout=0.3)


async def test_a_busy_device_waits_even_if_our_cache_covers_the_shortfall(tmp_path: Path) -> None:
    # Con un job corriendo el gate conserva todo su valor: no apilar un segundo
    # job sobre un device bajo presion.
    settings = make_settings(
        tmp_path, PER_DEVICE_GPU_CONCURRENCY=2, MIN_FREE_VRAM_MB=1024,
        RESOURCE_POLL_INTERVAL_SECONDS=0.05,
    )
    semaphores = DeviceSemaphores(
        settings, resource_probes={"gpu": CacheHeavyProbe(free_mb=100, own_mb=8000)}
    )

    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold_first() -> None:
        async with semaphores.acquire("dml:0"):
            entered.set()
            await release.wait()

    holder = asyncio.create_task(hold_first())
    await entered.wait()

    async def acquire_second() -> None:
        async with semaphores.acquire("dml:0"):
            pass

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(acquire_second(), timeout=0.3)
    release.set()
    await holder


async def test_probes_without_own_usage_keep_the_old_behavior(tmp_path: Path) -> None:
    # Un probe viejo (sin own_usage_mb) no habilita el bypass: fail-safe.
    settings = make_settings(
        tmp_path, PER_DEVICE_GPU_CONCURRENCY=1, MIN_FREE_VRAM_MB=1024,
        RESOURCE_POLL_INTERVAL_SECONDS=0.05,
    )
    semaphores = DeviceSemaphores(settings, resource_probes={"gpu": ConstantProbe(100)})

    async def acquire_once() -> None:
        async with semaphores.acquire("dml:0"):
            pass

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(acquire_once(), timeout=0.3)
