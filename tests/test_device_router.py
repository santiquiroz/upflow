from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from app.config import Settings
from app.services.device_router import (
    DeviceRouter,
    _numeric_device_sort_key,
    compatible_devices,
    has_compatible_device,
    is_device_compatible,
    pick_least_loaded_device,
)
from app.services.device_semaphores import DeviceSemaphores
from app.services.devices_service import DeviceInfo
from app.services.model_registry import ModelKind

# ---------------------------------------------------------------------------
# SP7 Task 2 - the optional "auto" device router.
#
# Pure selection logic (compatibility + least-loaded pick) is unit tested in
# isolation from asyncio locking here. The atomic acquire_auto race behavior
# (two concurrent auto jobs never landing on the same free device) is
# exercised end-to-end through JobManager/VideoJobManager in
# tests/test_auto_route.py, using the same fake-engine pattern as SP7 T1's
# tests/test_multigpu_concurrency.py.
# ---------------------------------------------------------------------------

CPU: DeviceInfo = {"id": "cpu", "kind": "cpu", "name": "CPU", "backend": "cpu"}
GPU0: DeviceInfo = {"id": "dml:0", "kind": "gpu", "name": "GPU 0", "backend": "directml"}
GPU1: DeviceInfo = {"id": "dml:1", "kind": "gpu", "name": "GPU 1", "backend": "directml"}


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {"RUNTIME_DIR": str(tmp_path)}
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


# ---------------------------------------------------------------------------
# is_device_compatible / compatible_devices / has_compatible_device
# ---------------------------------------------------------------------------


def test_ncnn_model_is_compatible_only_with_gpu_devices() -> None:
    assert is_device_compatible(GPU0, ModelKind.builtin_ncnn) is True
    assert is_device_compatible(CPU, ModelKind.builtin_ncnn) is False


def test_onnx_model_is_compatible_with_cpu_and_gpu() -> None:
    assert is_device_compatible(CPU, ModelKind.onnx) is True
    assert is_device_compatible(GPU0, ModelKind.onnx) is True


def test_compatible_devices_filters_cpu_out_for_ncnn() -> None:
    result = compatible_devices([CPU, GPU0, GPU1], ModelKind.builtin_ncnn)

    assert result == [GPU0, GPU1]


def test_compatible_devices_keeps_everything_for_onnx() -> None:
    result = compatible_devices([CPU, GPU0, GPU1], ModelKind.onnx)

    assert result == [CPU, GPU0, GPU1]


def test_has_compatible_device_false_when_ncnn_and_only_cpu_present() -> None:
    assert has_compatible_device([CPU], ModelKind.builtin_ncnn) is False


def test_has_compatible_device_true_when_onnx_and_only_cpu_present() -> None:
    assert has_compatible_device([CPU], ModelKind.onnx) is True


# ---------------------------------------------------------------------------
# pick_least_loaded_device
# ---------------------------------------------------------------------------


def test_pick_least_loaded_device_prefers_fully_free_device(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1)
    semaphores = DeviceSemaphores(settings)
    # Force dml:0's semaphore to exist with in_flight already at capacity by
    # driving a real acquire and inspecting mid-hold, avoiding private state.

    async def scenario() -> str:
        async with semaphores.acquire("dml:0"):
            return pick_least_loaded_device([GPU0, GPU1], semaphores)

    picked = asyncio.run(scenario())

    assert picked == "dml:1", "dml:0 is busy (in_flight=1/1); dml:1 is fully free"


def test_pick_least_loaded_device_breaks_ties_on_lowest_device_id(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1)
    semaphores = DeviceSemaphores(settings)

    picked = pick_least_loaded_device([GPU1, GPU0], semaphores)

    assert picked == "dml:0", "both fully free -- deterministic tie-break must pick the lowest id"


def test_pick_least_loaded_device_prefers_gpu_over_cpu_despite_more_cpu_slots(tmp_path: Path) -> None:
    # CPU_CONCURRENCY(2) > PER_DEVICE_GPU_CONCURRENCY(1): the idle CPU exposes
    # MORE free slots than the idle GPU, but auto-routing must still pick the
    # GPU (the fast path) -- otherwise onnx auto jobs pile onto the CPU.
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1, CPU_CONCURRENCY=2)
    semaphores = DeviceSemaphores(settings)

    picked = pick_least_loaded_device([CPU, GPU0], semaphores)

    assert picked == "dml:0", "both idle, but a free GPU must win over a free CPU"


def test_pick_least_loaded_device_single_candidate_is_trivially_picked(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    semaphores = DeviceSemaphores(settings)

    assert pick_least_loaded_device([GPU0], semaphores) == "dml:0"


def test_pick_least_loaded_device_tie_break_is_numeric_not_lexical(tmp_path: Path) -> None:
    # dml:2 must win over dml:10 on a tie -- a lexical string sort would put
    # "dml:10" before "dml:2" and route to the wrong device.
    settings = make_settings(tmp_path)
    semaphores = DeviceSemaphores(settings)
    gpu2: DeviceInfo = {"id": "dml:2", "kind": "gpu", "name": "GPU 2", "backend": "directml"}
    gpu10: DeviceInfo = {"id": "dml:10", "kind": "gpu", "name": "GPU 10", "backend": "directml"}

    assert pick_least_loaded_device([gpu10, gpu2], semaphores) == "dml:2"


def test_numeric_device_sort_key_orders_dml_devices_numerically() -> None:
    assert _numeric_device_sort_key("dml:2") < _numeric_device_sort_key("dml:10")


# ---------------------------------------------------------------------------
# DeviceRouter.acquire_auto
# ---------------------------------------------------------------------------


async def test_acquire_auto_raises_value_error_when_no_compatible_device(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    router = DeviceRouter(DeviceSemaphores(settings))

    with pytest.raises(ValueError, match="No compatible device"):
        async with router.acquire_auto([CPU], ModelKind.builtin_ncnn):
            pass  # pragma: no cover - must never enter


async def test_acquire_auto_yields_the_picked_device_id_and_holds_its_semaphore(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1)
    semaphores = DeviceSemaphores(settings)
    router = DeviceRouter(semaphores)

    async with router.acquire_auto([GPU0], ModelKind.builtin_ncnn) as device_id:
        assert device_id == "dml:0"
        assert semaphores.in_flight("dml:0") == 1

    assert semaphores.in_flight("dml:0") == 0, "the semaphore permit must be released on exit"


async def test_acquire_auto_never_routes_ncnn_to_cpu_even_when_cpu_is_idle(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    semaphores = DeviceSemaphores(settings)
    router = DeviceRouter(semaphores)

    async with router.acquire_auto([CPU, GPU0], ModelKind.builtin_ncnn) as device_id:
        assert device_id == "dml:0"


async def test_two_concurrent_auto_picks_distribute_across_two_free_gpus(tmp_path: Path) -> None:
    """The atomicity guarantee: two auto-routed jobs racing to pick a device
    with two fully-idle GPUs must land on DIFFERENT devices, not both on
    dml:0 -- regression guard for the selection-lock design in DeviceRouter.
    """
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1)
    semaphores = DeviceSemaphores(settings)
    router = DeviceRouter(semaphores)
    picked: list[str] = []

    async def pick_and_hold() -> None:
        async with router.acquire_auto([GPU0, GPU1], ModelKind.builtin_ncnn) as device_id:
            picked.append(device_id)
            await asyncio.sleep(0.05)

    await asyncio.gather(pick_and_hold(), pick_and_hold())

    assert sorted(picked) == ["dml:0", "dml:1"], f"expected one job per device, got {picked}"


async def test_auto_pick_blocks_until_a_busy_compatible_device_frees_instead_of_deadlocking(
    tmp_path: Path,
) -> None:
    """Only one compatible device exists and it's saturated: acquire_auto
    must block (not raise, not deadlock) until the holder releases."""
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1)
    semaphores = DeviceSemaphores(settings)
    router = DeviceRouter(semaphores)
    events: list[str] = []

    async def hold_first() -> None:
        async with semaphores.acquire("dml:0"):
            events.append("first-acquired")
            await asyncio.sleep(0.05)
            events.append("first-released")

    async def auto_pick_second() -> None:
        await asyncio.sleep(0.01)  # let hold_first grab the permit first
        async with router.acquire_auto([GPU0], ModelKind.builtin_ncnn) as device_id:
            events.append(f"auto-acquired-{device_id}")

    await asyncio.wait_for(asyncio.gather(hold_first(), auto_pick_second()), timeout=1.0)

    assert events == ["first-acquired", "first-released", "auto-acquired-dml:0"], (
        "auto pick must wait for the busy device to free, not deadlock or skip ahead"
    )


async def test_auto_pick_takes_the_device_that_frees_first_not_a_stale_pick(tmp_path: Path) -> None:
    """The Critical this fix targets: both GPUs saturated, and the device the
    router would NOT tie-break to (dml:1) frees FIRST. A waiting auto pick
    must take dml:1 the moment it frees -- NOT stay parked on dml:0 (its
    tie-break favourite) and leave dml:1 idle for the rest of dml:0's job.
    """
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1)
    semaphores = DeviceSemaphores(settings)
    router = DeviceRouter(semaphores)
    dml0_released_at = 0.0
    auto_acquired_at = 0.0
    auto_device = ""

    async def hold_dml0_long() -> None:
        nonlocal dml0_released_at
        async with semaphores.acquire("dml:0"):
            await asyncio.sleep(0.30)
        dml0_released_at = time.monotonic()

    async def hold_dml1_short() -> None:
        async with semaphores.acquire("dml:1"):
            await asyncio.sleep(0.06)

    async def auto_pick() -> None:
        nonlocal auto_acquired_at, auto_device
        await asyncio.sleep(0.02)  # ensure both holds are active first
        async with router.acquire_auto([GPU0, GPU1], ModelKind.builtin_ncnn) as device_id:
            auto_acquired_at = time.monotonic()
            auto_device = device_id

    await asyncio.wait_for(
        asyncio.gather(hold_dml0_long(), hold_dml1_short(), auto_pick()), timeout=2.0
    )

    assert auto_device == "dml:1", "auto must take the device that freed first (dml:1), not a stale dml:0 pick"
    assert auto_acquired_at < dml0_released_at, "auto must not wait for the busy dml:0 to free"


# ---------------------------------------------------------------------------
# DeviceRouter.acquire_auto -- resource-gated admission (subproyecto B fix
# wave: the auto-router used to filter on raw free_capacity() only, so it
# could route to a device with a free job slot but not enough VRAM/RAM, and
# its wait loop never re-checked resource state that clears externally).
# ---------------------------------------------------------------------------


class DeviceKeyedProbe:
    """Returns a fixed value per device_id -- lets a test make ONE compatible
    device resource-starved while a DIFFERENT compatible device has plenty.
    A single constant probe (as in test_device_semaphores.py) can't express
    that, since it starves every device of that kind identically."""

    def __init__(self, values: dict[str, int | None]) -> None:
        self._values = values

    def free_capacity_mb(self, device_id: str) -> int | None:
        return self._values.get(device_id)


class SequenceProbe:
    """Returns each value in `values` in order, repeating the last one once
    exhausted -- simulates external VRAM pressure resolving over time
    without any job of ours ever calling release(). Mirrors the identically
    named helper in tests/test_device_semaphores.py; `calls` is public so
    tests can assert the poll loop actually re-checked more than once."""

    def __init__(self, values: list[int | None]) -> None:
        self._values = values
        self.calls = 0

    def free_capacity_mb(self, device_id: str) -> int | None:
        value = self._values[min(self.calls, len(self._values) - 1)]
        self.calls += 1
        return value


async def test_auto_pick_skips_a_device_with_job_slots_but_insufficient_vram(tmp_path: Path) -> None:
    """dml:0 has a free job slot but not enough VRAM; dml:1 has both. The old
    free_capacity()-only filter would consider dml:0 "free" and the
    lowest-id tie-break would route there anyway -- this is the regression
    guard for gating the filter on has_capacity() instead.
    """
    settings = make_settings(tmp_path, PER_DEVICE_GPU_CONCURRENCY=1, MIN_FREE_VRAM_MB=1024)
    probe = DeviceKeyedProbe({"dml:0": 256, "dml:1": 2048})
    semaphores = DeviceSemaphores(settings, resource_probes={"gpu": probe})
    router = DeviceRouter(semaphores)

    async def auto_pick() -> str:
        async with router.acquire_auto([GPU0, GPU1], ModelKind.builtin_ncnn) as device_id:
            return device_id

    picked = await asyncio.wait_for(auto_pick(), timeout=0.5)

    assert picked == "dml:1", "dml:0 has a free job slot but insufficient VRAM; must route to dml:1 instead"


async def test_auto_pick_waits_when_the_only_compatible_device_is_resource_starved(
    tmp_path: Path,
) -> None:
    """No alternate device exists: the auto-router must block (not admit,
    not raise) while the sole compatible device stays under the VRAM floor."""
    settings = make_settings(
        tmp_path, PER_DEVICE_GPU_CONCURRENCY=1, MIN_FREE_VRAM_MB=1024,
        RESOURCE_POLL_INTERVAL_SECONDS=0.05,
    )
    semaphores = DeviceSemaphores(settings, resource_probes={"gpu": DeviceKeyedProbe({"dml:0": 256})})
    router = DeviceRouter(semaphores)

    async def auto_pick() -> None:
        async with router.acquire_auto([GPU0], ModelKind.builtin_ncnn):
            pass  # pragma: no cover - must never be admitted within the timeout

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(auto_pick(), timeout=0.3)


async def test_auto_pick_admits_once_external_vram_pressure_clears_via_poll(tmp_path: Path) -> None:
    """The periodic-poll path this fix adds to _reserve_least_loaded: dml:0
    keeps a free job slot the whole time but starts under the VRAM floor and
    only clears it a few probe calls later. No job of ours ever calls
    release() here, so admission can only come from the bounded
    condition.wait() timeout re-checking has_capacity() -- never from
    notify_all(), which this test never triggers.
    """
    settings = make_settings(
        tmp_path, PER_DEVICE_GPU_CONCURRENCY=1, MIN_FREE_VRAM_MB=1024,
        RESOURCE_POLL_INTERVAL_SECONDS=0.05,
    )
    probe = SequenceProbe([256, 256, 256, 2048])
    semaphores = DeviceSemaphores(settings, resource_probes={"gpu": probe})
    router = DeviceRouter(semaphores)

    async def acquire_and_check() -> None:
        async with router.acquire_auto([GPU0], ModelKind.builtin_ncnn) as device_id:
            assert device_id == "dml:0"
            assert semaphores.in_flight("dml:0") == 1

    await asyncio.wait_for(acquire_and_check(), timeout=1.0)

    assert probe.calls >= 4, "admission must come from repeated polling, not a single lucky check"
