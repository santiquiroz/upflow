from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.config import Settings
from app.services.device_router import (
    DeviceRouter,
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


def test_pick_least_loaded_device_single_candidate_is_trivially_picked(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    semaphores = DeviceSemaphores(settings)

    assert pick_least_loaded_device([GPU0], semaphores) == "dml:0"


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
