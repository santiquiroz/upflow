from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from app.services.device_semaphores import DeviceSemaphores
from app.services.devices_service import DeviceInfo
from app.services.model_registry import ModelKind

# ---------------------------------------------------------------------------
# SP7 Task 2 - the optional "auto" device router.
#
# Compatibility (is_device_compatible / compatible_devices / has_compatible_
# device) is pure and synchronous: it only inspects a device kind against a
# model kind, so it's fully testable without asyncio or real GPUs.
#
# pick_least_loaded_device is also pure given a DeviceSemaphores snapshot
# (capacity_for/in_flight are plain reads), so its selection logic is unit
# testable in isolation from the locking below.
#
# DeviceRouter.acquire_auto is the only piece with real concurrency
# concerns: two auto-routed jobs racing to pick a device must never both
# land on the same free device when a different one is idle. See its
# docstring for the no-deadlock argument.
# ---------------------------------------------------------------------------


def is_device_compatible(device: DeviceInfo, model_kind: ModelKind) -> bool:
    """builtin-ncnn requires a Vulkan GPU device; onnx runs on cpu or any GPU."""
    if model_kind == ModelKind.builtin_ncnn:
        return device["kind"] == "gpu"
    return True


def compatible_devices(devices: list[DeviceInfo], model_kind: ModelKind) -> list[DeviceInfo]:
    return [device for device in devices if is_device_compatible(device, model_kind)]


def has_compatible_device(devices: list[DeviceInfo], model_kind: ModelKind) -> bool:
    return bool(compatible_devices(devices, model_kind))


def pick_least_loaded_device(compatible: list[DeviceInfo], device_semaphores: DeviceSemaphores) -> str:
    """Deterministically picks the compatible device with the most free capacity.

    Ties -- including "every candidate is equally busy" -- break on the
    lowest device id, so callers get a stable pick instead of flapping
    between equally-loaded devices. `compatible` must be non-empty.
    """

    def free_capacity(device_id: str) -> int:
        return device_semaphores.capacity_for(device_id) - device_semaphores.in_flight(device_id)

    return min(
        (device["id"] for device in compatible),
        key=lambda device_id: (-free_capacity(device_id), device_id),
    )


class DeviceRouter:
    """Resolves device="auto" jobs to a concrete device_id and reserves it.

    Selection ("which compatible device has the most free capacity") and
    reservation (actually entering that device's DeviceSemaphores permit)
    happen atomically under `_selection_lock`: the lock is held across the
    semaphore's `__aenter__`, so when the picked device has free capacity
    that entry resolves without suspending and the in_flight increment it
    performs is visible to the next picker before the lock is released.
    That's what stops two auto-routed jobs from ever landing on the same
    free device while another compatible one sits idle.

    No deadlock: `_selection_lock` is a plain asyncio.Lock, never a device
    semaphore, and nothing holds a device semaphore while trying to acquire
    it -- there is no cyclic wait. When every compatible device is already
    saturated, the picked device's semaphore entry blocks while still
    holding the selection lock, so other auto-picks queue up behind it
    until a permit frees. That is ordinary backpressure (every semaphore
    permit is always released in a `finally` by the job holding it), not a
    deadlock -- but it does mean concurrent auto-routing decisions serialize
    while the pool is fully saturated. Given ENABLE_AUTO_ROUTE is optional
    and off by default, and the alternative (a per-job reservation ledger)
    meaningfully increases complexity for a case where the system has no
    free capacity anyway, this tradeoff is accepted -- see
    .superpowers/sdd/sp7-task-2-report.md for the self-review.
    """

    def __init__(self, device_semaphores: DeviceSemaphores) -> None:
        self._device_semaphores = device_semaphores
        self._selection_lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire_auto(self, devices: list[DeviceInfo], model_kind: ModelKind) -> AsyncIterator[str]:
        compatible = compatible_devices(devices, model_kind)
        if not compatible:
            raise ValueError(
                f"No compatible device available for model kind {model_kind.value!r} (requested device='auto')"
            )
        async with AsyncExitStack() as stack:
            async with self._selection_lock:
                device_id = pick_least_loaded_device(compatible, self._device_semaphores)
                await stack.enter_async_context(self._device_semaphores.acquire(device_id))
            yield device_id
