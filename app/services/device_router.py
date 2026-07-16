from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

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
# (capacity_for/in_flight are plain reads), so its selection ranking is unit
# testable in isolation from the locking below.
#
# DeviceRouter.acquire_auto holds the concurrency concerns: two auto-routed
# jobs racing to pick a device must never both land on the same free device
# when a different one is idle, AND a picker must never sit on a busy device
# while a DIFFERENT compatible device is free (or frees first). See its
# docstring for the no-deadlock / no-head-of-line-blocking argument.
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


def _numeric_device_sort_key(device_id: str) -> tuple[str, int, str]:
    """Sort key that orders "dml:2" before "dml:10" (numeric, not lexical).

    Splits "<prefix>:<n>" into (prefix, n, "") when the suffix is a plain
    integer; anything else (e.g. "cpu") falls back to (id, -1, id) so it
    still sorts deterministically, just outside the numeric device family.
    """
    prefix, _, suffix = device_id.partition(":")
    if suffix.isdigit():
        return (prefix, int(suffix), "")
    return (device_id, -1, device_id)


def _device_kind_rank(kind: str) -> int:
    """Auto-routing prefers a GPU over the CPU; CPU is a fallback, not a peer.

    Without this, ranking purely by free capacity misroutes onnx "auto" jobs:
    the defaults ship CPU_CONCURRENCY(2) > PER_DEVICE_GPU_CONCURRENCY(1), so an
    idle CPU exposes MORE free slots than an idle GPU and would win -- the
    opposite of the router's goal (dispatch onnx work to a free GPU to save
    time). GPUs rank ahead of everything else; the CPU is only chosen when no
    compatible GPU has free capacity.
    """
    return 0 if kind == "gpu" else 1


def pick_least_loaded_device(compatible: list[DeviceInfo], device_semaphores: DeviceSemaphores) -> str:
    """Deterministically picks the best compatible device for an auto job.

    Ranking, in order: GPU before CPU (see `_device_kind_rank`), then most
    free capacity, then lowest numeric device id. Ties -- including "every
    candidate is equally busy" -- break on the lowest numeric id, so callers
    get a stable pick instead of flapping between equally-loaded devices.
    `compatible` must be non-empty.
    """

    def sort_key(device: DeviceInfo) -> tuple[int, int, tuple[str, int, str]]:
        free_capacity = device_semaphores.free_capacity(device["id"])
        return (_device_kind_rank(device["kind"]), -free_capacity, _numeric_device_sort_key(device["id"]))

    return min(compatible, key=sort_key)["id"]


class DeviceRouter:
    """Resolves device="auto" jobs to a concrete device_id and reserves it.

    `acquire_auto` never blocks while holding selection state. Under the
    shared `DeviceSemaphores.release_condition` it looks for a compatible
    device that has FREE capacity right now and reserves it with a
    non-blocking take (`reserve`, an in_flight increment done under the same
    lock -- instantaneous, so two concurrent auto-picks can't both grab the
    same free device: the first's increment is visible to the second before
    the lock is released). If NO compatible device is currently free it holds
    nothing and `await`s the condition, then re-selects from scratch on the
    next turn.

    Why that matters (the head-of-line bug this avoids): an earlier design
    picked a device eagerly and then blocked on it while holding the
    selection lock. With two saturated GPUs, a picker married to dml:0 would
    keep dml:1 idle for dml:0's entire job runtime when dml:1 freed first,
    and queue every other picker behind it. Here the picker holds no device
    until one is actually free, and whichever compatible device frees first
    is the one taken -- no idle device, no picker blocking another picker.

    No deadlock: the only lock is the leaf `release_condition`, held just to
    check-and-reserve or to `wait()` (which releases it while suspended). No
    device is ever held while waiting for another. No busy-wait: waiting is a
    real `condition.wait()`, woken by a permit release, not a spin. The
    reserved permit is always released in `acquire_auto`'s `finally` (and if
    cancelled while waiting, nothing was reserved, so nothing leaks).
    """

    def __init__(self, device_semaphores: DeviceSemaphores) -> None:
        self._device_semaphores = device_semaphores

    @asynccontextmanager
    async def acquire_auto(self, devices: list[DeviceInfo], model_kind: ModelKind) -> AsyncIterator[str]:
        compatible = compatible_devices(devices, model_kind)
        if not compatible:
            raise ValueError(
                f"No compatible device available for model kind {model_kind.value!r} (requested device='auto')"
            )
        device_id = await self._reserve_least_loaded(compatible)
        try:
            yield device_id
        finally:
            await self._device_semaphores.release(device_id)

    async def _reserve_least_loaded(self, compatible: list[DeviceInfo]) -> str:
        semaphores = self._device_semaphores
        condition = semaphores.release_condition
        async with condition:
            while True:
                free = [device for device in compatible if semaphores.free_capacity(device["id"]) > 0]
                if free:
                    device_id = pick_least_loaded_device(free, semaphores)
                    semaphores.reserve(device_id)
                    return device_id
                # Every compatible device is saturated: release the lock and
                # sleep until SOME permit frees, then re-select (the freed
                # device might be any compatible one, not one we pre-chose).
                await condition.wait()
