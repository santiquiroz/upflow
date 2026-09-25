from __future__ import annotations

import logging
import threading
from typing import Protocol

logger = logging.getLogger(__name__)


class GpuSessionOwner(Protocol):
    def release_device(self, device: str) -> None: ...


class GpuSessionCoordinator:
    """Exclusion mutua por device entre motores con cache de sesiones ONNX.

    Cuando un motor distinto pide el mismo device, el dueño anterior libera
    SOLO su entrada para ese device (release_device) -- no afecta su cache
    para otros devices, y no afecta a otros motores en devices distintos.
    """

    def __init__(self) -> None:
        self._owners: dict[str, GpuSessionOwner] = {}
        self._known: dict[int, GpuSessionOwner] = {}
        self._lock = threading.Lock()

    def register(self, owner: GpuSessionOwner) -> None:
        with self._lock:
            self._remember(owner)

    def acquire(self, device: str, owner: GpuSessionOwner) -> None:
        with self._lock:
            self._remember(owner)
            previous = self._owners.get(device)
            if previous is not None and previous is not owner:
                previous.release_device(device)
            self._owners[device] = owner

    def invalidate_device(self, device: str) -> None:
        # D3D12 comparte un device por adaptador y proceso: una remoción mata
        # las sesiones de todos los motores, no solo las del dueño actual.
        with self._lock:
            self._owners.pop(device, None)
            for owner in list(self._known.values()):
                _release_quietly(owner, device)

    def _remember(self, owner: GpuSessionOwner) -> None:
        self._known.setdefault(id(owner), owner)


def _release_quietly(owner: GpuSessionOwner, device: str) -> None:
    try:
        owner.release_device(device)
    except Exception as exc:  # noqa: BLE001 -- una sesión muerta no frena la difusión a los demás dueños
        logger.warning(
            "gpu_session_coordinator: %s no pudo liberar %s tras invalidarlo: %s",
            type(owner).__name__,
            device,
            exc,
        )
