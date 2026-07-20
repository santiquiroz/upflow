from __future__ import annotations

import threading
from typing import Protocol


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
        self._lock = threading.Lock()

    def acquire(self, device: str, owner: GpuSessionOwner) -> None:
        with self._lock:
            previous = self._owners.get(device)
            if previous is not None and previous is not owner:
                previous.release_device(device)
            self._owners[device] = owner
