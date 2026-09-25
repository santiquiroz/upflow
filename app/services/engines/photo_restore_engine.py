from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings
from app.services import ep_registry
from app.services.devices_service import adapter_free_vram_mb
from app.services.dml_device import try_parse_dml_device_id
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.missing_pack import missing_pack_message
from app.services.restore_models import RESTORE_MODELS, RestoreModelSpec, model_path

SessionFactory = Callable[..., Any]
FreeVramProbe = Callable[[str], int | None]

_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class SessionKey:
    model_id: str
    device: str
    precision: str


@dataclass(frozen=True, slots=True)
class _LiveSession:
    session: Any
    estimated_mb: float


def estimated_vram_mb(model_file: Path, vram_factor: float) -> float:
    return model_file.stat().st_size * vram_factor / _BYTES_PER_MB


def dml_free_vram_mb(device: str) -> int | None:
    index = try_parse_dml_device_id(device)
    return None if index is None else adapter_free_vram_mb(index)


def is_gpu_device(device: str) -> bool:
    return try_parse_dml_device_id(device) is not None


def configure_session_options(
    options: Any, spec: RestoreModelSpec, device: str, *, disable_all_level: Any
) -> Any:
    if spec.ort_disable_all:
        options.graph_optimization_level = disable_all_level
    if is_gpu_device(device):
        # DML maneja su propio allocator: el patron de memoria de ORT no sirve ahi (como GMFSS).
        options.enable_mem_pattern = False
        options.intra_op_num_threads = 1
    return options


def build_session_options(spec: RestoreModelSpec, device: str) -> Any:
    import onnxruntime as ort

    return configure_session_options(
        ort.SessionOptions(),
        spec,
        device,
        disable_all_level=ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    )


class PhotoRestoreEngine:
    def __init__(
        self,
        settings: Settings,
        gpu_coordinator: GpuSessionCoordinator,
        *,
        models: Mapping[str, RestoreModelSpec] = RESTORE_MODELS,
        create_session: SessionFactory = ep_registry.create_session,
        free_vram_mb: FreeVramProbe = dml_free_vram_mb,
    ) -> None:
        self.settings = settings
        self.gpu_coordinator = gpu_coordinator
        self._models = models
        self._create_session = create_session
        self._free_vram_mb = free_vram_mb
        self._sessions: OrderedDict[SessionKey, _LiveSession] = OrderedDict()
        self._phase_devices: set[str] = set()
        # Reentrante: la sesion se crea con el candado tomado para que desalojo y alta sean atomicos.
        self._lock = threading.RLock()
        gpu_coordinator.register(self)

    def begin_phase(self, device: str) -> None:
        self.gpu_coordinator.acquire(device, self)
        with self._lock:
            self._phase_devices.add(device)

    def session(self, model_id: str, device: str, precision: str) -> Any:
        spec = self._models[model_id]
        key = SessionKey(model_id, device, precision)
        with self._lock:
            self._require_phase(device)
            return self._cached_or_created(spec, key)

    def release_device(self, device: str) -> None:
        with self._lock:
            self._phase_devices.discard(device)
            for key in self._keys_on(device):
                del self._sessions[key]

    def release_before_ncnn(self, device: str) -> bool:
        # ncnn corre en Vulkan, fuera del coordinator: sin VRAM libre medible, liberar es lo seguro.
        if not is_gpu_device(device):
            return False
        free_mb = self._free_vram_mb(device)
        if free_mb is not None and free_mb >= self.settings.restore_ncnn_headroom_mb:
            return False
        self.release_device(device)
        return True

    def live_sessions(self, device: str) -> tuple[SessionKey, ...]:
        with self._lock:
            return tuple(self._keys_on(device))

    def estimated_vram_mb(self, device: str) -> float:
        with self._lock:
            return self._estimated_mb_on(device)

    def _require_phase(self, device: str) -> None:
        if device not in self._phase_devices:
            raise RuntimeError(
                f"PhotoRestoreEngine does not own {device!r}: call begin_phase before asking for a session"
            )

    def _cached_or_created(self, spec: RestoreModelSpec, key: SessionKey) -> Any:
        cached = self._sessions.get(key)
        if cached is not None:
            self._sessions.move_to_end(key)
            return cached.session
        model_file = self._model_file(spec, key.precision)
        needed_mb = estimated_vram_mb(model_file, spec.vram_factor)
        self._evict_to_fit(key.device, needed_mb)
        session = self._create_session(
            str(model_file),
            key.device,
            self.settings,
            sess_options_factory=lambda: build_session_options(spec, key.device),
            prefer_native=False,
        )
        self._sessions[key] = _LiveSession(session, needed_mb)
        return session

    def _model_file(self, spec: RestoreModelSpec, precision: str) -> Path:
        path = model_path(self.settings.restore_model_dir_path, spec.id, precision, self._models)
        if path is None:
            raise ValueError(f"Model {spec.id!r} has no {precision} file")
        if not path.is_file():
            raise RuntimeError(missing_pack_message(spec.pack_id, detail=f"No se encontró {path.name}."))
        return path

    def _evict_to_fit(self, device: str, needed_mb: float) -> None:
        while self._keys_on(device) and not self._fits(device, needed_mb):
            del self._sessions[self._keys_on(device)[0]]

    def _fits(self, device: str, needed_mb: float) -> bool:
        within_count = len(self._keys_on(device)) < self.settings.restore_max_live_sessions
        within_vram = self._estimated_mb_on(device) + needed_mb <= self.settings.restore_session_cache_mb
        return within_count and within_vram

    def _keys_on(self, device: str) -> list[SessionKey]:
        return [key for key in self._sessions if key.device == device]

    def _estimated_mb_on(self, device: str) -> float:
        return sum(live.estimated_mb for key, live in self._sessions.items() if key.device == device)
