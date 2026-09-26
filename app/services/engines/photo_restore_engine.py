from __future__ import annotations

import asyncio
import contextlib
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np

from app.config import Settings
from app.services import ep_registry
from app.services.devices_service import DevicesService, adapter_free_vram_mb
from app.services.dml_device import try_parse_dml_device_id
from app.services.engines.onnx_video_upscaler import is_device_removed_error
from app.services.engines.restore_canary import canary_rule_for, canary_score, has_non_finite
from app.services.engines.tiled_restore_runner import TileInfer, session_tile_infer
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.missing_pack import missing_pack_message
from app.services.restore_models import (
    RESTORE_MODELS,
    VENDORED_MODELS,
    RestoreModelSpec,
    VendoredModel,
    model_path,
)

SessionFactory = Callable[..., Any]
FreeVramProbe = Callable[[str], int | None]
InferFactory = Callable[[Any], TileInfer]
T = TypeVar("T")

_BYTES_PER_MB = 1024 * 1024
DEVICE_REMOVED_CODE = "restore.error.deviceRemoved"
DEVICE_REMOVED_MESSAGE = "The GPU driver reset. Restart Upflow to use the GPU again."
NON_FINITE_REASON = "NaN/Inf output"


class DeviceHealth(Protocol):
    def mark_unhealthy(self, device_id: str) -> None: ...

    def is_healthy(self, device_id: str) -> bool: ...


class DeviceRemovedError(RuntimeError):
    code = DEVICE_REMOVED_CODE

    def __init__(self, device: str) -> None:
        super().__init__(DEVICE_REMOVED_MESSAGE)
        self.device = device


class NonFiniteOutputError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SessionKey:
    model_id: str
    device: str
    precision: str


@dataclass(frozen=True, slots=True)
class _LiveSession:
    session: Any
    estimated_mb: float


@dataclass(frozen=True, slots=True)
class Fp16Rejection:
    model_id: str
    device: str
    reason: str


@dataclass(frozen=True, slots=True)
class ReadyInfer:
    model_id: str
    device: str
    precision: str
    infer: TileInfer


def guard_finite(output: np.ndarray, model_id: str, precision: str) -> np.ndarray:
    if has_non_finite(output):
        raise NonFiniteOutputError(f"{model_id} / {precision} produced {NON_FINITE_REASON}")
    return output


def clamp_unit(output: np.ndarray) -> np.ndarray:
    return np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)


async def run_cancellable(blocking: Callable[..., T], *args: Any) -> T:
    # Cancelar la tarea asyncio no frena el hilo que sigue en la GPU: se avisa por el evento y se
    # espera al hilo antes de propagar, asi el permiso del device se suelta con la GPU ya libre.
    cancel_event = threading.Event()
    worker = asyncio.ensure_future(asyncio.to_thread(blocking, *args, cancel_event))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancel_event.set()
        with contextlib.suppress(BaseException):
            await worker
        raise


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
        vendored: Mapping[str, VendoredModel] = VENDORED_MODELS,
        create_session: SessionFactory = ep_registry.create_session,
        free_vram_mb: FreeVramProbe = dml_free_vram_mb,
        device_health: DeviceHealth | None = None,
    ) -> None:
        self.settings = settings
        self.gpu_coordinator = gpu_coordinator
        self._vendored = vendored
        self._models = {**{key: model.spec for key, model in vendored.items()}, **models}
        self._create_session = create_session
        self._free_vram_mb = free_vram_mb
        self._device_health = device_health or DevicesService(settings)
        self._sessions: OrderedDict[SessionKey, _LiveSession] = OrderedDict()
        self._phase_devices: set[str] = set()
        self._fp16_verdicts: dict[tuple[str, str], bool] = {}
        self._fp16_rejections: dict[tuple[str, str], Fp16Rejection] = {}
        # Reentrante: la sesion se crea con el candado tomado para que desalojo y alta sean atomicos.
        self._lock = threading.RLock()
        gpu_coordinator.register(self)

    def begin_phase(self, device: str) -> None:
        if not self._device_health.is_healthy(device):
            raise DeviceRemovedError(device)
        self.gpu_coordinator.acquire(device, self)
        with self._lock:
            self._phase_devices.add(device)

    def model_spec(self, model_id: str) -> RestoreModelSpec:
        return self._models[model_id]

    def session(self, model_id: str, device: str, precision: str) -> Any:
        spec = self._models[model_id]
        key = SessionKey(model_id, device, precision)
        with self._lock:
            self._require_phase(device)
            return self._cached_or_created(spec, key)

    def ready_infer(
        self,
        model_id: str,
        device: str,
        sample: np.ndarray,
        *,
        reference_device: str | None = None,
        infer_for: InferFactory = session_tile_infer,
        clamp: bool = True,
    ) -> ReadyInfer:
        precision = self.precision_for(
            model_id, device, sample, reference_device=reference_device, infer_for=infer_for
        )
        infer = self.tile_infer(model_id, device, precision, infer_for=infer_for, clamp=clamp)
        return ReadyInfer(model_id, device, precision, infer)

    def precision_for(
        self,
        model_id: str,
        device: str,
        sample: np.ndarray,
        *,
        reference_device: str | None = None,
        infer_for: InferFactory = session_tile_infer,
    ) -> str:
        if not self._wants_fp16(self._models[model_id], device):
            return "fp32"
        verdict = self._fp16_verdict(model_id, device)
        if verdict is None:
            verdict = self._run_canary(model_id, device, sample, reference_device or device, infer_for)
        return "fp16" if verdict else "fp32"

    def tile_infer(
        self,
        model_id: str,
        device: str,
        precision: str,
        *,
        infer_for: InferFactory = session_tile_infer,
        clamp: bool = True,
    ) -> TileInfer:
        raw = self._raw_infer(model_id, device, precision, infer_for)

        def infer(tile: np.ndarray) -> np.ndarray:
            output = self._guarded(model_id, device, precision, raw(tile))
            return clamp_unit(output) if clamp else output

        return infer

    def fp16_rejections(self) -> tuple[Fp16Rejection, ...]:
        with self._lock:
            return tuple(self._fp16_rejections.values())

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
        session = self._open_session(spec, model_file, key.device)
        self._sessions[key] = _LiveSession(session, needed_mb)
        return session

    def _open_session(self, spec: RestoreModelSpec, model_file: Path, device: str) -> Any:
        return self._create_session(
            str(model_file),
            device,
            self.settings,
            sess_options_factory=lambda: build_session_options(spec, device),
            prefer_native=False,
        )

    def _wants_fp16(self, spec: RestoreModelSpec, device: str) -> bool:
        return is_gpu_device(device) and spec.fp16_filename is not None and self.settings.onnx_prefer_fp16

    def _fp16_verdict(self, model_id: str, device: str) -> bool | None:
        with self._lock:
            return self._fp16_verdicts.get((model_id, device))

    def _run_canary(
        self, model_id: str, device: str, sample: np.ndarray, reference_device: str, infer_for: InferFactory
    ) -> bool:
        candidate = self._raw_infer(model_id, device, "fp16", infer_for)(sample)
        if has_non_finite(candidate):
            return self._record_fp16(model_id, device, NON_FINITE_REASON)
        reference = self._reference_output(model_id, reference_device, sample, infer_for)
        rule = canary_rule_for(model_id)
        score = canary_score(rule, guard_finite(reference, model_id, "fp32"), candidate)
        return self._record_fp16(model_id, device, None if rule.passes(score) else rule.describe(score))

    def _reference_output(
        self, model_id: str, reference_device: str, sample: np.ndarray, infer_for: InferFactory
    ) -> np.ndarray:
        if is_gpu_device(reference_device):
            return self._raw_infer(model_id, reference_device, "fp32", infer_for)(sample)
        # Referencia en CPU: sesion efimera que no ocupa lugar en el LRU ni pide fase.
        spec = self._models[model_id]
        session = self._open_session(spec, self._model_file(spec, "fp32"), reference_device)
        return infer_for(session)(sample)

    def _record_fp16(self, model_id: str, device: str, rejection: str | None) -> bool:
        key = (model_id, device)
        with self._lock:
            self._fp16_verdicts[key] = rejection is None
            if rejection is not None:
                self._fp16_rejections[key] = Fp16Rejection(model_id, device, rejection)
                self._sessions.pop(SessionKey(model_id, device, "fp16"), None)
        return rejection is None

    def _raw_infer(self, model_id: str, device: str, precision: str, infer_for: InferFactory) -> TileInfer:
        def infer(tile: np.ndarray) -> np.ndarray:
            with self.removal_classified(device):
                return infer_for(self.session(model_id, device, precision))(tile)

        return infer

    def _guarded(self, model_id: str, device: str, precision: str, output: np.ndarray) -> np.ndarray:
        try:
            return guard_finite(output, model_id, precision)
        except NonFiniteOutputError:
            if precision == "fp16":
                self._record_fp16(model_id, device, NON_FINITE_REASON)
            raise

    @contextlib.contextmanager
    def removal_classified(self, device: str) -> Iterator[None]:
        try:
            yield
        except Exception as exc:
            if not (is_gpu_device(device) and is_device_removed_error(exc)):
                raise
            self._on_device_removed(device)
            raise DeviceRemovedError(device) from exc

    def _on_device_removed(self, device: str) -> None:
        # Sin reintento: D3D12 comparte un device por adaptador y proceso, asi que murieron todas las sesiones.
        self._device_health.mark_unhealthy(device)
        self.gpu_coordinator.invalidate_device(device)

    def _model_file(self, spec: RestoreModelSpec, precision: str) -> Path:
        vendored = self._vendored.get(spec.id)
        if vendored is not None:
            return self._vendored_file(vendored, precision)
        path = model_path(self.settings.restore_model_dir_path, spec.id, precision, self._models)
        if path is None:
            raise ValueError(f"Model {spec.id!r} has no {precision} file")
        if not path.is_file():
            raise RuntimeError(missing_pack_message(spec.pack_id, detail=f"No se encontró {path.name}."))
        return path

    def _vendored_file(self, vendored: VendoredModel, precision: str) -> Path:
        if precision not in vendored.spec.files_by_precision():
            raise ValueError(f"Model {vendored.spec.id!r} has no {precision} file")
        path = Path(vendored.path_of(self.settings))
        if not path.is_file():
            raise RuntimeError(missing_pack_message(vendored.pack, detail=f"No se encontró {path.name}."))
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
