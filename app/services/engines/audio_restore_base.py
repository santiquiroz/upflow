from __future__ import annotations

import threading
from collections import OrderedDict
from math import gcd
from pathlib import Path
from typing import Any, Callable, ClassVar

import numpy as np

from app.config import Settings
from app.services.engines.multichannel_restore import restore_multichannel
from app.services.engines.onnx_common import wrap_onnx_error
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.missing_pack import missing_pack_message
from app.services.process_runner import is_non_empty_file

# ---------------------------------------------------------------------------
# Shared base for the in-process ONNX audio restorers (Apollo, AudioSR):
# settings + GpuSessionCoordinator wiring, the per-device LRU session cache
# (built outside the lock, mirroring OnnxUpscaler._get_session) and the audio
# I/O helpers at the class sample rate. Subclasses keep their own
# `_create_session` / `_create_sessions` seams -- tests monkeypatch those BY
# NAME -- so the base only reaches them through the `_build_session`
# delegation each subclass defines.
# ---------------------------------------------------------------------------


def is_cpu_device(device: str) -> bool:
    return device.strip().lower() == "cpu"


def resample_audio(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return signal.astype(np.float32)
    from scipy.signal import resample_poly

    divisor = gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // divisor
    down = int(source_rate) // divisor
    return resample_poly(signal, up, down).astype(np.float32)


def load_audio(input_wav: Path, target_rate: int) -> np.ndarray:
    import soundfile as sf

    data, sample_rate = sf.read(str(input_wav), dtype="float32", always_2d=True)
    channels = [resample_audio(data[:, c], sample_rate, target_rate) for c in range(data.shape[1])]
    return np.stack(channels, axis=1)


def save_wav(output_wav: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), audio, sample_rate)


class OnnxAudioRestorer:
    sample_rate: ClassVar[int]
    session_cache_size: ClassVar[int] = 1
    pack_name: ClassVar[str]
    missing_pack_detail: ClassVar[str]
    engine_label: ClassVar[str]
    load_error_context: ClassVar[str]

    def __init__(self, settings: Settings, gpu_coordinator: GpuSessionCoordinator) -> None:
        self.settings = settings
        self.gpu_coordinator = gpu_coordinator
        self._session_cache: OrderedDict[str, Any] = OrderedDict()
        self._session_lock = threading.Lock()

    def available(self) -> bool:
        raise NotImplementedError

    def _build_session(self, device: str) -> Any:
        raise NotImplementedError

    def release_device(self, device: str) -> None:
        with self._session_lock:
            self._session_cache.pop(device, None)

    def _require_available(self) -> None:
        if not self.available():
            raise RuntimeError(missing_pack_message(self.pack_name, detail=self.missing_pack_detail))

    def _get_session_for_device(self, device: str) -> Any:
        self.gpu_coordinator.acquire(device, self)
        with self._session_lock:
            cached = self._session_cache.get(device)
            if cached is not None:
                self._session_cache.move_to_end(device)
                return cached

        # Session build happens outside the lock (slow graph load); a rare
        # double-build on a concurrent miss just wastes work (last insert
        # wins), same trade-off as OnnxUpscaler._get_session.
        try:
            session = self._build_session(device)
        except Exception as exc:  # onnxruntime raises its own native exception types
            raise wrap_onnx_error(f"{self.load_error_context} {device!r}", exc) from exc

        with self._session_lock:
            self._session_cache[device] = session
            self._session_cache.move_to_end(device)
            if len(self._session_cache) > self.session_cache_size:
                self._session_cache.popitem(last=False)
        return session

    def _load_audio(self, input_wav: Path) -> np.ndarray:
        return load_audio(input_wav, self.sample_rate)

    def _restore_and_save(
        self, audio: np.ndarray, restore_mono: Callable[[np.ndarray], np.ndarray], output_wav: Path
    ) -> None:
        restored = restore_multichannel(audio, restore_mono)
        save_wav(output_wav, restored, self.sample_rate)

    def _ensure_output_file(self, output_wav: Path) -> None:
        if not is_non_empty_file(output_wav):
            raise RuntimeError(
                f"{self.engine_label} restoration completed but no output file was produced"
            )
