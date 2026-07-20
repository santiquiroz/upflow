from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections import OrderedDict
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings
from app.services.engines.audiosr.assets import GRAPH_NAMES, AudioSrAssets
from app.services.engines.audiosr.driver import AudioSrDriver
from app.services.engines.onnx_upscaler import _build_providers, _wrap_onnx_error
from app.services.gpu_session_coordinator import GpuSessionCoordinator

# ---------------------------------------------------------------------------
# AudioSR restoration (ONNX, in-process). Second restore engine next to
# Apollo: latent-diffusion super-resolution (any band -> 48kHz), much heavier
# but general-purpose. Port: santiquiroz/port-audiosr-onnx (4 graphs + numpy
# DDIM/CFG driver, parity-validated against the PyTorch original).
#
# Session cache holds the 4 graphs of ONE device (LRU 1): a full set is
# ~1.7GB of weights, so caching per-device like Apollo would double VRAM/RAM.
#
# TDR: unlike Apollo there is no chunk-size knob -- each DDIM step is one
# monolithic UNet call over the model's fixed 10.24s window (~90ms on a
# 7800 XT). A GPU ~20x slower could hit Windows' ~2s TDR limit; the fallback
# is device=cpu (documented in .env.example).
# ---------------------------------------------------------------------------

AUDIOSR_SAMPLE_RATE = 48000


class AudioSrRestorer:
    def __init__(self, settings: Settings, gpu_coordinator: GpuSessionCoordinator) -> None:
        self.settings = settings
        self.gpu_coordinator = gpu_coordinator
        self._session_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._session_lock = threading.Lock()

    def available(self) -> bool:
        return self.settings.audiosr_available()

    def release_device(self, device: str) -> None:
        with self._session_lock:
            self._session_cache.pop(device, None)

    async def run(self, input_wav: Path, output_wav: Path, device: str) -> None:
        cancel_event = threading.Event()
        worker = asyncio.ensure_future(
            asyncio.to_thread(self._run_and_save, input_wav, output_wav, device, cancel_event)
        )
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            # to_thread can't interrupt the worker; the driver polls this event
            # at every stage boundary. Waiting for the thread to actually
            # finish keeps the caller's finally (rmtree of the work dir) from
            # racing a straggler _save_wav that would resurrect the directory.
            cancel_event.set()
            with contextlib.suppress(Exception):
                await worker
            raise
        if not self._is_non_empty_file(output_wav):
            raise RuntimeError("AudioSR restoration completed but no output file was produced")

    def _run_and_save(
        self, input_wav: Path, output_wav: Path, device: str, cancel_event: threading.Event
    ) -> None:
        if not self.available():
            raise RuntimeError(
                "AudioSR restoration is not available. Enable ENABLE_AUDIOSR and install the models "
                "(scripts/download-audiosr-onnx.ps1)."
            )
        audio = _load_mono_48k(input_wav)
        if audio.shape[-1] == 0:
            raise RuntimeError(
                "The uploaded audio decoded to zero samples; the file is empty or corrupted"
            )
        sessions = self._get_sessions(device)
        assets = AudioSrAssets.load(self.settings.audiosr_model_dir_path)
        throttle = 0.0 if _is_cpu_device(device) else self.settings.audiosr_gpu_throttle_seconds

        driver = AudioSrDriver(assets, _session_runner(sessions))
        restored = driver.restore(
            audio,
            ddim_steps=self.settings.audiosr_ddim_steps,
            cancel_event=cancel_event,
            step_throttle=(lambda: time.sleep(throttle)) if throttle > 0 else None,
        )
        _save_wav(output_wav, restored)

    def _get_sessions(self, device: str) -> dict[str, Any]:
        self.gpu_coordinator.acquire(device, self)
        with self._session_lock:
            cached = self._session_cache.get(device)
            if cached is not None:
                self._session_cache.move_to_end(device)
                return cached

        try:
            sessions = self._create_sessions(device)
        except Exception as exc:  # onnxruntime raises its own native exception types
            raise _wrap_onnx_error(
                f"Failed to load AudioSR models on device {device!r}", exc
            ) from exc

        with self._session_lock:
            self._session_cache[device] = sessions
            self._session_cache.move_to_end(device)
            if len(self._session_cache) > 1:
                self._session_cache.popitem(last=False)
        return sessions

    def _create_sessions(self, device: str) -> dict[str, Any]:
        # Monkeypatchable seam: unit tests override this to inject fake numpy
        # sessions and never touch real onnxruntime.
        import onnxruntime as ort

        providers = _build_providers(device)
        model_dir = self.settings.audiosr_model_dir_path
        return {
            name: ort.InferenceSession(str(model_dir / f"{name}.onnx"), providers=providers)
            for name in GRAPH_NAMES
        }

    @staticmethod
    def _is_non_empty_file(path: Path) -> bool:
        return path.exists() and path.stat().st_size > 0


def _session_runner(sessions: dict[str, Any]):
    def run_graph(name: str, feeds: dict[str, np.ndarray]) -> np.ndarray:
        session = sessions[name]
        feeds = {k: np.ascontiguousarray(v) for k, v in feeds.items()}
        try:
            result = session.run(None, feeds)[0]
        except Exception as exc:  # onnxruntime raises its own native exception types
            raise _wrap_onnx_error(f"AudioSR {name} inference failed", exc) from exc
        return np.asarray(result, dtype=np.float32)

    return run_graph


def _is_cpu_device(device: str) -> bool:
    return device.strip().lower() == "cpu"


def _load_mono_48k(input_wav: Path) -> np.ndarray:
    import soundfile as sf

    data, sample_rate = sf.read(str(input_wav), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    return _resample(mono, sample_rate, AUDIOSR_SAMPLE_RATE)


def _resample(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return signal.astype(np.float32)
    from scipy.signal import resample_poly

    divisor = gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // divisor
    down = int(source_rate) // divisor
    return resample_poly(signal, up, down).astype(np.float32)


def _save_wav(output_wav: Path, audio: np.ndarray) -> None:
    import soundfile as sf

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), audio, AUDIOSR_SAMPLE_RATE)
