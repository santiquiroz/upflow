from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.services.engines.audio_restore_base import OnnxAudioRestorer, is_cpu_device
from app.services.engines.audiosr.assets import GRAPH_NAMES, AudioSrAssets
from app.services.engines.audiosr.driver import AudioSrDriver
from app.services.engines.onnx_common import wrap_onnx_error

# ---------------------------------------------------------------------------
# AudioSR restoration (ONNX, in-process). Second restore engine next to
# Apollo: latent-diffusion super-resolution (any band -> 48kHz), much heavier
# but general-purpose. Port: santiquiroz/port-audiosr-onnx (4 graphs + numpy
# DDIM/CFG driver, parity-validated against the PyTorch original).
#
# Session cache, coordinator wiring and audio I/O live in OnnxAudioRestorer
# (audio_restore_base.py), shared with Apollo. The cache holds the 4 graphs
# of ONE device (LRU 1): a full set is ~1.7GB of weights, so caching
# per-device like Apollo would double VRAM/RAM.
#
# TDR: unlike Apollo there is no chunk-size knob -- each DDIM step is one
# monolithic UNet call over the model's fixed 10.24s window (~90ms on a
# 7800 XT). A GPU ~20x slower could hit Windows' ~2s TDR limit; the fallback
# is device=cpu (documented in .env.example).
# ---------------------------------------------------------------------------

AUDIOSR_SAMPLE_RATE = 48000


class AudioSrRestorer(OnnxAudioRestorer):
    sample_rate = AUDIOSR_SAMPLE_RATE
    session_cache_size = 1
    pack_name = "audiosr"
    missing_pack_detail = "Ademas hay que prender ENABLE_AUDIOSR para usar la restauracion AudioSR."
    engine_label = "AudioSR"
    load_error_context = "Failed to load AudioSR models on device"

    def available(self) -> bool:
        return self.settings.audiosr_available()

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
            # racing a straggler save_wav that would resurrect the directory.
            cancel_event.set()
            with contextlib.suppress(Exception):
                await worker
            raise
        self._ensure_output_file(output_wav)

    def _run_and_save(
        self, input_wav: Path, output_wav: Path, device: str, cancel_event: threading.Event
    ) -> None:
        self._require_available()
        audio = self._load_audio(input_wav)
        if audio.shape[0] == 0:
            raise RuntimeError(
                "The uploaded audio decoded to zero samples; the file is empty or corrupted"
            )
        assets = AudioSrAssets.load(self.settings.audiosr_model_dir_path)
        _require_precision_matches_device(assets, device)
        sessions = self._get_sessions(device)
        throttle = 0.0 if is_cpu_device(device) else self.settings.audiosr_gpu_throttle_seconds

        driver = AudioSrDriver(assets, _session_runner(sessions))

        def restore_mono(mono: np.ndarray) -> np.ndarray:
            return driver.restore(
                mono,
                ddim_steps=self.settings.audiosr_ddim_steps,
                cancel_event=cancel_event,
                step_throttle=(lambda: time.sleep(throttle)) if throttle > 0 else None,
            )

        self._restore_and_save(audio, restore_mono, output_wav)

    def _get_sessions(self, device: str) -> dict[str, Any]:
        return self._get_session_for_device(device)

    def _build_session(self, device: str) -> Any:
        # Delegación y no llamada directa desde la base: los tests
        # monkeypatchean `_create_sessions` por nombre en la instancia.
        return self._create_sessions(device)

    def _create_sessions(self, device: str) -> dict[str, Any]:
        # Monkeypatchable seam: unit tests override this to inject fake numpy
        # sessions and never touch real onnxruntime.
        from app.services import ep_registry

        model_dir = self.settings.audiosr_model_dir_path
        return {
            name: ep_registry.create_session(
                str(model_dir / f"{name}.onnx"), device, self.settings
            )
            for name in GRAPH_NAMES
        }


def _require_precision_matches_device(assets: AudioSrAssets, device: str) -> None:
    """El pack fp16 no sirve en CPU, y hay que decirlo ANTES de crear la sesion.

    El EP de CPU tiene muchos menos kernels fp16: los que faltan revientan a
    mitad de la corrida con un error de ONNX Runtime que no dice como salir de
    ahi, y los que existen suelen ser mas lentos que fp32. Como el pack se elige
    al instalar y el dispositivo se elige por trabajo, las dos decisiones pueden
    contradecirse mucho despues.
    """
    if not is_cpu_device(device):
        return
    if assets.manifest.get("precision") != "fp16":
        return
    raise RuntimeError(
        "El pack de AudioSR instalado es fp16 y solo sirve en GPU. Para correrlo "
        "en CPU hay que reinstalarlo en fp32 (el boton de descarga usa la "
        "precision que corresponde al dispositivo por defecto)."
    )


def _session_runner(sessions: dict[str, Any]):
    def run_graph(name: str, feeds: dict[str, np.ndarray]) -> np.ndarray:
        session = sessions[name]
        feeds = {k: np.ascontiguousarray(v) for k, v in feeds.items()}
        try:
            result = session.run(None, feeds)[0]
        except Exception as exc:  # onnxruntime raises its own native exception types
            raise wrap_onnx_error(f"AudioSR {name} inference failed", exc) from exc
        return np.asarray(result, dtype=np.float32)

    return run_graph
