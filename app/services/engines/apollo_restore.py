from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings
from app.services.dml_device import DML_DEVICE_PREFIX, parse_dml_device_id
from app.services.engines.audio_restore_base import OnnxAudioRestorer, is_cpu_device
from app.services.engines.onnx_upscaler import _wrap_onnx_error
from app.services.gpu_session_coordinator import GpuSessionCoordinator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Apollo audio restoration (ONNX, in-process). Reconstructs the high band a
# lossy codec threw away. The model is audio->audio, self-contained, 44.1kHz
# mono, input tensor "audio" [1,1,n] -> output "restored" [1,1,n].
#
# Multi-EP: `dml:N` runs on DirectML(device_id=N) (any AMD/NVIDIA/Intel GPU)
# and `cpu` on the CPU EP -- NOT AMD-specific. Session cache, coordinator
# wiring and audio I/O live in OnnxAudioRestorer (audio_restore_base.py),
# shared with AudioSR; here the cache is LRU(2) keyed by device.
#
# Chunking: DirectML breaks on long tensors, so inference runs in chunks of
# AUDIO_RESTORE_CHUNK_SECONDS with a 0.5s Hann overlap-add (ported from the
# validated apollo spike chunked_dml.py). available() is gated purely by
# settings.audio_restore_available(): a missing model/flag yields False with no
# exception, so the app never breaks when Apollo is not installed.
# ---------------------------------------------------------------------------

APOLLO_SAMPLE_RATE = 44100
# 0.15s de crossfade Hann alcanza para juntar chunks sin costura audible; con chunks
# chicos (1.0s, por el limite TDR) un overlap de 0.5s era 50% de computo redundante.
OVERLAP_SECONDS = 0.15
# En CPU los chunks son grandes (~30s), asi que un crossfade de 0.5s es redundancia
# despreciable y suaviza mejor los pocos bordes.
CPU_OVERLAP_SECONDS = 0.5
SESSION_CACHE_SIZE = 2
ONNX_INPUT_NAME = "audio"
ONNX_OUTPUT_NAME = "restored"


def _import_onnxruntime() -> Any:
    import onnxruntime as ort

    return ort


class ApolloRestorer(OnnxAudioRestorer):
    sample_rate = APOLLO_SAMPLE_RATE
    session_cache_size = SESSION_CACHE_SIZE
    pack_name = "apollo"
    missing_pack_detail = "Ademas hay que prender ENABLE_AUDIO_RESTORE."
    engine_label = "Apollo"
    load_error_context = "Failed to load Apollo model on device"

    def __init__(self, settings: Settings, gpu_coordinator: GpuSessionCoordinator) -> None:
        super().__init__(settings, gpu_coordinator)
        self._iobinding_warned = False

    def available(self) -> bool:
        return self.settings.audio_restore_available()

    async def run(self, input_wav: Path, output_wav: Path, device: str) -> None:
        # available()/session build/inference all touch native libraries
        # (onnxruntime, soundfile) so they run off the event loop in one
        # to_thread call, mirroring OnnxUpscaler.run.
        await asyncio.to_thread(self._run_and_save, input_wav, output_wav, device)
        self._ensure_output_file(output_wav)

    def _run_and_save(self, input_wav: Path, output_wav: Path, device: str) -> None:
        self._require_available()
        audio = self._load_audio(input_wav)
        session = self._get_session(device)
        is_cpu = is_cpu_device(device)
        # En GPU (dml:N) el cómputo satura la única tarjeta y el escritorio se
        # laguea; un respiro entre chunks le devuelve la GPU al compositor. En CPU
        # no hay contencion de GPU, asi que no se aplica (seria solo mas lento).
        throttle = 0.0 if is_cpu else self.settings.audio_restore_gpu_throttle_seconds
        # CPU no tiene el limite TDR de DirectML, asi que usa chunks GRANDES: el
        # modelo ve mas contexto y hay menos bordes -> mejor calidad (los chunks
        # de 1s de la GPU emborronan pasajes dinamicos por falta de contexto y
        # crossfades frecuentes). GPU se queda con el chunk chico TDR-safe.
        chunk_seconds = self.settings.audio_restore_cpu_chunk_seconds if is_cpu else self.settings.audio_restore_chunk_seconds
        overlap_seconds = CPU_OVERLAP_SECONDS if is_cpu else OVERLAP_SECONDS

        def restore_mono(mono: np.ndarray) -> np.ndarray:
            return self._restore_chunked(session, mono, device, throttle, chunk_seconds, overlap_seconds)

        self._restore_and_save(audio, restore_mono, output_wav)

    def _restore_chunked(
        self, session: Any, audio: np.ndarray, device: str, throttle: float = 0.0,
        chunk_seconds: float = 1.0, overlap_seconds: float = OVERLAP_SECONDS,
    ) -> np.ndarray:
        total = audio.shape[-1]
        window_length = max(1, int(chunk_seconds * APOLLO_SAMPLE_RATE))
        overlap = min(int(overlap_seconds * APOLLO_SAMPLE_RATE), window_length // 2)
        hop = max(1, window_length - overlap)
        window = np.hanning(window_length).astype(np.float64)

        accumulator = np.zeros(total, dtype=np.float64)
        weight_sum = np.zeros(total, dtype=np.float64)
        start = 0
        while start < total:
            end = min(start + window_length, total)
            segment = audio[start:end]
            weights = window[: end - start]
            restored = self._infer_chunk(session, segment, device)
            accumulator[start:end] += restored * weights
            weight_sum[start:end] += weights
            if end >= total:
                break
            start += hop
            if throttle > 0:
                time.sleep(throttle)  # deja respirar al escritorio entre inferencias GPU
        return (accumulator / np.maximum(weight_sum, 1e-8)).astype(np.float32)

    def _infer_chunk(self, session: Any, segment: np.ndarray, device: str) -> np.ndarray:
        batch = segment.reshape(1, 1, -1).astype(np.float32)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        if device.startswith(DML_DEVICE_PREFIX):
            bound = self._infer_iobinding(session, batch, input_name, output_name, device)
            if bound is not None:
                return np.asarray(bound, dtype=np.float64).reshape(-1)
        try:
            result = session.run([output_name], {input_name: batch})[0]
        except Exception as exc:  # onnxruntime raises its own native exception types
            raise _wrap_onnx_error("Apollo inference failed", exc) from exc
        return np.asarray(result, dtype=np.float64).reshape(-1)

    def _infer_iobinding(
        self, session: Any, batch: np.ndarray, input_name: str, output_name: str, device: str
    ) -> np.ndarray | None:
        # Best-effort, same contract as OnnxVideoUpscaler._infer_iobinding: any
        # failure (older ort, EP quirk, missing io_binding on a test double)
        # falls back to a plain run rather than failing the job, and a
        # persistent failure is logged once (not once per chunk) so it
        # doesn't silently downgrade every chunk to the slower path forever.
        try:
            ort = _import_onnxruntime()
            device_id = parse_dml_device_id(device)
            io_binding = session.io_binding()
            input_value = ort.OrtValue.ortvalue_from_numpy(batch, "dml", device_id)
            io_binding.bind_ortvalue_input(input_name, input_value)
            io_binding.bind_output(output_name, "dml")
            session.run_with_iobinding(io_binding)
            return io_binding.copy_outputs_to_cpu()[0]
        except Exception:  # noqa: BLE001
            if not self._iobinding_warned:
                self._iobinding_warned = True
                logger.warning(
                    "Apollo ONNX IO binding failed on %s; falling back to the slower plain-run path", device,
                    exc_info=True,
                )
            return None

    def _get_session(self, device: str) -> Any:
        return self._get_session_for_device(device)

    def _build_session(self, device: str) -> Any:
        # Delegación y no llamada directa desde la base: los tests
        # monkeypatchean `_create_session` por nombre en la instancia.
        return self._create_session(device)

    def _create_session(self, device: str) -> Any:
        # Monkeypatchable seam: unit tests override this to inject a fake numpy
        # session and never touch real onnxruntime.
        from app.services import ep_registry

        return ep_registry.create_session(
            str(self.settings.apollo_restore_model_path), device, self.settings
        )
