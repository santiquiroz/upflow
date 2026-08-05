from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings
from app.services.media_decode import build_decode_to_wav_command, needs_decoding
from app.services.subtitles import (
    TranscriptSegment,
    merge_chunk_segments,
    segments_from_offsets,
)

logger = logging.getLogger(__name__)

# Whisper trabaja a 16 kHz mono. No es preferencia: el extractor de features lo
# asume y no remuestrea por su cuenta.
TARGET_SAMPLE_RATE = 16000

# El extractor RELLENA O TRUNCA todo a exactamente 30 segundos. Medido: 3 s, 30 s y
# 120 s de entrada producen las tres el mismo tensor (1, 80, 3000). Sin trocear, un
# audio de dos minutos pierde noventa segundos EN SILENCIO, que es la peor forma de
# equivocarse. Por eso el troceo no es una optimizacion, es correctitud.
CHUNK_SECONDS = 30
CHUNK_SAMPLES = TARGET_SAMPLE_RATE * CHUNK_SECONDS

# Presupuesto de tokens por trozo. Treinta segundos de habla densa no pasan de esto
# y el techo evita que un modelo que entra en bucle corra para siempre.
MAX_TOKENS_PER_CHUNK = 220


def build_load_kwargs(device: str, settings: Settings | None = None) -> dict[str, Any]:
    """Los kwargs con los que se carga el modelo.

    Se extrae de _create_model para que la bandera que decide si transcribe o
    devuelve ruido se pueda verificar sin mockear optimum ni transformers.

    Los providers salen del ep_registry, no armados a mano: si no, un plugin EP
    instalado aceleraba escalado y editor pero no transcripcion, sin avisar.
    """
    from app.config import get_settings
    from app.services import ep_registry

    kwargs: dict[str, Any] = {
        # Vetado con DirectML en este repo.
        "use_io_binding": False,
        # NO es opcional. Con el decoder merged, que es el DEFAULT, DirectML carga,
        # corre sin excepcion y devuelve BASURA: medido 2026-07-29, "I know Kung
        # Fu." se convierte en "Ioti plur plur plur fighting...". Ver
        # docs/superpowers/specs/2026-07-29-whisper-onnx-spike-findings.md
        "use_merged": False,
    }
    kwargs.update(ep_registry.loader_kwargs(device, settings or get_settings()))
    return kwargs


class TranscribeCancelled(Exception):
    pass


@dataclass(slots=True, frozen=True)
class TranscribeRequest:
    # None deja que el modelo detecte el idioma. Los modelos `.en` son
    # monolingues y lo ignoran.
    language: str | None = None


def _resample_linear(audio: Any, source_rate: int) -> Any:
    import numpy as np

    if source_rate == TARGET_SAMPLE_RATE:
        return audio
    target_length = int(len(audio) * TARGET_SAMPLE_RATE / source_rate)
    return np.interp(
        np.linspace(0, len(audio), target_length, endpoint=False),
        np.arange(len(audio)),
        audio,
    ).astype("float32")


def _decoded_copy(path: Path, ffmpeg: str) -> Path:
    """Deja un WAV al lado del original y devuelve su ruta.

    Se hace ANTES de tocar el modelo: si el contenedor no sirve, el error sale
    de ffmpeg con un mensaje entendible y sin haber tomado la GPU.
    """
    import subprocess

    destination = path.with_suffix(".decoded.wav")
    command = build_decode_to_wav_command(
        ffmpeg=ffmpeg, source=path, destination=destination, sample_rate=TARGET_SAMPLE_RATE
    )
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not destination.exists():
        raise RuntimeError(
            "No se pudo leer el audio de este archivo. "
            f"ffmpeg dijo: {(result.stderr or '').strip()[-300:]}"
        )
    return destination


def load_audio_mono_16k(path: Path) -> Any:
    """Deja el audio como espera Whisper: mono, float32, 16 kHz."""
    import soundfile as sf

    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    return _resample_linear(mono, sample_rate)


def split_into_chunks(audio: Any) -> list[Any]:
    """Trozos de 30 segundos, SIN solape.

    Nota historica: el docstring anterior decia que `return_timestamps=True` no
    devuelve marcas en este camino (medido 2026-07-29). Eso quedo DESMENTIDO el
    2026-08-04 (`scripts/spike_whisper_timestamps.py`): las marcas existen, lo
    que las borraba era decodificar con `skip_special_tokens=True`. Ver
    docs/superpowers/specs/2026-08-04-whisper-timestamps-correccion.md.

    El solape sigue sin usarse porque no hace falta para subtitulos: los
    segmentos ya vienen cortados por frase.
    """
    if len(audio) == 0:
        return []
    return [audio[start : start + CHUNK_SAMPLES] for start in range(0, len(audio), CHUNK_SAMPLES)]


class TranscribeEngine:
    def __init__(self, settings: Settings, gpu_coordinator: Any) -> None:
        self.settings = settings
        self.gpu_coordinator = gpu_coordinator
        self._cache: OrderedDict[tuple[str, str], tuple[Any, Any]] = OrderedDict()
        self._cache_lock = threading.Lock()

    def release_device(self, device: str) -> None:
        with self._cache_lock:
            for key in [key for key in self._cache if key[1] == device]:
                self._cache.pop(key, None)

    async def run(
        self,
        *,
        model_id: str,
        model_dir: Path,
        audio_path: Path,
        request: TranscribeRequest,
        device: str,
        progress_cb: Callable[[int, int], None],
    ) -> list[TranscriptSegment]:
        cancel_event = threading.Event()
        worker = asyncio.ensure_future(
            asyncio.to_thread(
                self._run_blocking,
                model_id,
                model_dir,
                audio_path,
                request,
                device,
                cancel_event,
                progress_cb,
            )
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancel_event.set()
            with contextlib.suppress(BaseException):
                await worker
            raise

    def _run_blocking(
        self,
        model_id: str,
        model_dir: Path,
        audio_path: Path,
        request: TranscribeRequest,
        device: str,
        cancel_event: threading.Event,
        progress_cb: Callable[[int, int], None],
    ) -> list[TranscriptSegment]:
        # Decodificar ANTES de pedir el modelo: un contenedor de video que no se
        # puede leer tiene que fallar sin haber tomado el dispositivo.
        decoded: Path | None = None
        if needs_decoding(audio_path):
            decoded = _decoded_copy(audio_path, str(self.settings.ffmpeg_binary_path))
        try:
            model, processor = self._get_model(model_id, model_dir, device)
            chunks = split_into_chunks(load_audio_mono_16k(decoded or audio_path))
            if not chunks:
                return []

            per_chunk: list[list[TranscriptSegment]] = []
            for index, chunk in enumerate(chunks):
                if cancel_event.is_set():
                    raise TranscribeCancelled()
                per_chunk.append(self._transcribe_chunk(model, processor, chunk, request))
                progress_cb(index + 1, len(chunks))

            return merge_chunk_segments(per_chunk, chunk_seconds=CHUNK_SECONDS)
        finally:
            if decoded is not None:
                decoded.unlink(missing_ok=True)

    def _transcribe_chunk(
        self, model: Any, processor: Any, chunk: Any, request: TranscribeRequest
    ) -> list[TranscriptSegment]:
        features = processor(
            chunk, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        ).input_features
        kwargs: dict[str, Any] = {
            "max_new_tokens": MAX_TOKENS_PER_CHUNK,
            # Sin esto el modelo no emite los tokens de tiempo y no hay
            # subtitulos posibles, solo texto corrido.
            "return_timestamps": True,
        }
        if request.language:
            kwargs["language"] = request.language
        tokens = model.generate(input_features=features, **kwargs)
        decoded = processor.batch_decode(
            tokens, skip_special_tokens=True, output_offsets=True
        )[0]
        offsets = decoded.get("offsets") if isinstance(decoded, dict) else None
        if offsets:
            return segments_from_offsets(list(offsets), chunk_seconds=CHUNK_SECONDS)
        # Sin offsets todavia se puede entregar la transcripcion: el chunk entero
        # pasa a ser un solo segmento. Peor granularidad, nunca texto perdido.
        text = decoded.get("text") if isinstance(decoded, dict) else decoded
        return [TranscriptSegment(start=0.0, end=float(CHUNK_SECONDS), text=str(text or ""))]

    def _get_model(self, model_id: str, model_dir: Path, device: str) -> tuple[Any, Any]:
        self.gpu_coordinator.acquire(device, self)
        key = (model_id, device)
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
        pair = self._create_model(model_dir, device)
        with self._cache_lock:
            self._cache[key] = pair
            self._cache.move_to_end(key)
            while len(self._cache) > 1:
                self._cache.popitem(last=False)
        return pair

    def _create_model(self, model_dir: Path, device: str) -> tuple[Any, Any]:
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        from transformers import AutoProcessor

        model = ORTModelForSpeechSeq2Seq.from_pretrained(
            str(model_dir), **build_load_kwargs(device, self.settings)
        )
        processor = AutoProcessor.from_pretrained(str(model_dir))
        return model, processor
