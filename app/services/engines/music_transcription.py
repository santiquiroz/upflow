"""Transcripcion de un stem a notas (Basic Pitch ONNX, F3a).

`nmp.onnx` (spotify/basic-pitch, Apache-2.0) es un grafo AUTOCONTENIDO: recibe
audio crudo mono a 22050 Hz y calcula el CQT armonico ADENTRO del grafo, asi
que aca no hace falta reimplementar ningun frontend espectral — solo ventanear
el audio como lo espera el modelo y desventanear su salida. Las constantes de
ventaneo/anotacion (AUDIO_N_SAMPLES, FFT_HOP, ANNOTATIONS_FPS, el solape de
DEFAULT_OVERLAPPING_FRAMES, los nombres de tensor de entrada/salida) son las
publicas de `basic_pitch/constants.py` e `inference.py`; se copian aca porque
la politica del repo es cero deps nuevas (nada de instalar el paquete
`basic-pitch`, que trae tensorflow).

Postprocesado: esto NO reimplementa el algoritmo completo de Spotify
(melodia trick, inferencia de pitch bend desde el contour, filtro por
frecuencia). `notes_from_activations` es una segmentacion mas simple —
agrupa frames contiguos por encima de `frame_threshold` en cada bin de altura,
cortando en un nuevo `onset_threshold` — a proposito: la salida es un
BORRADOR editable (contrato F3a), no una transcripcion definitiva.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from app.services.engines.audio_restore_base import load_audio
from app.services.engines.onnx_common import get_cached_session

# --- constantes publicas de basic-pitch (constants.py) ---------------------

AUDIO_SAMPLE_RATE = 22050
FFT_HOP = 256
AUDIO_WINDOW_LENGTH = 2  # segundos
ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP  # 86
ANNOTATION_HOP = 1.0 / ANNOTATIONS_FPS
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP  # 43844
N_FREQ_BINS_NOTES = 88
# El bin 0 de "note"/"onset" es A0 (27.5 Hz), la tecla mas grave del piano.
MIDI_BASE_NOTE = 21
DEFAULT_OVERLAPPING_FRAMES = 30

# --- umbrales publicos de basic-pitch (inference.py / note_creation.py) ----

DEFAULT_ONSET_THRESHOLD = 0.5
DEFAULT_FRAME_THRESHOLD = 0.3
DEFAULT_MINIMUM_NOTE_LENGTH_MS = 127.7
MIN_NOTE_FRAMES = max(1, round((DEFAULT_MINIMUM_NOTE_LENGTH_MS / 1000.0) / ANNOTATION_HOP))

# --- contrato ONNX del grafo publicado (inference.py::Model.predict) -------

_ONNX_INPUT_NAME = "serving_default_input_2:0"
_ONNX_OUTPUT_NOTE = "StatefulPartitionedCall:1"
_ONNX_OUTPUT_ONSET = "StatefulPartitionedCall:2"
_ONNX_OUTPUT_CONTOUR = "StatefulPartitionedCall:0"

# --- cuantizacion (decision del implementador, contrato F3a #9/#10): sin ---
# deteccion de tempo real, se asume 120 BPM fijo y una grilla de semicorchea
# para que el MIDI/MusicXML crudo (con onsets a milisegundos sueltos) sea
# leible en un editor de partituras. midi_writer y musicxml_writer reusan
# estas constantes para que los dos formatos queden alineados a la MISMA
# grilla.
DEFAULT_TEMPO_BPM = 120.0
DEFAULT_GRID_SECONDS = 60.0 / DEFAULT_TEMPO_BPM / 4.0  # semicorchea


@dataclass(frozen=True, slots=True)
class NoteEvent:
    onset_time: float
    offset_time: float
    pitch_midi: int
    confidence: float


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return audio.mean(axis=1).astype(np.float32)


def notes_from_activations(
    note_probs: np.ndarray,
    onset_probs: np.ndarray,
    *,
    onset_threshold: float = DEFAULT_ONSET_THRESHOLD,
    frame_threshold: float = DEFAULT_FRAME_THRESHOLD,
    minimum_note_frames: int = MIN_NOTE_FRAMES,
) -> list[NoteEvent]:
    """Nota = frames contiguos de UN bin de altura con `note_probs` por encima
    de `frame_threshold`. Un nuevo `onset_threshold` a mitad de un tramo activo
    corta la nota anterior y arranca una nueva (permite dos notas seguidas en
    el mismo bin sin silencio entre medio). `pitch_midi = MIDI_BASE_NOTE + bin`
    porque basic-pitch usa un bin por semitono empezando en A0 (MIDI 21).

    Puro numpy, sin ONNX: es lo que hace determinista y testeable el contrato
    "matriz de activacion conocida -> eventos MIDI exactos".
    """
    frames, bins = note_probs.shape
    events: list[NoteEvent] = []
    for pitch in range(bins):
        start: int | None = None
        window: list[float] = []
        for t in range(frames):
            active = bool(note_probs[t, pitch] >= frame_threshold)
            is_onset = bool(onset_probs[t, pitch] >= onset_threshold)
            if active and (start is None or (is_onset and t > start)):
                if start is not None:
                    events.append(_close_note(start, t - 1, pitch, window))
                start = t
                window = []
            if active:
                window.append(float(note_probs[t, pitch]))
            elif start is not None:
                events.append(_close_note(start, t - 1, pitch, window))
                start = None
                window = []
        if start is not None:
            events.append(_close_note(start, frames - 1, pitch, window))
    kept = [event for event in events if _note_frames(event) >= minimum_note_frames]
    kept.sort(key=lambda event: (event.onset_time, event.pitch_midi))
    return kept


def _note_frames(event: NoteEvent) -> int:
    return round((event.offset_time - event.onset_time) / ANNOTATION_HOP)


def _close_note(start_frame: int, end_frame: int, pitch: int, window: list[float]) -> NoteEvent:
    confidence = float(np.mean(window)) if window else 0.0
    return NoteEvent(
        onset_time=start_frame * ANNOTATION_HOP,
        offset_time=(end_frame + 1) * ANNOTATION_HOP,
        pitch_midi=MIDI_BASE_NOTE + pitch,
        confidence=confidence,
    )


def quantize_notes(
    notes: list[NoteEvent], grid_seconds: float = DEFAULT_GRID_SECONDS
) -> list[NoteEvent]:
    """Redondea onset/offset al multiplo de `grid_seconds` mas cercano, sin
    detectar tempo real (decision #9/#10 del contrato F3a): asume una grilla
    fija para que la salida sea legible en un editor de partituras, no una
    transcripcion ritmicamente exacta."""
    quantized: list[NoteEvent] = []
    for note in notes:
        onset = round(note.onset_time / grid_seconds) * grid_seconds
        offset = round(note.offset_time / grid_seconds) * grid_seconds
        if offset <= onset:
            offset = onset + grid_seconds
        quantized.append(NoteEvent(onset, offset, note.pitch_midi, note.confidence))
    return quantized


def _unwrap_output(
    stacked: np.ndarray, original_length: int, n_overlapping_frames: int, hop_size: int
) -> np.ndarray:
    """Mismo algebra que `basic_pitch.inference.unwrap_output`: recorta el
    solape de cada ventana y trunca al numero de frames que la duracion REAL
    del audio (sin el relleno de ceros del ultimo tramo) produce."""
    n_olap = int(0.5 * n_overlapping_frames)
    trimmed = stacked[:, n_olap:-n_olap, :] if n_olap > 0 else stacked
    windows, frames, bins = trimmed.shape
    flattened = trimmed.reshape(windows * frames, bins)
    n_expected_windows = original_length / hop_size
    n_frames_per_window = (AUDIO_WINDOW_LENGTH * ANNOTATIONS_FPS) - n_overlapping_frames
    return flattened[: int(n_expected_windows * n_frames_per_window), :]


class MusicTranscriptionEngine:
    """ONNX Basic Pitch, con el MISMO patron de sesion cacheada que
    `OnnxStemSeparator` (clave `device::model_id`, `wrap_onnx_error`) aunque
    esto no separa stems -- transcribe UN stem ya decodido a eventos de nota."""

    pack_name = "music-transcription"
    default_model_id: ClassVar[str] = "basic_pitch"
    session_cache_size: ClassVar[int] = 1

    def __init__(self, settings: Any, gpu_coordinator: Any) -> None:
        self.settings = settings
        self.gpu_coordinator = gpu_coordinator
        self._session_cache: OrderedDict[str, Any] = OrderedDict()
        self._session_lock = threading.Lock()

    def available(self) -> bool:
        return self.settings.music_transcription_available()

    def transcribe_file(
        self, input_wav: Path, device: str, model_id: str | None = None
    ) -> list[NoteEvent]:
        self._require_available()
        audio = load_audio(input_wav, AUDIO_SAMPLE_RATE)  # [samples, canales] @ 22050
        return self.transcribe(_to_mono(audio), device, model_id)

    def transcribe(
        self, mono_audio: np.ndarray, device: str, model_id: str | None = None
    ) -> list[NoteEvent]:
        self._require_available()
        session = self._get_session(device, model_id or self.default_model_id)
        note_probs, onset_probs = self._run_windowed(session, mono_audio)
        return notes_from_activations(note_probs, onset_probs)

    def _require_available(self) -> None:
        if not self.available():
            from app.services.missing_pack import missing_pack_message

            raise RuntimeError(missing_pack_message(self.pack_name))

    def _run_windowed(self, session: Any, mono: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        overlap_len = DEFAULT_OVERLAPPING_FRAMES * FFT_HOP
        hop_size = AUDIO_N_SAMPLES - overlap_len
        original_length = int(mono.shape[0])
        padded = np.concatenate(
            [np.zeros(overlap_len // 2, dtype=np.float32), mono.astype(np.float32)]
        )
        note_windows: list[np.ndarray] = []
        onset_windows: list[np.ndarray] = []
        for start in range(0, padded.shape[0], hop_size):
            window = padded[start : start + AUDIO_N_SAMPLES]
            if window.shape[0] < AUDIO_N_SAMPLES:
                window = np.pad(window, (0, AUDIO_N_SAMPLES - window.shape[0]))
            batch = window.reshape(1, AUDIO_N_SAMPLES, 1).astype(np.float32)
            note, onset = self._infer(session, batch)
            note_windows.append(note[0])
            onset_windows.append(onset[0])
        note_stitched = _unwrap_output(
            np.stack(note_windows), original_length, DEFAULT_OVERLAPPING_FRAMES, hop_size
        )
        onset_stitched = _unwrap_output(
            np.stack(onset_windows), original_length, DEFAULT_OVERLAPPING_FRAMES, hop_size
        )
        return note_stitched, onset_stitched

    def _infer(self, session: Any, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        note, onset, _contour = session.run(
            [_ONNX_OUTPUT_NOTE, _ONNX_OUTPUT_ONSET, _ONNX_OUTPUT_CONTOUR],
            {_ONNX_INPUT_NAME: batch},
        )
        return np.asarray(note), np.asarray(onset)

    # -- sesion: misma forma que separator_base.OnnxStemSeparator -----------

    def _get_session(self, device: str, model_id: str) -> Any:
        self.gpu_coordinator.acquire(device, self)
        key = self._cache_key(device, model_id)
        return get_cached_session(
            self._session_cache,
            self._session_lock,
            key,
            lambda: self._create_session(device, model_id),
            f"Failed to load music transcription model on device {device!r}",
            cache_size=self.session_cache_size,
        )

    def release_device(self, device: str) -> None:
        prefix = f"{device}::"
        with self._session_lock:
            for key in [k for k in self._session_cache if k.startswith(prefix)]:
                self._session_cache.pop(key, None)

    @staticmethod
    def _cache_key(device: str, model_id: str) -> str:
        return f"{device}::{model_id}"

    def _create_session(self, device: str, model_id: str) -> Any:
        # Seam monkeypatcheable por nombre, como el resto de los motores ONNX.
        from app.services import ep_registry

        return ep_registry.create_session(
            str(self.settings.music_transcription_model_path), device, self.settings
        )
