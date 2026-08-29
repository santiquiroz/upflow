from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.config import Settings
from app.services.engines.music_transcription import (
    ANNOTATION_HOP,
    AUDIO_N_SAMPLES,
    DEFAULT_GRID_SECONDS,
    MIDI_BASE_NOTE,
    MIN_NOTE_FRAMES,
    MusicTranscriptionEngine,
    NoteEvent,
    notes_from_activations,
    quantize_notes,
)
from app.services.gpu_session_coordinator import GpuSessionCoordinator

# ---------------------------------------------------------------------------
# F3a - music_transcription: segmentacion pura (notes_from_activations),
# cuantizacion, y el motor ONNX con el patron de sesion cacheada de
# separator_base (clave device::model_id, `_create_session` monkeypatcheable).
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        RUNTIME_DIR=str(tmp_path / "runtime"),
        # Aislado: sin esto el default relativo resuelve contra la raiz REAL
        # del proyecto (resolve_against_project_root), no contra tmp_path.
        MUSIC_TRANSCRIPTION_MODEL=str(tmp_path / "no-model" / "nmp.onnx"),
        _env_file=None,
        **overrides,
    )


def make_settings_with_model(tmp_path: Path) -> Settings:
    model = tmp_path / "vendor" / "music-transcription" / "nmp.onnx"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(b"fake-onnx")
    return Settings(RUNTIME_DIR=str(tmp_path / "runtime"), MUSIC_TRANSCRIPTION_MODEL=str(model), _env_file=None)


# ---------------------------------------------------------------------------
# notes_from_activations: matriz de activacion conocida -> eventos exactos
# ---------------------------------------------------------------------------


def test_a_contiguous_active_span_becomes_one_note() -> None:
    note_probs = np.array([[0.5], [0.4], [0.35], [0.1], [0.6]])
    onset_probs = np.array([[0.9], [0.1], [0.1], [0.0], [0.6]])

    events = notes_from_activations(note_probs, onset_probs, minimum_note_frames=1)

    assert events == [
        NoteEvent(
            onset_time=0.0,
            offset_time=pytest.approx(3 * ANNOTATION_HOP),
            pitch_midi=MIDI_BASE_NOTE,
            confidence=pytest.approx((0.5 + 0.4 + 0.35) / 3),
        ),
        NoteEvent(
            onset_time=pytest.approx(4 * ANNOTATION_HOP),
            offset_time=pytest.approx(5 * ANNOTATION_HOP),
            pitch_midi=MIDI_BASE_NOTE,
            confidence=pytest.approx(0.6),
        ),
    ]


def test_a_fresh_onset_mid_span_splits_into_two_notes() -> None:
    note_probs = np.array([[0.5], [0.5], [0.5], [0.5]])
    onset_probs = np.array([[0.9], [0.1], [0.9], [0.1]])

    events = notes_from_activations(note_probs, onset_probs, minimum_note_frames=1)

    assert events == [
        NoteEvent(0.0, pytest.approx(2 * ANNOTATION_HOP), MIDI_BASE_NOTE, pytest.approx(0.5)),
        NoteEvent(
            pytest.approx(2 * ANNOTATION_HOP),
            pytest.approx(4 * ANNOTATION_HOP),
            MIDI_BASE_NOTE,
            pytest.approx(0.5),
        ),
    ]


def test_pitch_bin_maps_to_midi_starting_at_a0() -> None:
    note_probs = np.zeros((3, 4))
    note_probs[:, 2] = 0.8
    onset_probs = np.zeros((3, 4))
    onset_probs[0, 2] = 0.9

    events = notes_from_activations(note_probs, onset_probs, minimum_note_frames=1)

    assert len(events) == 1
    assert events[0].pitch_midi == MIDI_BASE_NOTE + 2


def test_notes_shorter_than_the_minimum_length_are_dropped_by_default() -> None:
    # 5 frames << MIN_NOTE_FRAMES (~11 a 86 fps): se descarta con los umbrales
    # por defecto, sin pasar minimum_note_frames a mano.
    frames = 5
    assert frames < MIN_NOTE_FRAMES
    note_probs = np.zeros((frames, 1))
    note_probs[:, 0] = 0.8
    onset_probs = np.zeros((frames, 1))
    onset_probs[0, 0] = 0.9

    assert notes_from_activations(note_probs, onset_probs) == []


def test_notes_at_or_above_the_minimum_length_survive_the_default_filter() -> None:
    frames = MIN_NOTE_FRAMES + 4
    note_probs = np.zeros((frames, 1))
    note_probs[:, 0] = 0.8
    onset_probs = np.zeros((frames, 1))
    onset_probs[0, 0] = 0.9

    events = notes_from_activations(note_probs, onset_probs)

    assert len(events) == 1
    assert events[0].onset_time == 0.0
    assert events[0].offset_time == pytest.approx(frames * ANNOTATION_HOP)


def test_events_are_sorted_by_onset_then_pitch() -> None:
    note_probs = np.zeros((3, 2))
    note_probs[1:, 0] = 0.8  # pitch bin 0 activo desde frame 1
    note_probs[0:, 1] = 0.8  # pitch bin 1 activo desde frame 0
    onset_probs = np.zeros((3, 2))
    onset_probs[1, 0] = 0.9
    onset_probs[0, 1] = 0.9

    events = notes_from_activations(note_probs, onset_probs, minimum_note_frames=1)

    assert [e.pitch_midi for e in events] == [MIDI_BASE_NOTE + 1, MIDI_BASE_NOTE]


def test_a_note_still_active_at_the_end_of_the_matrix_closes_at_the_last_frame() -> None:
    note_probs = np.array([[0.5], [0.5], [0.5]])
    onset_probs = np.array([[0.9], [0.0], [0.0]])

    events = notes_from_activations(note_probs, onset_probs, minimum_note_frames=1)

    assert len(events) == 1
    assert events[0].offset_time == pytest.approx(3 * ANNOTATION_HOP)


# ---------------------------------------------------------------------------
# quantize_notes
# ---------------------------------------------------------------------------


def test_quantize_notes_rounds_to_the_nearest_grid_multiple() -> None:
    grid = DEFAULT_GRID_SECONDS
    note = NoteEvent(onset_time=0.03, offset_time=0.19, pitch_midi=60, confidence=1.0)

    quantized = quantize_notes([note], grid)

    assert quantized == [NoteEvent(0.0, pytest.approx(2 * grid), 60, 1.0)]


def test_quantize_notes_never_collapses_a_note_to_zero_duration() -> None:
    grid = DEFAULT_GRID_SECONDS
    note = NoteEvent(onset_time=0.1, offset_time=0.11, pitch_midi=60, confidence=1.0)

    quantized = quantize_notes([note], grid)

    assert quantized[0].offset_time > quantized[0].onset_time
    assert quantized[0].offset_time - quantized[0].onset_time == pytest.approx(grid)


# ---------------------------------------------------------------------------
# Motor: sesion cacheada (mismo patron que separator_base.OnnxStemSeparator)
# ---------------------------------------------------------------------------


def test_available_follows_the_model_file_on_disk(tmp_path: Path) -> None:
    coordinator = GpuSessionCoordinator()
    without_model = MusicTranscriptionEngine(make_settings(tmp_path / "a"), coordinator)
    with_model = MusicTranscriptionEngine(make_settings_with_model(tmp_path / "b"), coordinator)

    assert without_model.available() is False
    assert with_model.available() is True


def test_transcribe_when_unavailable_raises_actionable_error(tmp_path: Path) -> None:
    engine = MusicTranscriptionEngine(make_settings(tmp_path), GpuSessionCoordinator())

    with pytest.raises(RuntimeError, match="transcripcion"):
        engine.transcribe(np.zeros(1000, dtype=np.float32), device="cpu")


def test_get_session_calls_coordinator_acquire_before_creating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator = GpuSessionCoordinator()
    engine = MusicTranscriptionEngine(make_settings_with_model(tmp_path), coordinator)
    calls: list[tuple[str, Any]] = []
    monkeypatch.setattr(coordinator, "acquire", lambda device, owner: calls.append((device, owner)))
    monkeypatch.setattr(engine, "_create_session", lambda device, model_id: "fake-session")

    engine._get_session("dml:0", "basic_pitch")

    assert calls == [("dml:0", engine)]


def test_get_session_caches_by_device_and_model_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = MusicTranscriptionEngine(make_settings_with_model(tmp_path), GpuSessionCoordinator())
    built: list[tuple[str, str]] = []

    def fake_create(device: str, model_id: str) -> str:
        built.append((device, model_id))
        return f"session-{device}-{model_id}"

    monkeypatch.setattr(engine, "_create_session", fake_create)

    first = engine._get_session("cpu", "basic_pitch")
    second = engine._get_session("cpu", "basic_pitch")

    assert first == second == "session-cpu-basic_pitch"
    assert built == [("cpu", "basic_pitch")]


def test_release_device_clears_only_that_devices_cached_sessions(tmp_path: Path) -> None:
    engine = MusicTranscriptionEngine(make_settings_with_model(tmp_path), GpuSessionCoordinator())
    engine._session_cache["dml:0::basic_pitch"] = "a"
    engine._session_cache["dml:1::basic_pitch"] = "b"

    engine.release_device("dml:0")

    assert "dml:0::basic_pitch" not in engine._session_cache
    assert "dml:1::basic_pitch" in engine._session_cache


# ---------------------------------------------------------------------------
# Motor: sesion ONNX fakeada de punta a punta -- ventaneo real + unwrap real
# sobre audio de ~2s (una ventana entera), eventos MIDI exactos.
# ---------------------------------------------------------------------------


class FakeTranscriptionSession:
    """Devuelve SIEMPRE la misma activacion: un pitch fijo activo en TODOS
    los frames de la ventana, con un pico de onset solo en el frame 0 (que el
    recorte del solape termina descartando -- ver el modulo bajo prueba)."""

    def __init__(self, pitch_bin: int) -> None:
        frames_per_window = 172  # AUDIO_WINDOW_LENGTH(2) * ANNOTATIONS_FPS(86)
        note = np.zeros((1, frames_per_window, 88), dtype=np.float32)
        note[0, :, pitch_bin] = 1.0
        onset = np.zeros((1, frames_per_window, 88), dtype=np.float32)
        onset[0, 0, pitch_bin] = 1.0
        contour = np.zeros((1, frames_per_window, 264), dtype=np.float32)
        self._note = note
        self._onset = onset
        self._contour = contour
        self.calls = 0

    def run(self, output_names: list[str], feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls += 1
        return [self._note, self._onset, self._contour]


def test_transcribe_runs_the_fake_session_and_derives_one_sustained_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine = MusicTranscriptionEngine(make_settings_with_model(tmp_path), GpuSessionCoordinator())
    pitch_bin = 39  # MIDI 60 (C4)
    session = FakeTranscriptionSession(pitch_bin)
    monkeypatch.setattr(engine, "_create_session", lambda device, model_id: session)
    mono = np.zeros(AUDIO_N_SAMPLES, dtype=np.float32)

    notes = engine.transcribe(mono, device="cpu")

    # Dos ventanas para este largo de audio (43844 muestras): el motor tiene
    # que haber llamado la sesion mas de una vez.
    assert session.calls == 2
    assert notes == [
        NoteEvent(
            onset_time=0.0,
            offset_time=pytest.approx(2.0),
            pitch_midi=60,
            confidence=pytest.approx(1.0),
        )
    ]
