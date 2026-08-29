from __future__ import annotations

from pathlib import Path

from app.services.engines.midi_writer import (
    TICKS_PER_QUARTER,
    _variable_length_quantity,
    build_midi_bytes,
    seconds_to_ticks,
    write_midi,
)
from app.services.engines.music_transcription import NoteEvent

# ---------------------------------------------------------------------------
# F3a - midi_writer: Standard MIDI File (formato 0) escrito a mano. Bytes
# verificados a mano contra la spec publica de Standard MIDI Files 1.0.
# ---------------------------------------------------------------------------


def test_vlq_encodes_small_values_as_a_single_byte() -> None:
    assert _variable_length_quantity(0) == b"\x00"
    assert _variable_length_quantity(0x40) == b"\x40"
    assert _variable_length_quantity(0x7F) == b"\x7f"


def test_vlq_encodes_128_with_the_documented_two_byte_form() -> None:
    # Ejemplo canonico de la spec de MIDI: 128 -> 0x81 0x00.
    assert _variable_length_quantity(128) == b"\x81\x00"


def test_vlq_encodes_480_ticks() -> None:
    assert _variable_length_quantity(480) == b"\x83\x60"


def test_seconds_to_ticks_at_default_tempo() -> None:
    # 120 BPM, 480 ticks/negra -> 960 ticks/segundo.
    assert seconds_to_ticks(0.0) == 0
    assert seconds_to_ticks(0.5, TICKS_PER_QUARTER, 120.0) == 480
    assert seconds_to_ticks(1.0, TICKS_PER_QUARTER, 120.0) == 960


def test_build_midi_bytes_matches_the_hand_verified_single_note_file() -> None:
    notes = [NoteEvent(onset_time=0.0, offset_time=0.5, pitch_midi=60, confidence=1.0)]

    data = build_midi_bytes(notes)

    header = (
        b"MThd"
        + b"\x00\x00\x00\x06"
        + b"\x00\x00"  # formato 0
        + b"\x00\x01"  # 1 track
        + b"\x01\xe0"  # 480 ticks/negra
    )
    track = (
        b"\x00" + b"\xff\x51\x03" + b"\x07\xa1\x20"  # delta 0, Set Tempo 500000us
        + b"\x00" + b"\x90\x3c\x60"  # delta 0, Note On C4 vel 96
        + b"\x83\x60" + b"\x80\x3c\x00"  # delta 480, Note Off C4
        + b"\x00" + b"\xff\x2f\x00"  # delta 0, End of Track
    )
    expected = header + b"MTrk" + len(track).to_bytes(4, "big") + track
    assert data == expected


def test_build_midi_bytes_on_empty_notes_is_a_valid_header_plus_empty_track() -> None:
    data = build_midi_bytes([])

    track = b"\x00\xff\x51\x03\x07\xa1\x20" + b"\x00\xff\x2f\x00"
    assert data == (
        b"MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0"
        + b"MTrk"
        + len(track).to_bytes(4, "big")
        + track
    )


def test_a_note_off_wins_the_tie_at_the_same_tick_as_a_note_on() -> None:
    # Dos notas consecutivas de la MISMA altura, sin hueco: el Note Off de la
    # primera y el Note On de la segunda caen en el mismo tick. El corte tiene
    # que ir primero para no superponer las dos notas un instante.
    notes = [
        NoteEvent(onset_time=0.0, offset_time=0.5, pitch_midi=60, confidence=1.0),
        NoteEvent(onset_time=0.5, offset_time=1.0, pitch_midi=60, confidence=1.0),
    ]

    data = build_midi_bytes(notes)

    note_off_index = data.index(b"\x80\x3c\x00")
    note_on_second_index = data.index(b"\x90\x3c\x60", note_off_index)
    assert note_off_index < note_on_second_index


def test_pitch_is_clamped_into_the_valid_midi_range() -> None:
    notes = [NoteEvent(onset_time=0.0, offset_time=0.1, pitch_midi=200, confidence=1.0)]

    data = build_midi_bytes(notes)

    assert b"\x90\x7f" in data  # 127, no 200


def test_write_midi_creates_parent_directories_and_writes_the_same_bytes(
    tmp_path: Path,
) -> None:
    notes = [NoteEvent(onset_time=0.0, offset_time=0.5, pitch_midi=60, confidence=1.0)]
    destination = tmp_path / "nested" / "song.mid"

    write_midi(notes, destination)

    assert destination.read_bytes() == build_midi_bytes(notes)
