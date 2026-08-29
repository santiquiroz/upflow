"""Standard MIDI File (formato 0), escrito a mano -- cero deps (nada de
`mido`/`pretty_midi`, decision #9 del contrato F3a).

Un solo track: cabecera `MThd` + un chunk `MTrk` con un evento Set Tempo fijo
(120 BPM, ver `music_transcription.DEFAULT_TEMPO_BPM` -- no hay deteccion de
tempo real), un par Note On/Note Off por nota codificado con delta-time en
variable-length quantity (VLQ), y el End of Track final. Referencia del
formato binario: "Standard MIDI Files 1.0" (MMA/AMEI), spec publica.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from app.services.engines.music_transcription import DEFAULT_TEMPO_BPM, NoteEvent

TICKS_PER_QUARTER = 480
DEFAULT_VELOCITY = 96
_NOTE_ON = 0x90
_NOTE_OFF = 0x80
_META_EVENT = 0xFF
_META_SET_TEMPO = 0x51
_META_END_OF_TRACK = 0x2F


def _variable_length_quantity(value: int) -> bytes:
    if value < 0:
        raise ValueError(f"VLQ no admite valores negativos: {value}")
    chunks = [value & 0x7F]
    value >>= 7
    while value:
        chunks.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(chunks))


def seconds_to_ticks(
    seconds: float, ticks_per_quarter: int = TICKS_PER_QUARTER, tempo_bpm: float = DEFAULT_TEMPO_BPM
) -> int:
    ticks_per_second = ticks_per_quarter * tempo_bpm / 60.0
    return round(seconds * ticks_per_second)


def _meta_event(event_type: int, data: bytes) -> bytes:
    return bytes([_META_EVENT, event_type]) + _variable_length_quantity(len(data)) + data


def build_midi_bytes(
    notes: Sequence[NoteEvent],
    *,
    ticks_per_quarter: int = TICKS_PER_QUARTER,
    tempo_bpm: float = DEFAULT_TEMPO_BPM,
    velocity: int = DEFAULT_VELOCITY,
    channel: int = 0,
) -> bytes:
    """Un evento Note On y uno Note Off por nota. Cuando dos notas quedan en el
    MISMO tick, los Note Off van primero (prioridad 0 contra 1 de Note On):
    corta la nota vieja antes de prender la nueva en vez de dejarlas
    superpuestas un instante, y hace el orden reproducible byte a byte."""
    events: list[tuple[int, int, bytes]] = []
    for note in notes:
        onset_tick = seconds_to_ticks(note.onset_time, ticks_per_quarter, tempo_bpm)
        offset_tick = seconds_to_ticks(note.offset_time, ticks_per_quarter, tempo_bpm)
        if offset_tick <= onset_tick:
            offset_tick = onset_tick + 1
        pitch = max(0, min(127, note.pitch_midi))
        events.append((onset_tick, 1, bytes([_NOTE_ON | channel, pitch, velocity])))
        events.append((offset_tick, 0, bytes([_NOTE_OFF | channel, pitch, 0])))
    events.sort(key=lambda item: (item[0], item[1], item[2]))

    microseconds_per_quarter = round(60_000_000 / tempo_bpm)
    track = bytearray()
    track += _variable_length_quantity(0)
    track += _meta_event(_META_SET_TEMPO, microseconds_per_quarter.to_bytes(3, "big"))
    previous_tick = 0
    for tick, _priority, event_bytes in events:
        track += _variable_length_quantity(tick - previous_tick)
        track += event_bytes
        previous_tick = tick
    track += _variable_length_quantity(0)
    track += _meta_event(_META_END_OF_TRACK, b"")

    header = (
        b"MThd"
        + (6).to_bytes(4, "big")
        + (0).to_bytes(2, "big")  # formato 0: un solo track
        + (1).to_bytes(2, "big")  # ntracks
        + ticks_per_quarter.to_bytes(2, "big")
    )
    track_chunk = b"MTrk" + len(track).to_bytes(4, "big") + bytes(track)
    return header + track_chunk


def write_midi(notes: Sequence[NoteEvent], destination: Path, **kwargs: object) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(build_midi_bytes(notes, **kwargs))
