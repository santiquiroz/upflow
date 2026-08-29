"""MusicXML minimo, escrito a mano -- cero deps (nada de `music21`, decision
#9 del contrato F3a). Un solo `<part>` por stem (contrato F3a): quien quiera
varios stems en una partitura los abre por separado y los combina en
MuseScore/Guitar Pro, que es exactamente el framing de "borrador editable".

Cuantizacion: reusa `music_transcription.quantize_notes` (misma grilla fija
que `midi_writer`, sin deteccion de tempo real) para que los onsets caigan en
un multiplo entero de la grilla, convertible a duraciones enteras en
`<divisions>`. Notas simultaneas (mismo slot de grilla) se funden en un
acorde (`<chord/>`); una nota que cruza un limite de compas se parte en dos
`<note>` ligadas con `<tie>`/`<notations><tied>` -- lo minimo para que la
duracion total de cada compas cierre en 4/4 sin perder la nota.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

from app.services.engines.music_transcription import DEFAULT_GRID_SECONDS, NoteEvent, quantize_notes

SLOTS_PER_MEASURE = 16  # 4/4 a grilla de semicorchea (4 negras * 4 semicorcheas)

_STEP_ALTER_BY_PITCH_CLASS: dict[int, tuple[str, int]] = {
    0: ("C", 0), 1: ("C", 1), 2: ("D", 0), 3: ("D", 1), 4: ("E", 0), 5: ("F", 0),
    6: ("F", 1), 7: ("G", 0), 8: ("G", 1), 9: ("A", 0), 10: ("A", 1), 11: ("B", 0),
}

_TYPE_BY_SLOTS = {1: "16th", 2: "eighth", 4: "quarter", 8: "half", 16: "whole"}


def _pitch_to_step_alter_octave(pitch_midi: int) -> tuple[str, int, int]:
    step, alter = _STEP_ALTER_BY_PITCH_CLASS[pitch_midi % 12]
    octave = pitch_midi // 12 - 1
    return step, alter, octave


def _timeline_events(
    notes: Sequence[NoteEvent], grid_seconds: float
) -> list[tuple[int, int, list[int]]]:
    """(onset_slot, duration_slots, pitches) por evento, cubriendo TODA la
    linea de tiempo sin huecos: los huecos entre notas se rellenan con
    silencio (`pitches=[]`) y dos notas con el mismo onset se funden en un
    acorde. Las duraciones se recortan para no superponerse con el siguiente
    onset -- una nota "larga" que en verdad estaba sostenida bajo otra ya
    perdio esa informacion en la cuantizacion, y forzar un acorde parcial
    seria peor que recortarla."""
    if not notes:
        return []
    quantized = quantize_notes(list(notes), grid_seconds)
    by_onset: dict[int, list[tuple[int, int]]] = {}
    for note in quantized:
        onset_slot = round(note.onset_time / grid_seconds)
        duration_slots = max(1, round((note.offset_time - note.onset_time) / grid_seconds))
        by_onset.setdefault(onset_slot, []).append((duration_slots, note.pitch_midi))

    onsets = sorted(by_onset)
    events: list[tuple[int, int, list[int]]] = []
    cursor = 0
    for index, onset_slot in enumerate(onsets):
        if onset_slot > cursor:
            events.append((cursor, onset_slot - cursor, []))
        group = by_onset[onset_slot]
        duration = min(duration for duration, _ in group)
        if index + 1 < len(onsets):
            duration = min(duration, onsets[index + 1] - onset_slot)
        duration = max(1, duration)
        pitches = sorted({pitch for _, pitch in group})
        events.append((onset_slot, duration, pitches))
        cursor = onset_slot + duration
    return events


def _new_measure(number: int, *, with_attributes: bool = False) -> ET.Element:
    measure = ET.Element("measure", number=str(number))
    if with_attributes:
        attributes = ET.SubElement(measure, "attributes")
        ET.SubElement(attributes, "divisions").text = "1"
        key = ET.SubElement(attributes, "key")
        ET.SubElement(key, "fifths").text = "0"
        time = ET.SubElement(attributes, "time")
        ET.SubElement(time, "beats").text = "4"
        ET.SubElement(time, "beat-type").text = "4"
        clef = ET.SubElement(attributes, "clef")
        ET.SubElement(clef, "sign").text = "G"
        ET.SubElement(clef, "line").text = "2"
    return measure


def _rest_element(duration_slots: int) -> ET.Element:
    note = ET.Element("note")
    ET.SubElement(note, "rest")
    ET.SubElement(note, "duration").text = str(duration_slots)
    note_type = _TYPE_BY_SLOTS.get(duration_slots)
    if note_type is not None:
        ET.SubElement(note, "type").text = note_type
    return note


def _note_element(
    pitch_midi: int, duration_slots: int, *, chord: bool, tie_start: bool, tie_stop: bool
) -> ET.Element:
    note = ET.Element("note")
    if chord:
        ET.SubElement(note, "chord")
    step, alter, octave = _pitch_to_step_alter_octave(pitch_midi)
    pitch_el = ET.SubElement(note, "pitch")
    ET.SubElement(pitch_el, "step").text = step
    if alter:
        ET.SubElement(pitch_el, "alter").text = str(alter)
    ET.SubElement(pitch_el, "octave").text = str(octave)
    ET.SubElement(note, "duration").text = str(duration_slots)
    if tie_stop:
        ET.SubElement(note, "tie", type="stop")
    if tie_start:
        ET.SubElement(note, "tie", type="start")
    note_type = _TYPE_BY_SLOTS.get(duration_slots)
    if note_type is not None:
        ET.SubElement(note, "type").text = note_type
    if tie_stop or tie_start:
        notations = ET.SubElement(note, "notations")
        if tie_stop:
            ET.SubElement(notations, "tied", type="stop")
        if tie_start:
            ET.SubElement(notations, "tied", type="start")
    return note


def _build_part(events: list[tuple[int, int, list[int]]]) -> ET.Element:
    part = ET.Element("part", id="P1")
    measure_index = 1
    measure = _new_measure(measure_index, with_attributes=True)
    part.append(measure)
    position_in_measure = 0
    for _onset_slot, duration_slots, pitches in events:
        remaining = duration_slots
        is_rest = not pitches
        chunk_index = 0
        while remaining > 0:
            if position_in_measure == SLOTS_PER_MEASURE:
                measure_index += 1
                measure = _new_measure(measure_index)
                part.append(measure)
                position_in_measure = 0
            available = SLOTS_PER_MEASURE - position_in_measure
            chunk = min(remaining, available)
            is_first_chunk = chunk_index == 0
            is_last_chunk = chunk == remaining
            if is_rest:
                measure.append(_rest_element(chunk))
            else:
                for pitch_index, pitch in enumerate(pitches):
                    measure.append(
                        _note_element(
                            pitch,
                            chunk,
                            chord=pitch_index > 0,
                            tie_start=not is_last_chunk,
                            tie_stop=not is_first_chunk,
                        )
                    )
            position_in_measure += chunk
            remaining -= chunk
            chunk_index += 1
    return part


def build_musicxml_string(
    notes: Sequence[NoteEvent],
    *,
    part_name: str = "Music",
    grid_seconds: float = DEFAULT_GRID_SECONDS,
) -> str:
    events = _timeline_events(notes, grid_seconds)
    root = ET.Element("score-partwise", version="3.1")
    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = part_name
    root.append(_build_part(events))
    body = ET.tostring(root, encoding="unicode")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n'
        f"{body}\n"
    )


def write_musicxml(notes: Sequence[NoteEvent], destination: Path, **kwargs: object) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_musicxml_string(notes, **kwargs), encoding="utf-8")
