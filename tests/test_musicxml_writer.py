from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from app.services.engines.music_transcription import DEFAULT_GRID_SECONDS, NoteEvent
from app.services.engines.musicxml_writer import build_musicxml_string, write_musicxml

# ---------------------------------------------------------------------------
# F3a - musicxml_writer: MusicXML minimo, un part por stem, cuantizado a la
# grilla fija de music_transcription (sin deteccion de tempo). Se verifica
# PARSEANDO con xml.etree, no comparando strings.
# ---------------------------------------------------------------------------

GRID = DEFAULT_GRID_SECONDS  # semicorchea a 120 BPM = 0.125 s


def parse(notes: list[NoteEvent]) -> ET.Element:
    xml_text = build_musicxml_string(notes)
    assert xml_text.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert "<!DOCTYPE score-partwise" in xml_text
    return ET.fromstring(xml_text)


def measures(root: ET.Element) -> list[ET.Element]:
    part = root.find("part")
    assert part is not None
    return part.findall("measure")


def test_root_declares_the_single_part(parse=parse) -> None:
    root = parse([NoteEvent(0.0, 4 * GRID, 60, 1.0)])

    score_part = root.find("part-list/score-part")
    assert score_part is not None
    assert score_part.get("id") == "P1"
    assert root.find("part").get("id") == "P1"


def test_a_single_quarter_note_lands_in_one_measure() -> None:
    root = parse([NoteEvent(0.0, 4 * GRID, 60, 1.0)])

    measure_list = measures(root)
    assert len(measure_list) == 1
    notes = measure_list[0].findall("note")
    assert len(notes) == 1
    note = notes[0]
    assert note.find("pitch/step").text == "C"
    assert note.find("pitch/octave").text == "4"
    assert note.find("pitch/alter") is None
    assert note.find("duration").text == "4"
    assert note.find("type").text == "quarter"
    assert note.find("chord") is None
    assert note.find("tie") is None


def test_sharps_are_marked_with_alter_one() -> None:
    root = parse([NoteEvent(0.0, 1 * GRID, 61, 1.0)])  # C#4

    note = measures(root)[0].find("note")
    assert note.find("pitch/step").text == "C"
    assert note.find("pitch/alter").text == "1"
    assert note.find("pitch/octave").text == "4"


def test_two_notes_sharing_an_onset_become_a_chord() -> None:
    root = parse(
        [
            NoteEvent(0.0, 2 * GRID, 60, 1.0),
            NoteEvent(0.0, 4 * GRID, 64, 1.0),
        ]
    )

    notes = measures(root)[0].findall("note")
    assert len(notes) == 2
    assert notes[0].find("chord") is None
    assert notes[0].find("pitch/step").text == "C"
    assert notes[1].find("chord") is not None
    assert notes[1].find("pitch/step").text == "E"
    # La duracion del acorde se recorta a la nota MAS CORTA del grupo.
    assert notes[0].find("duration").text == "2"
    assert notes[1].find("duration").text == "2"


def test_a_gap_between_notes_is_filled_with_a_rest() -> None:
    root = parse([NoteEvent(2 * GRID, 4 * GRID, 60, 1.0)])

    notes = measures(root)[0].findall("note")
    assert len(notes) == 2
    rest, note = notes
    assert rest.find("rest") is not None
    assert rest.find("duration").text == "2"
    assert note.find("rest") is None
    assert note.find("duration").text == "2"


def test_a_note_crossing_a_measure_boundary_is_split_and_tied() -> None:
    # 16 slots por compas (4/4 a semicorchea): una nota que arranca en el slot
    # 14 y dura 4 cruza el limite del compas 1.
    root = parse([NoteEvent(14 * GRID, 18 * GRID, 60, 1.0)])

    measure_list = measures(root)
    assert len(measure_list) == 2

    first_measure_notes = measure_list[0].findall("note")
    rest, tied_note = first_measure_notes
    assert rest.find("rest") is not None
    assert rest.find("duration").text == "14"
    assert tied_note.find("duration").text == "2"
    assert tied_note.find("tie[@type='start']") is not None
    assert tied_note.find("tie[@type='stop']") is None
    assert tied_note.find("notations/tied[@type='start']") is not None

    second_measure_notes = measure_list[1].findall("note")
    assert len(second_measure_notes) == 1
    continuation = second_measure_notes[0]
    assert continuation.find("duration").text == "2"
    assert continuation.find("tie[@type='stop']") is not None
    assert continuation.find("tie[@type='start']") is None
    assert continuation.find("notations/tied[@type='stop']") is not None
    # Misma altura de un lado y del otro de la ligadura.
    assert continuation.find("pitch/step").text == tied_note.find("pitch/step").text
    assert continuation.find("pitch/octave").text == tied_note.find("pitch/octave").text


def test_no_notes_produces_a_single_attributes_only_measure() -> None:
    root = parse([])

    measure_list = measures(root)
    assert len(measure_list) == 1
    assert measure_list[0].find("attributes") is not None
    assert measure_list[0].findall("note") == []


def test_write_musicxml_creates_parent_directories(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "song.musicxml"

    write_musicxml([NoteEvent(0.0, 4 * GRID, 60, 1.0)], destination)

    assert destination.exists()
    ET.parse(destination)  # no explota
