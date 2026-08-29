from __future__ import annotations

from pathlib import Path

from app.services.engines.music_transcription import NoteEvent
from app.services.engines.tab_writer import (
    BASS_STANDARD_TUNING,
    GUITAR_STANDARD_TUNING,
    assign_frets,
    build_tab_text,
    write_tab,
)

# ---------------------------------------------------------------------------
# F3a - tab_writer: asignacion DP cuerda/traste (guitarra 6 cuerdas EADGBE,
# bajo 4 cuerdas EADG, afinacion estandar unicamente) + render de texto plano.
# ---------------------------------------------------------------------------


def note(pitch_midi: int) -> NoteEvent:
    return NoteEvent(onset_time=0.0, offset_time=0.1, pitch_midi=pitch_midi, confidence=1.0)


def test_a_single_note_lands_on_its_only_candidate_string() -> None:
    # E2 (40) solo entra en la cuerda mas grave, al aire.
    assert assign_frets([note(40)], GUITAR_STANDARD_TUNING) == [(0, 0)]


def test_dp_prefers_the_open_string_over_a_far_fret_to_minimize_total_movement() -> None:
    # E2 - A2 - E2: tocar el A2 en fret5 de la cuerda grave costaria 5+5=10 de
    # salto total; tocarlo al aire en la cuerda de A cuesta 0+0=0. La DP tiene
    # que elegir la segunda, aunque la primera nota "ya estaba" en esa cuerda.
    notes = [note(40), note(45), note(40)]

    assignment = assign_frets(notes, GUITAR_STANDARD_TUNING)

    assert assignment == [(0, 0), (1, 0), (0, 0)]


def test_dp_minimizes_total_fret_jump_over_a_greedy_per_note_choice() -> None:
    # Una nota que SI tiene varias opciones cercanas: la DP mira el costo
    # acumulado, no solo el salto inmediato.
    notes = [note(45), note(45), note(52)]  # A2(x2), D3+2=E3

    assignment = assign_frets(notes, GUITAR_STANDARD_TUNING)

    # Las dos A2 seguidas quedan en la MISMA cuerda/traste -- cero salto entre
    # ellas es siempre optimo salvo que la tercera nota lo compense, y aca no
    # lo compensa (D3 fret2 o E2 fret12 son ambas alcanzables desde (1,0)).
    assert assignment[0] == assignment[1]


def test_bass_uses_its_own_four_string_tuning() -> None:
    # E1 (28): al aire en la cuerda mas grave del bajo.
    assert assign_frets([note(28)], BASS_STANDARD_TUNING) == [(0, 0)]
    # E1 no existe en la afinacion de guitarra (la mas grave es E2=40).
    assert assign_frets([note(28)], GUITAR_STANDARD_TUNING) is None


def test_a_pitch_outside_the_instrument_range_returns_none() -> None:
    too_low = note(10)
    assert assign_frets([too_low], GUITAR_STANDARD_TUNING) is None
    assert assign_frets([too_low], BASS_STANDARD_TUNING) is None


def test_empty_notes_returns_an_empty_assignment() -> None:
    assert assign_frets([], GUITAR_STANDARD_TUNING) == []


def test_build_tab_text_rejects_unsupported_instruments_cleanly() -> None:
    assert build_tab_text([note(60)], "piano") is None


def test_build_tab_text_rejects_notes_outside_the_instrument_range() -> None:
    assert build_tab_text([note(10)], "guitar") is None


def test_build_tab_text_renders_the_open_low_e_on_the_bottom_row() -> None:
    text = build_tab_text([note(40)], "guitar")

    assert text is not None
    assert "afinacion estandar E-B-G-D-A-E" in text
    lines = text.strip("\n").splitlines()
    assert lines[-1] == "E|-0-|"  # cuerda mas grave, al aire, ultima fila
    assert len(lines) == 2 + 6  # encabezado + linea en blanco + 6 cuerdas


def test_write_tab_writes_the_file_and_returns_true(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "song.tab.txt"

    written = write_tab([note(40)], destination, "guitar")

    assert written is True
    assert destination.exists()
    assert "E|-0-|" in destination.read_text(encoding="utf-8")


def test_write_tab_returns_false_and_writes_nothing_for_an_unsupported_instrument(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "song.tab.txt"

    written = write_tab([note(60)], destination, "vocals")

    assert written is False
    assert not destination.exists()
