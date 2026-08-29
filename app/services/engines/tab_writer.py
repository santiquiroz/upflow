"""Tab de texto plano para guitarra (6 cuerdas EADGBE) y bajo (4 cuerdas EADG),
afinacion estandar unicamente -- afinaciones alternativas fuera de alcance v1
(decision #9 del contrato F3a). Cualquier otro instrumento devuelve `None`:
el llamador decide no escribir el archivo.

Asignacion cuerda/traste: programacion dinamica que minimiza la SUMA de saltos
de traste entre notas consecutivas (`|fret[i] - fret[i-1]|`), para que la tab
resultante lea como una mano que se mueve poco por el mastil en vez de saltar
al traste mas grave posible en cada nota.

Formato de salida (texto plano, sin dependencias): un bloque de N lineas (una
por cuerda, la mas AGUDA arriba, como una tab real), con una columna por nota
-- el numero de traste en la fila de su cuerda y guiones en las demas. NO es
proporcional en el tiempo (cada nota es una columna, sin importar su duracion):
es el formato mas simple que abre cualquier editor de texto o se pega en
Guitar Pro/TuxGuitar a mano; el MIDI/MusicXML de al lado ya llevan el ritmo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from app.services.engines.music_transcription import NoteEvent

# Cuerda mas GRAVE primero, como se declaran las afinaciones (indice 0 = la
# mas gruesa). MIDI de la cuerda al aire.
GUITAR_STANDARD_TUNING: tuple[int, ...] = (40, 45, 50, 55, 59, 64)  # E2 A2 D3 G3 B3 E4
BASS_STANDARD_TUNING: tuple[int, ...] = (28, 33, 38, 43)  # E1 A1 D2 G2

TUNINGS: dict[str, tuple[int, ...]] = {
    "guitar": GUITAR_STANDARD_TUNING,
    "bass": BASS_STANDARD_TUNING,
}

MAX_FRET = 24

_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

FretAssignment = tuple[int, int]  # (indice de cuerda, traste)


def _candidates(pitch_midi: int, tuning: tuple[int, ...]) -> list[FretAssignment]:
    return [
        (string_index, pitch_midi - open_note)
        for string_index, open_note in enumerate(tuning)
        if 0 <= pitch_midi - open_note <= MAX_FRET
    ]


def _cheapest_predecessor(
    candidate: FretAssignment,
    previous_layer: dict[FretAssignment, tuple[int, FretAssignment | None]],
) -> tuple[FretAssignment, int]:
    best_prev: FretAssignment | None = None
    best_cost: int | None = None
    # Orden fijo (traste, cuerda) ascendente: con costo empatado gana el
    # primero visto, o sea el traste mas grave -- desempate determinista.
    for prev_choice, (prev_cost, _) in sorted(
        previous_layer.items(), key=lambda item: (item[0][1], item[0][0])
    ):
        cost = prev_cost + abs(candidate[1] - prev_choice[1])
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_prev = prev_choice
    assert best_prev is not None and best_cost is not None
    return best_prev, best_cost


def assign_frets(
    notes: Sequence[NoteEvent], tuning: tuple[int, ...]
) -> list[FretAssignment] | None:
    """Una asignacion (cuerda, traste) por nota, o `None` si alguna nota queda
    fuera del rango del instrumento (mas grave que la cuerda al aire mas grave,
    o mas aguda que el traste `MAX_FRET` de la mas aguda)."""
    if not notes:
        return []
    layers: list[dict[FretAssignment, tuple[int, FretAssignment | None]]] = []
    for index, note in enumerate(notes):
        candidates = _candidates(note.pitch_midi, tuning)
        if not candidates:
            return None
        if index == 0:
            layers.append({candidate: (0, None) for candidate in candidates})
            continue
        previous_layer = layers[-1]
        layer: dict[FretAssignment, tuple[int, FretAssignment | None]] = {}
        for candidate in candidates:
            prev_choice, cost = _cheapest_predecessor(candidate, previous_layer)
            layer[candidate] = (cost, prev_choice)
        layers.append(layer)

    last_layer = layers[-1]
    last_choice = min(last_layer, key=lambda c: (last_layer[c][0], c[1], c[0]))
    path: list[FretAssignment] = [last_choice]
    for layer in reversed(layers[1:]):
        _, prev_choice = layer[path[-1]]
        assert prev_choice is not None
        path.append(prev_choice)
    path.reverse()
    return path


def _string_label(open_midi: int) -> str:
    return _NOTE_NAMES[open_midi % 12]


def _render_tab(
    notes: Sequence[NoteEvent],
    assignment: Sequence[FretAssignment],
    instrument: str,
    tuning: tuple[int, ...],
) -> str:
    n_strings = len(tuning)
    columns: list[list[str]] = [[] for _ in range(n_strings)]
    for _note, (string_index, fret) in zip(notes, assignment):
        fret_text = str(fret)
        for row in range(n_strings):
            columns[row].append(fret_text if row == string_index else "-" * len(fret_text))

    tuning_label = "-".join(_string_label(open_note) for open_note in reversed(tuning))
    lines = [
        f"Tab de {instrument} (afinacion estandar {tuning_label}), borrador cuantizado "
        "-- sin bends/tecnicas, abrir en Guitar Pro/TuxGuitar para pulir.",
        "",
    ]
    for row in reversed(range(n_strings)):  # cuerda mas aguda arriba
        label = _string_label(tuning[row])
        content = "-".join(columns[row])
        lines.append(f"{label}|-{content}-|")
    return "\n".join(lines) + "\n"


def build_tab_text(notes: Sequence[NoteEvent], instrument: str) -> str | None:
    tuning = TUNINGS.get(instrument)
    if tuning is None:
        return None
    assignment = assign_frets(notes, tuning)
    if assignment is None:
        return None
    return _render_tab(notes, assignment, instrument, tuning)


def write_tab(notes: Sequence[NoteEvent], destination: Path, instrument: str) -> bool:
    text = build_tab_text(notes, instrument)
    if text is None:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return True
