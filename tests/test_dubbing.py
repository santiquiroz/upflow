from __future__ import annotations

import numpy as np
import pytest

from app.services.dubbing import (
    SPEED_MAX,
    voice_for_language,
    DubbedPiece,
    assemble_track,
    count_overflowing,
    speed_for_slot,
)

# ---------------------------------------------------------------------------
# El doblaje tiene un problema que no tiene la traduccion: lo traducido casi
# nunca dura lo mismo que el original. Si cada linea se suelta donde cae, a los
# treinta segundos la voz habla sobre la escena equivocada.
#
# La salida es: sintetizar a la velocidad que hace falta para entrar en el hueco
# del original, con un tope — pasado cierto punto la voz deja de entenderse y
# apurarla mas es peor que dejarla salir del hueco.
# ---------------------------------------------------------------------------


class TestSpeedForSlot:
    def test_speech_that_already_fits_is_not_touched(self) -> None:
        assert speed_for_slot(natural_seconds=2.0, slot_seconds=2.0) == 1.0

    def test_speech_longer_than_the_slot_is_sped_up(self) -> None:
        # Tres segundos de voz en un hueco de dos: hay que ir a 1,5x.
        assert speed_for_slot(natural_seconds=3.0, slot_seconds=2.0) == pytest.approx(1.5)

    def test_speech_shorter_than_the_slot_is_left_alone(self) -> None:
        # Estirarla para llenar el hueco la dejaria arrastrada, y el silencio
        # que queda es el mismo que tenia el original.
        assert speed_for_slot(natural_seconds=1.0, slot_seconds=2.0) == 1.0

    def test_an_absurd_ratio_is_clamped_instead_of_making_it_unintelligible(self) -> None:
        # Diez segundos en un hueco de uno serian 10x: nadie entiende eso.
        assert speed_for_slot(natural_seconds=10.0, slot_seconds=1.0) == SPEED_MAX

    def test_a_much_shorter_line_is_left_alone_too(self) -> None:
        assert speed_for_slot(natural_seconds=0.2, slot_seconds=10.0) == 1.0

    def test_an_empty_slot_does_not_divide_by_zero(self) -> None:
        assert speed_for_slot(natural_seconds=2.0, slot_seconds=0.0) == SPEED_MAX

    def test_silence_keeps_the_normal_speed(self) -> None:
        assert speed_for_slot(natural_seconds=0.0, slot_seconds=2.0) == 1.0


class TestAssembleTrack:
    def test_each_piece_lands_at_its_own_second(self) -> None:
        pieces = [
            DubbedPiece(start=0.0, audio=np.ones(40, dtype=np.float32)),
            DubbedPiece(start=1.0, audio=np.ones(40, dtype=np.float32)),
        ]

        track = assemble_track(pieces, total_seconds=2.0, sample_rate=100)

        assert track[0] == pytest.approx(1.0)
        assert track[50] == pytest.approx(0.0)  # el hueco entre los dos
        assert track[100] == pytest.approx(1.0)

    def test_the_track_lasts_as_long_as_the_video(self) -> None:
        pieces = [DubbedPiece(start=0.0, audio=np.ones(10, dtype=np.float32))]

        track = assemble_track(pieces, total_seconds=3.0, sample_rate=100)

        assert len(track) == 300

    def test_a_piece_that_runs_past_the_end_stretches_the_track(self) -> None:
        # Cortar la ultima palabra para respetar una duracion nominal seria
        # perder contenido: el video se alarga, no la voz se corta.
        pieces = [DubbedPiece(start=0.9, audio=np.ones(50, dtype=np.float32))]

        track = assemble_track(pieces, total_seconds=1.0, sample_rate=100)

        assert len(track) == 140
        assert track[-1] == pytest.approx(1.0)

    def test_overlapping_pieces_are_mixed_not_dropped(self) -> None:
        # Donde dos lineas se pisan tiene que sonar MAS que donde suena una
        # sola: si la segunda se descartara, sonarian iguales.
        pieces = [
            DubbedPiece(start=0.0, audio=np.ones(100, dtype=np.float32)),
            DubbedPiece(start=0.5, audio=np.ones(100, dtype=np.float32)),
        ]

        track = assemble_track(pieces, total_seconds=2.0, sample_rate=100)

        assert track[60] > track[10]

    def test_the_mix_never_clips_the_waveform(self) -> None:
        # Dos voces sumadas se pasan de 1.0 y el wav sale distorsionado.
        pieces = [
            DubbedPiece(start=0.0, audio=np.full(100, 0.9, dtype=np.float32)),
            DubbedPiece(start=0.0, audio=np.full(100, 0.9, dtype=np.float32)),
        ]

        track = assemble_track(pieces, total_seconds=1.0, sample_rate=100)

        assert float(np.max(np.abs(track))) <= 1.0

    def test_an_empty_list_still_gives_a_silent_track_of_the_right_length(self) -> None:
        track = assemble_track([], total_seconds=2.0, sample_rate=100)

        assert len(track) == 200
        assert float(np.max(np.abs(track))) == 0.0


class TestCountOverflowing:
    def test_counts_the_lines_that_did_not_fit_even_at_full_speed(self) -> None:
        pieces = [
            DubbedPiece(start=0.0, audio=np.ones(100, dtype=np.float32), slot_seconds=1.0),
            DubbedPiece(start=1.0, audio=np.ones(300, dtype=np.float32), slot_seconds=1.0),
        ]

        assert count_overflowing(pieces, sample_rate=100) == 1

    def test_a_track_where_everything_fit_reports_zero(self) -> None:
        pieces = [DubbedPiece(start=0.0, audio=np.ones(50, dtype=np.float32), slot_seconds=1.0)]

        assert count_overflowing(pieces, sample_rate=100) == 0


class TestVoiceForLanguage:
    """Kokoro nombra sus voces con la inicial del idioma: `e` español, `f`
    francés, `a` inglés americano. Doblar al español con una voz inglesa suena a
    extranjero leyendo — no está roto, pero es peor de lo necesario."""

    VOCES = ["af_heart", "am_michael", "ef_dora", "em_alex", "ff_siwis"]

    def test_picks_a_voice_of_the_target_language(self) -> None:
        assert voice_for_language(self.VOCES, "es") in {"ef_dora", "em_alex"}

    def test_picks_the_french_voice_for_french(self) -> None:
        assert voice_for_language(self.VOCES, "fr") == "ff_siwis"

    def test_falls_back_to_the_first_voice_when_the_language_has_none(self) -> None:
        # Doblar con acento es peor que no doblar? No: es peor no doblar.
        assert voice_for_language(self.VOCES, "ja") == "af_heart"

    def test_without_any_installed_voice_there_is_nothing_to_pick(self) -> None:
        assert voice_for_language([], "es") is None

    def test_an_unknown_language_code_does_not_explode(self) -> None:
        assert voice_for_language(self.VOCES, "") == "af_heart"
