from __future__ import annotations

import numpy as np
import pytest

from app.services.tts_tokens import (
    MAX_TOKENS,
    build_input_ids,
    phonemes_to_tokens,
    style_for_length,
)

# ---------------------------------------------------------------------------
# Kokoro no recibe texto: recibe FONEMAS tokenizados con su propio vocabulario
# (115 simbolos, con IPA adentro). Y el "estilo" de la voz no es un vector fijo:
# el archivo de voz trae UNA FILA POR LONGITUD de secuencia, y hay que elegir la
# que corresponde al largo de este texto.
#
# Todo esto es aritmetica pura y se prueba sin bajar el modelo.
# ---------------------------------------------------------------------------

VOCAB = {"a": 10, "b": 11, "ə": 12, "ˈ": 13, " ": 14}


class TestPhonemesToTokens:
    def test_maps_each_symbol_through_the_vocabulary(self) -> None:
        assert phonemes_to_tokens("ab", VOCAB) == [10, 11]

    def test_keeps_ipa_and_stress_marks(self) -> None:
        # Descartar el acento cambiaria la prosodia: no es ruido.
        assert phonemes_to_tokens("ˈaə", VOCAB) == [13, 10, 12]

    def test_drops_a_symbol_the_model_cannot_read(self) -> None:
        # Mandar un id inventado produciria audio raro sin fallar.
        assert phonemes_to_tokens("aXb", VOCAB) == [10, 11]

    def test_an_empty_phrase_gives_no_tokens(self) -> None:
        assert phonemes_to_tokens("", VOCAB) == []


class TestBuildInputIds:
    def test_wraps_the_sequence_in_boundary_tokens(self) -> None:
        ids = build_input_ids([10, 11])
        assert ids.tolist() == [[0, 10, 11, 0]]

    def test_uses_the_integer_width_the_model_declares(self) -> None:
        assert build_input_ids([10]).dtype == np.int64

    def test_refuses_an_empty_sequence_instead_of_running_on_silence(self) -> None:
        with pytest.raises(ValueError):
            build_input_ids([])


class TestStyleForLength:
    def test_picks_the_row_that_matches_this_text(self) -> None:
        voices = np.arange(10 * 256, dtype=np.float32).reshape(10, 256)
        style = style_for_length(voices, 3)
        assert style.shape == (1, 256)
        assert style[0][0] == voices[3][0]

    def test_keeps_the_two_dimensional_shape_the_model_declares(self) -> None:
        # El modelo declara `style` con forma [1, 256]. Pasarle un vector plano
        # lo rechaza.
        voices = np.zeros((5, 256), dtype=np.float32)
        assert style_for_length(voices, 1).ndim == 2

    def test_a_text_longer_than_the_table_uses_the_last_row(self) -> None:
        # Sin este tope, un texto largo indexaria fuera del array y reventaria.
        voices = np.zeros((5, 256), dtype=np.float32)
        assert style_for_length(voices, 99).shape == (1, 256)


class TestTokenCeiling:
    def test_there_is_a_ceiling_so_one_request_cannot_hang_the_queue(self) -> None:
        assert MAX_TOKENS > 0

    def test_a_long_phrase_is_cut_at_the_ceiling(self) -> None:
        tokens = phonemes_to_tokens("a" * (MAX_TOKENS + 50), VOCAB)
        assert len(tokens) == MAX_TOKENS
