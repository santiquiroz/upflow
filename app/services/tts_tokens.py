"""Tokenizado y estilo de voz para Kokoro.

Kokoro no recibe texto sino FONEMAS tokenizados con su propio vocabulario (115
simbolos, con IPA adentro). Y el "estilo" de la voz no es un vector fijo: el
archivo de voz trae UNA FILA POR LONGITUD de secuencia, y hay que elegir la que
corresponde al largo de este texto.

Aritmetica pura: se prueba entera sin bajar el modelo.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# El modelo tiene una fila de estilo por longitud (510 en la voz medida). Un
# texto mas largo que eso indexaria fuera de la tabla, asi que se corta antes.
MAX_TOKENS = 500

# Kokoro espera la secuencia entre tokens de borde.
_BOUNDARY_TOKEN = 0


def phonemes_to_tokens(phonemes: str, vocab: dict[str, int]) -> list[int]:
    # Un simbolo que el vocabulario no conoce se DESCARTA en vez de mapearse a
    # un id inventado: un id fuera de tabla produce audio raro sin fallar.
    tokens = [vocab[symbol] for symbol in phonemes if symbol in vocab]
    return tokens[:MAX_TOKENS]


def build_input_ids(tokens: list[int]) -> Any:
    if not tokens:
        raise ValueError("No hay fonemas que sintetizar: el texto quedo vacio.")
    return np.array([[_BOUNDARY_TOKEN, *tokens, _BOUNDARY_TOKEN]], dtype=np.int64)


def style_for_length(voices: Any, token_count: int) -> Any:
    row = min(max(token_count, 0), len(voices) - 1)
    # El modelo declara `style` con forma [1, 256]: el slice conserva las dos
    # dimensiones, mientras que `voices[row]` devolveria un vector plano que el
    # modelo rechaza.
    return voices[row : row + 1]
