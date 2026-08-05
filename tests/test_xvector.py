from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services.xvector import (
    HOP_LENGTH,
    N_MELS,
    WIN_LENGTH,
    XvectorEncoder,
    XvectorUnavailable,
    compute_fbank,
    normalize_sentence,
)

# ---------------------------------------------------------------------------
# El TDNN se exporto a ONNX pero su frontend de features NO: revienta al trazar
# el STFT. Asi que el Fbank se reproduce aca, y un detalle mal copiado deja los
# embeddings fuera del espacio y la conversion deja de clonar SIN fallar.
#
# Verificado de punta a punta el 2026-08-05: clonando una muestra masculina
# sobre una voz femenina, la salida quedo a 0,777 del destino y 0,446 del
# origen. Con el frontend equivocado (WavLM) daba 0,609 y 0,673 — al reves.
# ---------------------------------------------------------------------------


def tono(segundos: float = 1.0, hz: float = 220.0) -> np.ndarray:
    t = np.linspace(0, segundos, int(16000 * segundos), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


class TestFbank:
    def test_gives_one_column_per_mel_band(self) -> None:
        assert compute_fbank(tono()).shape[1] == N_MELS

    def test_frame_count_follows_the_hop(self) -> None:
        audio = tono(1.0)
        esperado = 1 + (len(audio) - WIN_LENGTH) // HOP_LENGTH
        assert compute_fbank(audio).shape[0] == esperado

    def test_the_log_never_produces_infinities_on_silence(self) -> None:
        # Sin el epsilon, log(0) da -inf y el embedding sale NaN sin fallar.
        silencio = np.zeros(16000, dtype=np.float32)
        assert np.all(np.isfinite(compute_fbank(silencio)))

    def test_audio_shorter_than_one_window_is_refused(self) -> None:
        with pytest.raises(XvectorUnavailable):
            compute_fbank(np.zeros(WIN_LENGTH // 2, dtype=np.float32))

    def test_a_stereo_shaped_array_is_flattened_rather_than_refused(self) -> None:
        assert compute_fbank(tono().reshape(-1, 1)).shape[1] == N_MELS


class TestNormalization:
    def test_removes_the_sentence_mean(self) -> None:
        feats = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        assert np.allclose(normalize_sentence(feats).mean(axis=0), 0.0)

    def test_does_not_divide_by_the_deviation(self) -> None:
        # speechbrain usa std_norm=False. Dividir tambien por el desvio deja el
        # embedding fuera del espacio del modelo.
        feats = np.array([[0.0], [10.0]], dtype=np.float32)
        salida = normalize_sentence(feats)
        assert float(salida.std()) == pytest.approx(5.0)


class TestEncoder:
    def test_reports_unavailable_without_the_exported_model(self, tmp_path: Path) -> None:
        assert XvectorEncoder(tmp_path / "no-esta.onnx").available() is False

    def test_says_how_to_get_the_model_when_it_is_missing(self, tmp_path: Path) -> None:
        with pytest.raises(XvectorUnavailable, match="export_xvector_onnx"):
            XvectorEncoder(tmp_path / "no-esta.onnx").encode(tono())
