import numpy as np
import pytest

from app.services.engines.multichannel_restore import restore_multichannel


def _identity(mono: np.ndarray) -> np.ndarray:
    return mono


def test_mono_input_calls_restore_mono_directly():
    audio = np.array([[0.1], [0.2], [-0.3]], dtype=np.float32)
    calls = []

    def spy(mono):
        calls.append(mono.copy())
        return mono * 2

    result = restore_multichannel(audio, spy)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], audio[:, 0])
    # RMS-match normaliza la ganancia uniforme (x2) del modelo de vuelta al nivel
    # original, por eso el resultado recupera el nivel de entrada (no queda x2).
    np.testing.assert_allclose(result[:, 0], audio[:, 0], rtol=1e-6)
    assert result.shape == audio.shape


def test_stereo_round_trip_with_identity_restore_reproduces_original():
    rng = np.random.default_rng(0)
    audio = rng.uniform(-0.5, 0.5, size=(2000, 2)).astype(np.float32)
    result = restore_multichannel(audio, _identity)
    np.testing.assert_allclose(result, audio, atol=1e-6)


def test_stereo_side_channel_never_passed_to_restore_mono():
    audio = np.zeros((4, 2), dtype=np.float32)
    audio[:, 0] = [1.0, 1.0, 1.0, 1.0]  # L
    audio[:, 1] = [-1.0, -1.0, -1.0, -1.0]  # R -> mid=0, side=1
    calls = []

    def spy(mono):
        calls.append(mono.copy())
        return mono

    restore_multichannel(audio, spy)
    np.testing.assert_allclose(calls[0], np.zeros(4, dtype=np.float32))  # solo el mid (0) llega al modelo


def test_stereo_restoring_mid_changes_only_shared_content():
    audio = np.zeros((4, 2), dtype=np.float32)
    audio[:, 0] = [1.0, 1.0, 1.0, 1.0]
    audio[:, 1] = [1.0, 1.0, 1.0, 1.0]  # side=0, mid=1 (mono content)

    def double_mid(mono):
        return mono * 2

    result = restore_multichannel(audio, double_mid)
    # side=0; solo el mid (contenido compartido) se restaura, y RMS-match normaliza
    # la ganancia uniforme (x2) del modelo devolviendo el nivel original en ambos canales.
    np.testing.assert_allclose(result[:, 0], [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(result[:, 1], [1.0, 1.0, 1.0, 1.0])


def test_rms_matches_original_within_tolerance():
    rng = np.random.default_rng(1)
    audio = (rng.uniform(-0.5, 0.5, size=(4000, 2)) * 0.3).astype(np.float32)

    def amplify(mono):
        return mono * 5.0  # simula un modelo que altera el nivel

    result = restore_multichannel(audio, amplify)
    original_rms = np.sqrt(np.mean(audio**2))
    result_rms = np.sqrt(np.mean(result**2))
    assert result_rms == pytest.approx(original_rms, rel=0.05)


def test_more_than_two_channels_raises_not_implemented():
    audio = np.zeros((8, 6), dtype=np.float32)  # surround: Task 7
    with pytest.raises(NotImplementedError):
        restore_multichannel(audio, _identity)
