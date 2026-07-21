import numpy as np
import pytest

from app.services.engines.multichannel_layouts import restore_surround


def _identity(mono: np.ndarray) -> np.ndarray:
    return mono


def test_51_round_trip_identity_reproduces_original():
    rng = np.random.default_rng(2)
    audio = rng.uniform(-0.3, 0.3, size=(1000, 6)).astype(np.float32)
    result = restore_surround(audio, "5.1", _identity)
    np.testing.assert_allclose(result, audio, atol=1e-6)


def test_51_lfe_channel_untouched():
    audio = np.zeros((100, 6), dtype=np.float32)
    audio[:, 3] = 0.42  # LFE

    def fail_if_called(mono):
        raise AssertionError("LFE must never reach restore_mono")

    result = restore_surround(audio, "5.1", lambda mono: mono if np.allclose(mono, 0.42) is False else fail_if_called(mono))
    np.testing.assert_allclose(result[:, 3], audio[:, 3])


def test_51_center_channel_is_rms_matched_to_original_level():
    audio = np.zeros((100, 6), dtype=np.float32)
    audio[:, 2] = 0.5  # FC

    def double(mono):
        return mono * 2

    result = restore_surround(audio, "5.1", double)
    # Even with a double restore model, RMS-match brings the level back to the original 0.5
    np.testing.assert_allclose(result[:, 2], np.full(100, 0.5))


def test_71_has_eight_channels_round_trip():
    rng = np.random.default_rng(3)
    audio = rng.uniform(-0.3, 0.3, size=(500, 8)).astype(np.float32)
    result = restore_surround(audio, "7.1", _identity)
    np.testing.assert_allclose(result, audio, atol=1e-6)


def test_unknown_layout_raises():
    with pytest.raises(ValueError, match="layout"):
        restore_surround(np.zeros((10, 6), dtype=np.float32), "9.1", _identity)
