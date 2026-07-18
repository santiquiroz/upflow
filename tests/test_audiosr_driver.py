from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from app.services.engines.audiosr import dsp
from app.services.engines.audiosr.assets import AudioSrAssets
from app.services.engines.audiosr.ddim import DdimSchedule, combine_cfg, ddim_step
from app.services.engines.audiosr.driver import (
    AudioSrCancelled,
    AudioSrDriver,
    _crossfade_concat,
    _pad_to_unit,
    _split_windows,
)

# ---------------------------------------------------------------------------
# SP13 - AudioSR numpy driver. The numeric semantics are parity-locked against
# the PyTorch baseline in the port-audiosr-onnx toolkit; these tests cover the
# pure math and plumbing so regressions surface without the 1.7GB models.
# ---------------------------------------------------------------------------


def make_fake_assets(tmp_path: Path) -> AudioSrAssets:
    rng = np.random.default_rng(7)
    np.save(tmp_path / "alphas_cumprod.npy", np.linspace(0.9999, 0.0, 1000))
    np.save(tmp_path / "mel_basis.npy", rng.random((256, 1025), dtype=np.float32) * 0.01)
    manifest = {
        "scale_factor": 0.3342,
        "mel": {"basis_file": "mel_basis.npy"},
        "scheduler": {"alphas_cumprod_file": "alphas_cumprod.npy"},
        "cfg": {"guidance_scale": 3.5, "unconditional_value": -11.4981},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for name in ("vocoder", "vae_decoder", "vae_feature_extract", "ddpm"):
        (tmp_path / f"{name}.onnx").write_bytes(b"fake")
    return AudioSrAssets.load(tmp_path)


def make_fake_run_graph(calls: list[str] | None = None):
    def run_graph(name: str, feeds: dict[str, np.ndarray]) -> np.ndarray:
        if calls is not None:
            calls.append(name)
        if name == "vae_feature_extract":
            return feeds["noise"] * 0.1
        if name == "ddpm":
            return feeds["x"][:, :16] * 0.5
        if name == "vae_decoder":
            z = feeds["z"]
            frames = z.shape[2] * 8
            return np.full((1, 1, frames, 256), -3.0, dtype=np.float32)
        if name == "vocoder":
            mel = feeds["mel"]
            samples = mel.shape[2] * 480
            t = np.arange(samples, dtype=np.float32)
            return (0.3 * np.sin(2 * np.pi * 440 * t / 48000)).reshape(1, 1, -1)
        raise AssertionError(f"unexpected graph {name}")

    return run_graph


class TestDdimSchedule:
    def test_uniform_timesteps_match_audiosr_make_ddim_timesteps(self) -> None:
        schedule = DdimSchedule.build(np.linspace(1.0, 0.0, 1000), num_steps=50)

        assert schedule.timesteps[0] == 1
        assert schedule.timesteps[-1] == 981
        assert len(schedule.timesteps) == 50
        assert np.all(np.diff(schedule.timesteps) == 20)

    def test_alphas_prev_shifts_with_first_train_alpha(self) -> None:
        alphas_cumprod = np.linspace(0.999, 0.001, 1000)
        schedule = DdimSchedule.build(alphas_cumprod, num_steps=10)

        assert schedule.alphas_prev[0] == alphas_cumprod[0]
        np.testing.assert_allclose(
            schedule.alphas_prev[1:], alphas_cumprod[schedule.timesteps[:-1]]
        )

    def test_eta_one_yields_positive_sigmas(self) -> None:
        schedule = DdimSchedule.build(np.linspace(0.9999, 0.0001, 1000), num_steps=25)

        assert np.all(schedule.sigmas[1:] > 0)

    def test_ddim_step_recovers_x0_at_final_step(self) -> None:
        # With sigma=0 and a_prev=1 the update must return pred_x0 exactly.
        alphas_cumprod = np.linspace(0.9999, 0.0001, 1000)
        schedule = DdimSchedule.build(alphas_cumprod, num_steps=10)
        object.__setattr__(schedule, "alphas_prev", np.ones_like(schedule.alphas_prev))
        object.__setattr__(schedule, "sigmas", np.zeros_like(schedule.sigmas))
        x = np.ones((1, 16, 4, 4), dtype=np.float32)
        v = np.full_like(x, 0.5)
        t = int(schedule.timesteps[0])

        result = ddim_step(x, v, t, 0, schedule, np.zeros_like(x))

        sqrt_a = np.sqrt(alphas_cumprod[t])
        sqrt_1ma = np.sqrt(1 - alphas_cumprod[t])
        np.testing.assert_allclose(result, sqrt_a * x - sqrt_1ma * v, rtol=1e-6)

    def test_combine_cfg_matches_formula(self) -> None:
        v_cond = np.array([2.0])
        v_uncond = np.array([1.0])

        assert combine_cfg(v_cond, v_uncond, 3.5)[0] == pytest.approx(1.0 + 3.5 * 1.0)


class TestDsp:
    def test_normalize_wav_centers_and_scales_to_half_peak(self) -> None:
        wav = np.array([0.0, 2.0, -2.0, 4.0], dtype=np.float64)

        normalized = dsp.normalize_wav(wav)

        assert abs(np.max(np.abs(normalized)) - 0.5) < 1e-6

    def test_stft_magnitude_frame_count_is_100_fps(self) -> None:
        wav = np.random.default_rng(0).standard_normal(48000)

        mag = dsp.stft_magnitude(wav)

        assert mag.shape == (100, 1025)

    def test_pad_spec_pads_cuts_and_trims_odd_bins(self) -> None:
        spec = np.ones((10, 1025))

        padded = dsp.pad_spec(spec, 12)

        assert padded.shape == (12, 1024)
        assert padded[10:].sum() == 0

    def test_detect_cutoff_low_energy_clip_returns_nyquist(self) -> None:
        silent = np.zeros((100, 1024))
        silent[:, :2] = 1.0  # all energy below 1kHz

        assert dsp.detect_cutoff_hz(silent) == 24000.0

    def test_lowpass_simulate_removes_high_band(self) -> None:
        t = np.arange(48000) / 48000
        low = np.sin(2 * np.pi * 1000 * t)
        high = np.sin(2 * np.pi * 20000 * t)

        filtered = dsp.lowpass_simulate(low + high, cutoff_hz=4000)

        high_energy = dsp.stft_magnitude(filtered)[:, 800:].mean()
        low_energy = dsp.stft_magnitude(filtered)[:, 30:60].mean()
        assert high_energy < low_energy / 100

    def test_librosa_stft_istft_roundtrip(self) -> None:
        t = np.arange(48000) / 48000
        wav = 0.4 * np.sin(2 * np.pi * 440 * t)

        rebuilt = dsp.librosa_istft(dsp.librosa_stft(wav), len(wav))

        np.testing.assert_allclose(rebuilt, wav, atol=1e-8)

    def test_replace_low_band_keeps_length(self) -> None:
        rng = np.random.default_rng(1)
        restored = rng.standard_normal(48000) * 0.1
        lowpass = rng.standard_normal(48000) * 0.1

        out = dsp.replace_low_band_stft(restored, lowpass)

        assert out.shape == restored.shape


class TestWindowing:
    def test_short_clip_is_single_window(self) -> None:
        assert _split_windows(48000 * 5) == [(0, 48000 * 5)]

    def test_long_clip_splits_with_overlap(self) -> None:
        total = int(48000 * 15.36)

        windows = _split_windows(total)

        assert len(windows) == 2
        assert windows[0] == (0, int(48000 * 10.24))
        assert windows[1][1] == total

    def test_pad_to_unit_multiple_of_5_12s(self) -> None:
        padded = _pad_to_unit(np.ones(48000))

        assert padded.shape[-1] == int(48000 * 5.12)

    def test_crossfade_reconstructs_constant_signal(self) -> None:
        total = int(48000 * 15.36)
        windows = _split_windows(total)
        chunks = [np.ones(end - start) for start, end in windows]

        merged = _crossfade_concat(chunks, total)

        np.testing.assert_allclose(merged, np.ones(total), atol=1e-9)


class TestDriver:
    def test_restore_preserves_length_and_calls_graphs(self, tmp_path: Path) -> None:
        assets = make_fake_assets(tmp_path)
        calls: list[str] = []
        driver = AudioSrDriver(assets, make_fake_run_graph(calls))
        wav = np.random.default_rng(3).standard_normal(48000 * 2).astype(np.float32) * 0.2

        out = driver.restore(wav, ddim_steps=4)

        assert out.shape == wav.shape
        assert out.dtype == np.float32
        assert calls.count("vae_feature_extract") == 1
        assert calls.count("ddpm") == 8  # 4 steps x 2 CFG passes
        assert calls.count("vae_decoder") == 1
        assert calls.count("vocoder") == 1

    def test_progress_callback_reports_every_step(self, tmp_path: Path) -> None:
        assets = make_fake_assets(tmp_path)
        driver = AudioSrDriver(assets, make_fake_run_graph())
        seen: list[tuple[int, int]] = []
        wav = np.random.default_rng(4).standard_normal(48000).astype(np.float32) * 0.2

        driver.restore(wav, ddim_steps=3, progress_cb=lambda done, total: seen.append((done, total)))

        assert seen == [(1, 3), (2, 3), (3, 3)]

    def test_cancel_event_aborts_promptly(self, tmp_path: Path) -> None:
        assets = make_fake_assets(tmp_path)
        driver = AudioSrDriver(assets, make_fake_run_graph())
        cancel = threading.Event()
        cancel.set()
        wav = np.zeros(48000, dtype=np.float32)
        wav[0] = 0.5

        with pytest.raises(AudioSrCancelled):
            driver.restore(wav, ddim_steps=3, cancel_event=cancel)

    def test_long_input_processes_multiple_windows(self, tmp_path: Path) -> None:
        assets = make_fake_assets(tmp_path)
        calls: list[str] = []
        driver = AudioSrDriver(assets, make_fake_run_graph(calls))
        wav = np.random.default_rng(5).standard_normal(int(48000 * 12)).astype(np.float32) * 0.2

        out = driver.restore(wav, ddim_steps=2)

        assert out.shape == wav.shape
        assert calls.count("vae_feature_extract") == 2
        assert calls.count("ddpm") == 8  # 2 windows x 2 steps x 2 CFG


class TestAssets:
    def test_is_complete_requires_all_files(self, tmp_path: Path) -> None:
        make_fake_assets(tmp_path)

        assert AudioSrAssets.is_complete(tmp_path) is True

        (tmp_path / "ddpm.onnx").unlink()
        assert AudioSrAssets.is_complete(tmp_path) is False

    def test_load_exposes_manifest_scalars(self, tmp_path: Path) -> None:
        assets = make_fake_assets(tmp_path)

        assert assets.scale_factor == pytest.approx(0.3342)
        assert assets.guidance_scale == pytest.approx(3.5)
        assert assets.unconditional_value == pytest.approx(-11.4981)
        assert assets.alphas_cumprod.shape == (1000,)
