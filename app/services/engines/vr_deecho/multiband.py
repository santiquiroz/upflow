# Vendored from santiquiroz/port-uvr-deecho-onnx driver/multiband.py @ commit
# 02cd199 (see app/services/engines/vr_deecho/__init__.py for sync notes).
# Imports rewritten: `from driver import dsp` -> `from
# app.services.engines.vr_deecho import dsp`, `from driver.vr_params import
# ...` -> `from app.services.engines.vr_deecho.vr_params import ...` (this repo
# has no top-level `driver` package). No other change.
"""4band_v3 multiband analysis/synthesis around the combined 673-bin spectrogram.

Faithful numpy port of the reference flow in python-audio-separator (MIT):
VRSeparator.loading_mix + spec_utils.combine_spectrograms (analysis) and
spec_utils.cmb_spectrogram_to_wave (synthesis), specialised to the 4band_v3
v5.1 parameter set the De-Echo/De-Reverb models use (no channel conversion,
no mid-side, LP/HP crossfade masks).
"""
from __future__ import annotations

import numpy as np

from app.services.engines.vr_deecho import dsp
from app.services.engines.vr_deecho.vr_params import BANDS, BINS, PRE_FILTER_START, PRE_FILTER_STOP, SR


def band_waves(mix: np.ndarray) -> list[np.ndarray]:
    if mix.ndim == 1:
        mix = np.stack([mix, mix])
    waves = [None] * len(BANDS)
    waves[-1] = mix.astype(np.float32, copy=False)
    for idx in range(len(BANDS) - 2, -1, -1):
        waves[idx] = dsp.resample(waves[idx + 1], BANDS[idx + 1]["sr"], BANDS[idx]["sr"])
    return waves


def combine_spectrograms(specs: list[np.ndarray]) -> np.ndarray:
    n_frames = min(spec.shape[2] for spec in specs)
    combined = np.zeros((2, BINS + 1, n_frames), dtype=np.complex64)
    row = 0
    for band, spec in zip(BANDS, specs):
        height = band["crop_stop"] - band["crop_start"]
        combined[:, row : row + height] = spec[:, band["crop_start"] : band["crop_stop"], :n_frames]
        row += height
    combined *= dsp.lp_filter_mask(BINS + 1, PRE_FILTER_START, PRE_FILTER_STOP)
    return combined


def wave_to_combined_spec(mix: np.ndarray) -> np.ndarray:
    specs = [dsp.stft_stereo(wave, band["n_fft"], band["hl"]) for band, wave in zip(BANDS, band_waves(mix))]
    return combine_spectrograms(specs)


def _band_full_spec(combined: np.ndarray, band_idx: int, row: int) -> np.ndarray:
    band = BANDS[band_idx]
    n_bins = band["n_fft"] // 2 + 1
    spec = np.zeros((2, n_bins, combined.shape[2]), dtype=np.complex64)
    height = band["crop_stop"] - band["crop_start"]
    spec[:, band["crop_start"] : band["crop_stop"]] = combined[:, row : row + height]
    if band_idx > 0:
        spec *= dsp.hp_filter_mask(n_bins, band["hpf_start"], band["hpf_stop"] - 1)
    if band_idx < len(BANDS) - 1:
        spec *= dsp.lp_filter_mask(n_bins, band["lpf_start"], band["lpf_stop"])
    return spec


def combined_spec_to_wave(combined: np.ndarray) -> np.ndarray:
    wave = None
    row = 0
    for band_idx, band in enumerate(BANDS):
        band_wave = dsp.istft_stereo(_band_full_spec(combined, band_idx, row), band["hl"])
        row += band["crop_stop"] - band["crop_start"]
        wave = band_wave if wave is None else wave + band_wave
        if band_idx < len(BANDS) - 1:
            wave = dsp.resample(wave, band["sr"], BANDS[band_idx + 1]["sr"])
    return wave


def resample_to_sr(wave: np.ndarray, target_sr: int) -> np.ndarray:
    return dsp.resample(wave, SR, target_sr)
