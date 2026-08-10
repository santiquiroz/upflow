# Vendored from santiquiroz/port-uvr-deecho-onnx driver/dsp.py @ commit
# 23b2564 (see app/services/engines/vr_deecho/__init__.py for sync notes).
# Verbatim except this header -- no internal `driver.*` imports to rewrite.
"""librosa-compatible STFT/iSTFT and polyphase resampling in pure numpy/scipy.

stft/istft reproduce librosa.stft/librosa.istft defaults (hann periodic window,
win_length = n_fft, center=True, pad_mode="constant") bit-for-bit in float32.
resample matches librosa's res_type="polyphase" (scipy.signal.resample_poly with
sr // gcd factors), which is what the reference uses for every downsample.
The reference's upsamples use libsamplerate "sinc_fastest"; this driver uses the
same polyphase kernel instead -- divergence is measured, not hidden (see README).
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import resample_poly


def hann_periodic(n_fft: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)


def stft(wave: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    pad = n_fft // 2
    y = np.pad(wave.astype(np.float32, copy=False), pad, mode="constant")
    frames = sliding_window_view(y, n_fft)[::hop]
    spec = np.fft.rfft(frames * hann_periodic(n_fft), axis=-1)
    return spec.astype(np.complex64).T


def istft(spec: np.ndarray, hop: int) -> np.ndarray:
    n_bins, n_frames = spec.shape
    n_fft = 2 * (n_bins - 1)
    window = hann_periodic(n_fft)
    frames = np.fft.irfft(spec, n=n_fft, axis=0).real.astype(np.float32) * window[:, None]
    total = n_fft + hop * (n_frames - 1)
    wave = np.zeros(total, dtype=np.float32)
    win_sq_sum = np.zeros(total, dtype=np.float32)
    win_sq = window * window
    for t in range(n_frames):
        start = t * hop
        wave[start : start + n_fft] += frames[:, t]
        win_sq_sum[start : start + n_fft] += win_sq
    nonzero = win_sq_sum > np.finfo(np.float32).tiny
    wave[nonzero] /= win_sq_sum[nonzero]
    pad = n_fft // 2
    return wave[pad : total - pad]


def stft_stereo(wave: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    return np.stack([stft(wave[0], n_fft, hop), stft(wave[1], n_fft, hop)])


def istft_stereo(spec: np.ndarray, hop: int) -> np.ndarray:
    return np.stack([istft(spec[0], hop), istft(spec[1], hop)])


def resample(wave: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wave
    gcd = np.gcd(orig_sr, target_sr)
    out = resample_poly(wave, target_sr // gcd, orig_sr // gcd, axis=-1)
    return out.astype(np.float32, copy=False)


def lp_filter_mask(n_bins: int, bin_start: int, bin_stop: int) -> np.ndarray:
    return np.concatenate(
        [np.ones((bin_start - 1, 1)), np.linspace(1, 0, bin_stop - bin_start + 1)[:, None], np.zeros((n_bins - bin_stop, 1))]
    )


def hp_filter_mask(n_bins: int, bin_start: int, bin_stop: int) -> np.ndarray:
    return np.concatenate(
        [np.zeros((bin_stop + 1, 1)), np.linspace(0, 1, 1 + bin_start - bin_stop)[:, None], np.ones((n_bins - bin_start - 2, 1))]
    )
