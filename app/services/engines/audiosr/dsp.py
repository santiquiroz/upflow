from __future__ import annotations

import numpy as np

# Faithful numpy port of audiosr 0.0.7's DSP glue (utils.py, lowpass.py and the
# librosa calls inside LatentDiffusion.postprocessing). Every function here is
# parity-tested against tensors captured from the original PyTorch pipeline.

SAMPLE_RATE = 48000
N_FFT = 2048
HOP = 480
N_MELS = 256
STFT_PAD = (N_FFT - HOP) // 2  # 784, reflect
LOG_CLIP = 1e-5

POST_N_FFT = 2048
POST_HOP = 512


def hann_periodic(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def _frame(signal: np.ndarray, frame_length: int, hop: int) -> np.ndarray:
    n_frames = 1 + (signal.shape[-1] - frame_length) // hop
    strides = (signal.strides[-1] * hop, signal.strides[-1])
    return np.lib.stride_tricks.as_strided(
        signal, shape=(n_frames, frame_length), strides=strides
    )


def stft_magnitude(wav: np.ndarray) -> np.ndarray:
    """torch.stft(center=False) after reflect-pad 784: |STFT| as [frames, 1025]."""
    padded = np.pad(wav.astype(np.float64), (STFT_PAD, STFT_PAD), mode="reflect")
    frames = _frame(padded, N_FFT, HOP) * hann_periodic(N_FFT)
    return np.abs(np.fft.rfft(frames, axis=-1))


def log_mel(stft_mag_tf: np.ndarray, mel_basis: np.ndarray) -> np.ndarray:
    """[frames, 1025] -> log-mel [frames, 256] (clamp 1e-5, natural log)."""
    mel = stft_mag_tf @ mel_basis.T
    return np.log(np.clip(mel, LOG_CLIP, None))


def pad_spec(spec_tf: np.ndarray, target_frames: int) -> np.ndarray:
    n = spec_tf.shape[0]
    if n < target_frames:
        spec_tf = np.pad(spec_tf, ((0, target_frames - n), (0, 0)))
    elif n > target_frames:
        spec_tf = spec_tf[:target_frames]
    if spec_tf.shape[-1] % 2 != 0:
        spec_tf = spec_tf[..., :-1]
    return spec_tf


def normalize_wav(wav: np.ndarray) -> np.ndarray:
    wav = wav - np.mean(wav)
    wav = wav / (np.max(np.abs(wav)) + 1e-8)
    return (wav * 0.5).astype(np.float32)


def find_cutoff_index(cumulative_energy: np.ndarray, percentile: float) -> int:
    threshold = cumulative_energy[-1] * percentile
    for i in range(1, cumulative_energy.shape[0]):
        if cumulative_energy[-i] < threshold:
            return cumulative_energy.shape[0] - i
    return 0


def locate_cutoff_bin(magnitude_tf: np.ndarray, percentile: float) -> int:
    """Cumulative-energy cutoff over the LAST axis of a [frames, bins] magnitude."""
    energy = np.cumsum(np.sum(magnitude_tf, axis=0))
    return find_cutoff_index(energy, percentile)


def detect_cutoff_hz(stft_mag_tf: np.ndarray) -> float:
    cutoff_hz = (locate_cutoff_bin(stft_mag_tf, percentile=0.985) / 1024) * 24000
    if cutoff_hz < 1000:
        return 24000.0
    return float(cutoff_hz)


def lowpass_simulate(wav: np.ndarray, cutoff_hz: float, ftype: str = "butter") -> np.ndarray:
    """audiosr.lowpass.lowpass_filter: sosfiltfilt -> resample down/up -> sosfiltfilt."""
    if cutoff_hz >= 23999:
        return wav.astype(np.float64)
    from scipy.signal import bessel, butter, cheby1, ellip, resample_poly, sosfiltfilt

    nyq = SAMPLE_RATE / 2
    hi = int(cutoff_hz) / nyq
    order = 8
    if ftype == "butter":
        sos = butter(order, hi, btype="low", output="sos")
    elif ftype == "cheby1":
        sos = cheby1(order, 0.1, hi, btype="low", output="sos")
    elif ftype == "ellip":
        sos = ellip(order, 0.1, 60, hi, btype="low", output="sos")
    elif ftype == "bessel":
        sos = bessel(order, hi, btype="low", output="sos")
    else:
        raise ValueError(f"Unsupported lowpass type {ftype!r}")

    y = sosfiltfilt(sos, wav.astype(np.float64))
    y = _align_length(y, wav.shape[-1])
    fs_down = int(hi * SAMPLE_RATE)
    y2 = resample_poly(y, fs_down, SAMPLE_RATE)
    y2 = resample_poly(y2, SAMPLE_RATE, fs_down)
    y2 = _align_length(y2, wav.shape[-1])
    y2 = sosfiltfilt(sos, y2)
    return _align_length(y2, wav.shape[-1])


def _align_length(y: np.ndarray, target: int) -> np.ndarray:
    if y.shape[-1] == target:
        return y
    if y.shape[-1] < target:
        return np.pad(y, (0, target - y.shape[-1]))
    return y[:target]


def librosa_stft(y: np.ndarray) -> np.ndarray:
    """librosa 0.9.2 stft defaults: n_fft 2048, hop 512, hann, center,
    pad_mode='constant' (zeros - NOT reflect; verified against librosa 0.9.2)."""
    padded = np.pad(y.astype(np.float64), (POST_N_FFT // 2, POST_N_FFT // 2), mode="constant")
    frames = _frame(padded, POST_N_FFT, POST_HOP) * hann_periodic(POST_N_FFT)
    return np.fft.rfft(frames, axis=-1).T


def librosa_istft(stft_matrix: np.ndarray, length: int) -> np.ndarray:
    """librosa 0.9.2 istft: windowed overlap-add + window-sum-squares norm."""
    window = hann_periodic(POST_N_FFT)
    frames = np.fft.irfft(stft_matrix.T, n=POST_N_FFT, axis=-1) * window
    n_frames = frames.shape[0]
    expected = POST_N_FFT + POST_HOP * (n_frames - 1)
    y = np.zeros(expected, dtype=np.float64)
    win_sq_sum = np.zeros(expected, dtype=np.float64)
    win_sq = window**2
    for i in range(n_frames):
        start = i * POST_HOP
        y[start : start + POST_N_FFT] += frames[i]
        win_sq_sum[start : start + POST_N_FFT] += win_sq
    nonzero = win_sq_sum > np.finfo(np.float64).tiny
    y[nonzero] /= win_sq_sum[nonzero]
    y = y[POST_N_FFT // 2 :]
    return _align_length(y, length)


def replace_low_band_stft(restored: np.ndarray, lowpass_wav: np.ndarray) -> np.ndarray:
    """LatentDiffusion.postprocessing: anchor the low band to the input's STFT."""
    gt = lowpass_wav.astype(np.float64)
    length = restored.shape[-1]
    stft_gt_mag = np.abs(librosa_stft(gt))
    energy = np.cumsum(np.sum(stft_gt_mag, axis=-1))
    cutoff = find_cutoff_index(energy, 0.985)

    stft_gt = librosa_stft(gt)
    stft_out = librosa_stft(restored.astype(np.float64))
    energy_ratio = np.mean(
        np.sum(np.abs(stft_gt[cutoff])) / np.sum(np.abs(stft_out[cutoff, ...]))
    )
    energy_ratio = min(max(energy_ratio, 0.8), 1.2)
    stft_out[:cutoff, ...] = stft_gt[:cutoff, ...] / energy_ratio
    return librosa_istft(stft_out, length)
