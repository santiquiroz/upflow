from __future__ import annotations

import threading
from typing import Callable, Protocol

import numpy as np

from app.services.engines.audiosr.assets import AudioSrAssets
from app.services.engines.audiosr.ddim import DdimSchedule, combine_cfg, ddim_step
from app.services.engines.audiosr import dsp

# Torch-free reimplementation of audiosr's generate_batch. The neural graphs
# run through an injected run_graph callable (onnxruntime sessions in prod,
# fakes in tests); everything else is numpy and parity-tested against the
# instrumented PyTorch baseline (port-audiosr-onnx/refs/baseline).

WINDOW_SECONDS = 10.24          # native training window (latent T=128)
OVERLAP_SECONDS = 1.28          # wav-domain Hann crossfade between windows
PAD_UNIT_SECONDS = 5.12
FRAMES_PER_SECOND = 100
LATENT_DOWNSAMPLE = 8


class GraphRunner(Protocol):
    def __call__(self, name: str, feeds: dict[str, np.ndarray]) -> np.ndarray: ...


class AudioSrCancelled(Exception):
    pass


NoiseSource = Callable[[tuple[int, ...]], np.ndarray]


def _default_noise_source(rng: np.random.Generator) -> NoiseSource:
    def source(shape: tuple[int, ...]) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float32)

    return source


class AudioSrDriver:
    def __init__(
        self,
        assets: AudioSrAssets,
        run_graph: GraphRunner,
        noise_source: NoiseSource | None = None,
        seed: int = 42,
    ) -> None:
        self.assets = assets
        self.run_graph = run_graph
        self.noise_source = noise_source or _default_noise_source(np.random.default_rng(seed))

    def restore(
        self,
        wav48k: np.ndarray,
        ddim_steps: int = 50,
        guidance_scale: float | None = None,
        lowpass_type: str = "butter",
        progress_cb: Callable[[int, int], None] | None = None,
        cancel_event: threading.Event | None = None,
        step_throttle: Callable[[], None] | None = None,
    ) -> np.ndarray:
        """Any-band 48 kHz mono float -> restored 48 kHz mono float32 (same length)."""
        original_length = wav48k.shape[-1]
        wav = dsp.normalize_wav(np.asarray(wav48k, dtype=np.float64))
        wav = _pad_to_unit(wav)

        stft_full = dsp.stft_magnitude(wav)
        cutoff_hz = dsp.detect_cutoff_hz(dsp.pad_spec(stft_full, _frames_for(wav)))
        wav_lp = dsp.lowpass_simulate(wav, cutoff_hz, lowpass_type).astype(np.float32)

        windows = _split_windows(wav.shape[-1])
        guidance = self.assets.guidance_scale if guidance_scale is None else guidance_scale
        schedule = DdimSchedule.build(self.assets.alphas_cumprod, ddim_steps)

        total_steps = len(windows) * ddim_steps
        done_steps = 0
        restored_windows = []
        for start, end in windows:
            def on_step(_done_in_window: int) -> None:
                nonlocal done_steps
                done_steps += 1
                if progress_cb is not None:
                    progress_cb(done_steps, total_steps)

            restored = self._restore_window(
                wav[start:end], wav_lp[start:end], schedule, guidance,
                on_step, cancel_event, step_throttle,
            )
            restored_windows.append(restored)

        # The merge + STFT postproc scale with clip length, so a cancel that
        # lands after the last DDIM step must not pay for them.
        _raise_if_cancelled(cancel_event)
        merged = _crossfade_concat(restored_windows, wav.shape[-1])
        anchored = dsp.replace_low_band_stft(merged, wav_lp)
        return _final_normalize(anchored)[:original_length]

    def _restore_window(
        self,
        wav: np.ndarray,
        wav_lp: np.ndarray,
        schedule: DdimSchedule,
        guidance: float,
        on_step: Callable[[int], None],
        cancel_event: threading.Event | None,
        step_throttle: Callable[[], None] | None,
    ) -> np.ndarray:
        target_frames = _frames_for(wav)
        mel_lp = dsp.pad_spec(
            dsp.log_mel(dsp.stft_magnitude(wav_lp), self.assets.mel_basis), target_frames
        ).astype(np.float32)

        latent_t = target_frames // LATENT_DOWNSAMPLE
        latent_shape = (1, 16, latent_t, 32)

        cond = self.run_graph("vae_feature_extract", {
            "mel": mel_lp[None, None, ...],
            "noise": self.noise_source(latent_shape),
        }).astype(np.float32)
        uncond = np.full_like(cond, self.assets.unconditional_value)

        scale = np.float32(self.assets.scale_factor)
        cond_in = cond * scale
        uncond_in = uncond * scale

        x = self.noise_source(latent_shape).astype(np.float32)
        for i, t in enumerate(reversed(schedule.timesteps)):
            _raise_if_cancelled(cancel_event)
            index = len(schedule.timesteps) - i - 1
            timesteps = np.array([t], dtype=np.int64)
            v_cond = self.run_graph("ddpm", {
                "x": np.concatenate([x, cond_in], axis=1), "timesteps": timesteps,
            })
            v_uncond = self.run_graph("ddpm", {
                "x": np.concatenate([x, uncond_in], axis=1), "timesteps": timesteps,
            })
            v = combine_cfg(v_cond, v_uncond, guidance).astype(np.float32)
            noise = self.noise_source(latent_shape).astype(np.float32)
            x = ddim_step(x, v, int(t), index, schedule, noise).astype(np.float32)
            on_step(i + 1)
            if step_throttle is not None:
                step_throttle()

        _raise_if_cancelled(cancel_event)
        mel_out = self.run_graph("vae_decoder", {"z": x}).astype(np.float32)

        cutoff_melbin = dsp.locate_cutoff_bin(np.exp(mel_lp.astype(np.float64)), 0.985)
        mel_out[0, 0, :, :cutoff_melbin] = mel_lp[:, :cutoff_melbin]

        wav_out = self.run_graph("vocoder", {"mel": mel_out[0].transpose(0, 2, 1)})
        return np.asarray(wav_out, dtype=np.float64).reshape(-1)[: wav.shape[-1]]


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise AudioSrCancelled()


def _frames_for(wav: np.ndarray) -> int:
    return int(round(wav.shape[-1] / dsp.SAMPLE_RATE * FRAMES_PER_SECOND))


def _pad_to_unit(wav: np.ndarray) -> np.ndarray:
    unit = int(PAD_UNIT_SECONDS * dsp.SAMPLE_RATE)
    remainder = wav.shape[-1] % unit
    if remainder == 0:
        return wav
    return np.pad(wav, (0, unit - remainder))


def _split_windows(total_samples: int) -> list[tuple[int, int]]:
    window = int(WINDOW_SECONDS * dsp.SAMPLE_RATE)
    overlap = int(OVERLAP_SECONDS * dsp.SAMPLE_RATE)
    if total_samples <= window:
        return [(0, total_samples)]
    hop = window - overlap
    starts = list(range(0, total_samples - overlap, hop))
    windows = []
    for start in starts:
        end = min(start + window, total_samples)
        windows.append((start, end))
        if end >= total_samples:
            break
    return windows


def _crossfade_concat(windows: list[np.ndarray], total_samples: int) -> np.ndarray:
    if len(windows) == 1:
        return windows[0]
    window_len = int(WINDOW_SECONDS * dsp.SAMPLE_RATE)
    overlap = int(OVERLAP_SECONDS * dsp.SAMPLE_RATE)
    hop = window_len - overlap
    out = np.zeros(total_samples, dtype=np.float64)
    weight = np.zeros(total_samples, dtype=np.float64)
    for i, chunk in enumerate(windows):
        start = i * hop
        ramp = np.ones(chunk.shape[-1], dtype=np.float64)
        if i > 0:
            n = min(overlap, chunk.shape[-1])
            ramp[:n] = np.sin(0.5 * np.pi * np.linspace(0, 1, n)) ** 2
        if i < len(windows) - 1:
            n = min(overlap, chunk.shape[-1])
            ramp[-n:] = np.cos(0.5 * np.pi * np.linspace(0, 1, n)) ** 2
        out[start : start + chunk.shape[-1]] += chunk * ramp
        weight[start : start + chunk.shape[-1]] += ramp
    return out / np.maximum(weight, 1e-8)


def _final_normalize(wav: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(wav))
    wav = 0.5 * wav / (peak if peak > 0 else 1.0)
    return (wav - np.mean(wav)).astype(np.float32)
