"""Torch-free driver for the exported RoFormer graphs.

The graph is only the middle of the pipeline: `spec [1, F*C, T, 2]` in, complex
`mask [1, N, F*C, T, 2]` out. Everything else -- STFT, the complex multiply, the
DC filter, the iSTFT and the overlapping-chunk overlap-add -- lives here, in
numpy, and mirrors MSST's reference inference path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from app.services.engines.roformer.chunking import iter_chunks, pad_mix, plan_chunks, unpad_result
from app.services.engines.roformer.stft import istft_bands_last, stft_bands_last

RunGraph = Callable[[np.ndarray], np.ndarray]
ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class RoformerSpec:
    """The handful of numbers the driver needs; all come from the model config."""

    n_fft: int = 2048
    hop_length: int = 441
    chunk_size: int = 352800
    num_overlap: int = 2
    audio_channels: int = 2
    sample_rate: int = 44100
    stems: Sequence[str] = ("vocals",)
    # MSST zeroes the DC bin of the masked spectrogram before the iSTFT.
    zero_dc: bool = True

    @property
    def num_stems(self) -> int:
        return len(self.stems)

    @property
    def frames(self) -> int:
        return self.chunk_size // self.hop_length + 1


def complex_multiply(spec: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(a+bi)(c+di) on trailing-dim-2 real pairs. spec [..., 2], mask [..., 2]."""
    ar, ai = spec[..., 0], spec[..., 1]
    br, bi = mask[..., 0], mask[..., 1]
    return np.stack([ar * br - ai * bi, ar * bi + ai * br], axis=-1)


def apply_zero_dc(masked: np.ndarray, audio_channels: int) -> np.ndarray:
    """Zero the DC bin for every channel; layout is (freq, channel) interleaved."""
    masked[..., :audio_channels, :, :] = 0.0
    return masked


class RoformerDriver:
    """Vendorable: numpy + a callable that runs one graph invocation."""

    def __init__(self, run_graph: RunGraph, spec: RoformerSpec = RoformerSpec()):
        self.run_graph = run_graph
        self.spec = spec

    def separate_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """[C, chunk_size] -> [num_stems, C, chunk_size]."""
        s = self.spec
        spec = stft_bands_last(chunk, s.n_fft, s.hop_length)
        mask = np.asarray(self.run_graph(spec))
        masked = complex_multiply(spec[:, None], mask)[0]  # [N, F*C, T, 2]
        if s.zero_dc:
            masked = apply_zero_dc(masked, s.audio_channels)
        return np.stack(
            [
                istft_bands_last(
                    masked[n], s.audio_channels, s.n_fft, s.hop_length, chunk.shape[1]
                )
                for n in range(masked.shape[0])
            ]
        )

    def separate(
        self, mix: np.ndarray, on_chunk: ProgressCallback | None = None
    ) -> np.ndarray:
        """[C, N] float -> [num_stems, C, N] float, overlap-added across chunks."""
        if mix.ndim == 1:
            mix = np.repeat(mix[None], self.spec.audio_channels, axis=0)
        s = self.spec
        plan = plan_chunks(mix.shape[1], s.chunk_size, s.num_overlap)
        padded = pad_mix(np.asarray(mix, dtype=np.float64), plan)
        shape = (s.num_stems, *padded.shape)
        result = np.zeros(shape)
        counter = np.zeros(shape)
        done = 0
        for start, seg_len, part, window in iter_chunks(padded, plan):
            stems = self.separate_chunk(part.astype(np.float32))
            result[..., start : start + seg_len] += stems[..., :seg_len] * window[:seg_len]
            counter[..., start : start + seg_len] += window[:seg_len]
            done += 1
            if on_chunk is not None:
                on_chunk(done, plan.num_chunks)
        return unpad_result(result, counter, plan)
