# Vendored from santiquiroz/port-uvr-deecho-onnx driver/pipeline.py @ commit
# 02cd199 (see app/services/engines/vr_deecho/__init__.py for sync notes).
# Imports rewritten: `from driver import multiband` -> `from
# app.services.engines.vr_deecho import multiband`, `from driver.vr_params
# import ...` -> `from app.services.engines.vr_deecho.vr_params import ...`
# (this repo has no top-level `driver` package). No other change.
"""ONNX driver for the UVR VR De-Echo / De-Reverb / De-Noise models (FoxJoy, 4band_v3).

Torch-free: numpy + scipy pre/post around an onnxruntime session the caller owns.
The ONNX graphs take one magnitude window [1, 2, 673, 512] and return the full
sigmoid mask [1, 2, 673, 512]; this driver does everything around that --
multiband STFT analysis, global-max normalization, 384-frame ROI windowing with
64-frame edge crop, aggression curve, complex masking, multiband iSTFT synthesis.

Mirrors python-audio-separator's VRSeparator.inference_vr (MIT) with the
defaults that reference uses for these models: window_size=512, batch_size=1,
aggression=5, no TTA, no post-process, no high-end mirroring.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from app.services.engines.vr_deecho import multiband
from app.services.engines.vr_deecho.vr_params import AGGR_SPLIT_BIN, OFFSET, WINDOW_SIZE

RunGraph = Callable[[np.ndarray], np.ndarray]

# is_non_accom_stem mirrors the reference's `primary_stem in NON_ACCOM_STEMS`
# test (common_separator.py). It flips the aggression exponent to `1 - aggr`,
# so it MUST match the model's UVR primary_stem or the mask is wrong.
# De-Echo/De-Reverb are "No Other" (not in that tuple); DeNoise is "Other",
# which also means its mask isolates the NOISE -- primary/secondary are
# swapped relative to the De-Echo family.
MODEL_SPECS = {
    "UVR-De-Echo-Normal": {
        "primary_stem": "No Echo", "secondary_stem": "Echo", "nout": 48,
        "uvr_primary_stem": "No Other", "is_non_accom_stem": False,
    },
    "UVR-De-Echo-Aggressive": {
        "primary_stem": "No Echo", "secondary_stem": "Echo", "nout": 48,
        "uvr_primary_stem": "No Other", "is_non_accom_stem": False,
    },
    "UVR-DeEcho-DeReverb": {
        "primary_stem": "No Reverb", "secondary_stem": "Reverb", "nout": 64,
        "uvr_primary_stem": "No Other", "is_non_accom_stem": False,
    },
    "UVR-DeNoise": {
        "primary_stem": "Noise", "secondary_stem": "No Noise", "nout": 48,
        "uvr_primary_stem": "Other", "is_non_accom_stem": True,
    },
}


def make_padding(n_frames: int) -> tuple[int, int, int]:
    roi_size = WINDOW_SIZE - OFFSET * 2
    pad_right = roi_size - (n_frames % roi_size) + OFFSET
    return OFFSET, pad_right, roi_size


def predict_mask(mag_padded: np.ndarray, roi_size: int, run_graph: RunGraph) -> np.ndarray:
    patches = (mag_padded.shape[2] - 2 * OFFSET) // roi_size
    rois = []
    for i in range(patches):
        start = i * roi_size
        window = mag_padded[None, :, :, start : start + WINDOW_SIZE]
        mask = run_graph(np.ascontiguousarray(window))[0]
        rois.append(mask[:, :, OFFSET:-OFFSET])
    return np.concatenate(rois, axis=2)


def adjust_aggression(mask: np.ndarray, aggression: float, is_non_accom_stem: bool = False) -> np.ndarray:
    aggr = (aggression / 100.0) * 2.0
    if aggr == 0:
        return mask
    if is_non_accom_stem:
        aggr = 1 - aggr
    adjusted = mask.copy()
    adjusted[:, :AGGR_SPLIT_BIN] = np.power(adjusted[:, :AGGR_SPLIT_BIN], 1 + aggr / 3)
    adjusted[:, AGGR_SPLIT_BIN:] = np.power(adjusted[:, AGGR_SPLIT_BIN:], 1 + aggr)
    return adjusted


class DeEchoDriver:
    def __init__(self, run_graph: RunGraph, aggression: float = 5.0, is_non_accom_stem: bool = False):
        self.run_graph = run_graph
        self.aggression = aggression
        self.is_non_accom_stem = is_non_accom_stem

    def infer_mask(self, spec: np.ndarray) -> np.ndarray:
        mag = np.abs(spec)
        n_frames = mag.shape[2]
        pad_left, pad_right, roi_size = make_padding(n_frames)
        mag_padded = np.pad(mag, ((0, 0), (0, 0), (pad_left, pad_right)))
        mag_padded /= mag_padded.max()
        mask = predict_mask(mag_padded.astype(np.float32), roi_size, self.run_graph)
        return mask[:, :, :n_frames]

    def separate_spec(self, mix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        spec = multiband.wave_to_combined_spec(mix)
        mask = adjust_aggression(self.infer_mask(spec), self.aggression, self.is_non_accom_stem)
        mag, phase = np.abs(spec), np.angle(spec)
        primary_spec = mask * mag * np.exp(1.0j * phase)
        secondary_spec = (1 - mask) * mag * np.exp(1.0j * phase)
        return primary_spec, secondary_spec, mask

    def separate(self, mix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        primary_spec, secondary_spec, _ = self.separate_spec(mix)
        primary = multiband.combined_spec_to_wave(np.nan_to_num(primary_spec, nan=0.0, posinf=0.0, neginf=0.0))
        secondary = multiband.combined_spec_to_wave(np.nan_to_num(secondary_spec, nan=0.0, posinf=0.0, neginf=0.0))
        return primary, secondary
