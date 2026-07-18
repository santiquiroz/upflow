from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GRAPH_NAMES = ("vocoder", "vae_decoder", "vae_feature_extract", "ddpm")


@dataclass(frozen=True)
class AudioSrAssets:
    """Model directory contract produced by santiquiroz/port-audiosr-onnx."""

    model_dir: Path
    manifest: dict
    alphas_cumprod: np.ndarray
    mel_basis: np.ndarray

    @staticmethod
    def load(model_dir: Path) -> "AudioSrAssets":
        manifest = json.loads((model_dir / "manifest.json").read_text())
        alphas = np.load(model_dir / manifest["scheduler"]["alphas_cumprod_file"])
        basis = np.load(model_dir / manifest["mel"]["basis_file"])
        return AudioSrAssets(
            model_dir=model_dir,
            manifest=manifest,
            alphas_cumprod=alphas.astype(np.float64),
            mel_basis=basis.astype(np.float64),
        )

    @staticmethod
    def is_complete(model_dir: Path) -> bool:
        required = ["manifest.json", "alphas_cumprod.npy", "mel_basis.npy"]
        required += [f"{name}.onnx" for name in GRAPH_NAMES]
        return all((model_dir / name).exists() for name in required)

    def graph_path(self, name: str) -> Path:
        return self.model_dir / f"{name}.onnx"

    @property
    def scale_factor(self) -> float:
        return float(self.manifest["scale_factor"])

    @property
    def guidance_scale(self) -> float:
        return float(self.manifest["cfg"]["guidance_scale"])

    @property
    def unconditional_value(self) -> float:
        return float(self.manifest["cfg"]["unconditional_value"])
