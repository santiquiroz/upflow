from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

PSNR = "psnr"
DELTA_E = "delta_e"


@dataclass(frozen=True, slots=True)
class CanaryRule:
    metric: str
    threshold: float

    def passes(self, score: float) -> bool:
        if self.metric == PSNR:
            return score >= self.threshold
        return score <= self.threshold

    def describe(self, score: float) -> str:
        if self.metric == PSNR:
            return f"{_format(score)} dB"
        return f"ΔE p99 {_format(score)}"


# Umbrales del paso 8 de §6.1: los mismos para el export (proxy en CPU) y para el canario en GPU.
CANARY_RULES = {
    "restoration": CanaryRule(PSNR, 50.0),
    "faces": CanaryRule(PSNR, 45.0),
    "color": CanaryRule(DELTA_E, 2.0),
}
_KIND_PREFIXES = (("gfpgan", "faces"), ("restoreformer", "faces"), ("ddcolor", "color"))


def canary_rule_for(model_id: str) -> CanaryRule:
    kind = next((kind for prefix, kind in _KIND_PREFIXES if model_id.startswith(prefix)), "restoration")
    return CANARY_RULES[kind]


def psnr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    diff = np.clip(reference, 0.0, 1.0) - np.clip(candidate, 0.0, 1.0)
    mse = float(np.mean(np.square(diff, dtype=np.float64)))
    return math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def delta_e_p99(reference: np.ndarray, candidate: np.ndarray) -> float:
    # Con L identico, el ΔE entre dos predicciones ab es su distancia en el plano ab.
    distance = np.sqrt(np.sum(np.square(reference - candidate, dtype=np.float64), axis=-1))
    return float(np.percentile(distance, 99))


def canary_score(rule: CanaryRule, reference: np.ndarray, candidate: np.ndarray) -> float:
    if rule.metric == PSNR:
        return psnr_db(reference, candidate)
    return delta_e_p99(reference, candidate)


def has_non_finite(output: np.ndarray) -> bool:
    return not bool(np.isfinite(output).all())


def _format(score: float) -> str:
    return "∞" if math.isinf(score) else f"{score:.1f}"
