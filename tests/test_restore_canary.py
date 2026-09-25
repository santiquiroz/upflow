from __future__ import annotations

import math

import numpy as np
import pytest

from app.services.engines.restore_canary import canary_rule_for, canary_score, delta_e_p99, has_non_finite, psnr_db


@pytest.mark.parametrize(
    ("model_id", "metric", "threshold"),
    [
        ("drunet-deblock-color", "psnr", 50.0),
        ("gfpgan-v1.4", "psnr", 45.0),
        ("restoreformer-pp", "psnr", 45.0),
        ("ddcolor-tiny", "delta_e", 2.0),
    ],
)
def test_canary_thresholds_follow_the_model_kind(model_id: str, metric: str, threshold: float) -> None:
    rule = canary_rule_for(model_id)

    assert (rule.metric, rule.threshold) == (metric, threshold)


def test_psnr_clamps_to_unit_range_and_identical_is_infinite() -> None:
    reference = np.zeros((4, 4, 3), dtype=np.float32)

    assert math.isinf(psnr_db(reference, reference))
    assert psnr_db(reference, reference - 5.0) == math.inf
    assert psnr_db(reference, reference + 0.1) == pytest.approx(20.0)


def test_delta_e_p99_measures_the_ab_distance() -> None:
    reference = np.zeros((10, 10, 2), dtype=np.float32)
    candidate = reference.copy()
    candidate[..., 0] = 3.0
    candidate[..., 1] = 4.0

    assert delta_e_p99(reference, candidate) == pytest.approx(5.0)
    assert not canary_rule_for("ddcolor-tiny").passes(5.0)


def test_canary_score_uses_the_metric_of_the_rule() -> None:
    reference = np.zeros((4, 4, 2), dtype=np.float32)
    candidate = reference + 0.1

    assert canary_score(canary_rule_for("gfpgan-v1.4"), reference, candidate) == pytest.approx(20.0)
    assert canary_score(canary_rule_for("ddcolor-tiny"), reference, candidate) == pytest.approx(0.1 * math.sqrt(2))


def test_rules_describe_scores_for_the_report() -> None:
    assert canary_rule_for("drunet-color").describe(math.inf) == "∞ dB"
    assert canary_rule_for("drunet-color").describe(13.04) == "13.0 dB"
    assert canary_rule_for("ddcolor-tiny").describe(1.14) == "ΔE p99 1.1"


def test_non_finite_detects_nan_and_inf() -> None:
    assert has_non_finite(np.array([0.0, np.inf]))
    assert has_non_finite(np.array([np.nan]))
    assert not has_non_finite(np.array([0.0, 1.0]))
