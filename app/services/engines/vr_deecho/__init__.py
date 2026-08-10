"""Vendored from santiquiroz/port-uvr-deecho-onnx (public, MIT), pinned commit
02cd199 (`driver/` package: torch-free numpy/scipy pre/post around the four
FoxJoy VR De-Echo/De-Reverb/De-Noise ONNX graphs -- 4band_v3 multiband STFT
analysis, global-max normalization, 384-frame ROI windowing with 64-frame edge
crop, the aggression curve, complex masking and multiband iSTFT synthesis).

02cd199 is the commit whose tree these files were extracted from
(`git show 02cd199:driver/<file>`), and it is the one whose test suite
(`tests/test_dsp.py`, `tests/test_aggression.py`,
`tests/test_driver_torch_free.py`) and parity harness
(`toolkit/validate_ort.py`: mask p99.9 < 1e-4, stems SI-SDR 61-65 dB vs
python-audio-separator) gate this code.

Last upstream change to `driver/` was f382066, which added UVR-DeNoise: it
introduced the `is_non_accom_stem` argument on `DeEchoDriver`/
`adjust_aggression` (flips the aggression exponent to `1 - aggr` for models
whose UVR primary_stem is in NON_ACCOM_STEMS) and the `uvr_primary_stem` /
`is_non_accom_stem` / `secondary_stem` keys in MODEL_SPECS. Only De-Noise sets
it; the three De-Echo/De-Reverb models keep the previous behaviour, and the
default is False so the old call shape still works.

Files are copied verbatim from the port repo's `driver/` package except for one
mechanical change: `from driver.X import ...` becomes
`from app.services.engines.vr_deecho.X import ...`, because this repo has no
top-level `driver` package. No other logic was modified. Same vendoring
convention as `app/services/engines/gmfss/` (santiquiroz/port-gmfss-onnx) and
`app/services/engines/audiosr/` (santiquiroz/port-audiosr-onnx).

CHANGES GO UPSTREAM. Anything wrong with the DSP is fixed in
santiquiroz/port-uvr-deecho-onnx and re-vendored -- never patched here, or the
parity gates in that repo stop describing what Upflow actually runs.

To re-sync after a newer port commit: re-extract `driver/{vr_params,dsp,
multiband,pipeline}.py` at the desired commit (`git show <commit>:driver/<file>`
-- never the live working tree), reapply the same import rewrite, bump the pin
in these headers, and re-run `tests/test_vr_deecho_separator.py` plus the port
repo's own `tests/` for parity.

If a re-sync adds a model, check MODEL_SPECS for a NEW `is_non_accom_stem` /
inverted stem pair before wiring it into `vr_models.py`: which stem is the
clean one is per-model and cannot be read off the graph.

The ONNX session itself is NOT vendored: Upflow builds it through
`app.services.ep_registry` like every other engine, and hands the driver a
`run_graph` callable. That seam is also where cancellation and per-window
progress hook in -- see `app/services/engines/vr_deecho_separator.py`.
"""

from __future__ import annotations

from app.services.engines.vr_deecho.pipeline import MODEL_SPECS, DeEchoDriver

__all__ = ["DeEchoDriver", "MODEL_SPECS"]
