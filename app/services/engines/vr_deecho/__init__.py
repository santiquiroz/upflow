"""Vendored from santiquiroz/port-uvr-deecho-onnx (public, MIT), pinned commit
23b2564 (`driver/` package: torch-free numpy/scipy pre/post around the three
FoxJoy VR De-Echo/De-Reverb ONNX graphs -- 4band_v3 multiband STFT analysis,
global-max normalization, 384-frame ROI windowing with 64-frame edge crop, the
aggression curve, complex masking and multiband iSTFT synthesis).

`driver/` has not changed since commit ae8bb7c in that repo; 23b2564 is the pin
because it is the commit whose tree these files were extracted from
(`git show 23b2564:driver/<file>`), and it is the one whose test suite
(`tests/test_dsp.py`, `tests/test_driver_torch_free.py`) and parity harness
(`toolkit/validate_ort.py`: mask p99.9 < 1e-4, stems SI-SDR 61-65 dB vs
python-audio-separator) gate this code.

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

The ONNX session itself is NOT vendored: Upflow builds it through
`app.services.ep_registry` like every other engine, and hands the driver a
`run_graph` callable. That seam is also where cancellation and per-window
progress hook in -- see `app/services/engines/vr_deecho_separator.py`.
"""

from __future__ import annotations

from app.services.engines.vr_deecho.pipeline import MODEL_SPECS, DeEchoDriver

__all__ = ["DeEchoDriver", "MODEL_SPECS"]
