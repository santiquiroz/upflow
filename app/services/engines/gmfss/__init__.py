"""Vendored from santiquiroz/port-gmfss-onnx (public, MIT), pinned commit
bebd393 (driver/ package -- Phases 0-3: FeatureNet/GMFlow/MetricNet/FusionNet
ONNX graph composition + numpy/torch-CPU softsplat driver, with an optional
OpenCL GPU softsplat kernel and automatic CPU fallback).

Files are copied verbatim from the port repo's `driver/` package except for
one mechanical change: `from driver.X import ...` becomes
`from app.services.engines.gmfss.X import ...`, because this repo has no
top-level `driver` package. No other logic was modified. Same vendoring
convention as `app/services/engines/audiosr/` (vendored from
santiquiroz/port-audiosr-onnx).

To re-sync after a newer port commit: re-extract `driver/{assets,pipeline,
softsplat,softsplat_cl}.py` + `driver/kernels/splat.cl` from the port repo at
the desired commit (`git show <commit>:driver/<file>`), reapply the same
import rewrite, and re-run `tests/test_gmfss_engine.py` plus the port repo's
own `tests/test_pipeline.py` / `tests/test_softsplat.py` for parity.
"""

from __future__ import annotations
