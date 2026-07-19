"""Vendored from santiquiroz/port-gmfss-onnx (public, MIT), pinned commit
1c202c3 (driver/ package -- Phases 0-3: FeatureNet/GMFlow/MetricNet/FusionNet
ONNX graph composition + numpy/torch-CPU softsplat driver, with an optional
OpenCL GPU softsplat kernel and automatic CPU fallback).

`pipeline.py` was re-synced from the prior pin (bebd393) to 1c202c3 for
Task 4.2: adds the `SplatFn` Protocol and `GmfssDriver`'s optional `splat_fn`
constructor parameter (default None -> CPU `splat_softmax`, same as before;
callers can now inject `softsplat_cl.splat_softmax` for the GPU path). No
other vendored file changed between the two pins (`assets.py`, `softsplat.py`,
`softsplat_cl.py`, `kernels/splat.cl` are identical).

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
