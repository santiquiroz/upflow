"""Vendored from santiquiroz/port-bs-roformer-onnx (public, MIT), pinned commit
f5415a3 (`driver/` package: torch-free numpy pre/post around the exported
Mel-Band RoFormer / BS-RoFormer ONNX graphs -- STFT, the complex mask multiply,
the DC filter, the iSTFT and MSST's overlapping-chunk overlap-add).

f5415a3 is the commit whose tree these files were extracted from
(`git show f5415a3:driver/<file>`), and it is the one whose test suite
(`tests/test_driver.py`, `tests/test_stft.py`) and parity harness
(`toolkit/validate_ort.py`: mask p99.9 < 1e-4 and RMS < 1e-5 vs unpatched
upstream MSST torch, synth SI-SDR 138 dB, separation quality +0.29 dB over the
reference, gated on BOTH the CPU EP and DirectML) gate this code.

The graph is only the MIDDLE of the pipeline: ONNX has neither a complex dtype
nor `istft`, so the transform ends are amputated at export and live here
instead. `spec [1, F*C, T, 2]` in, complex `mask [1, N, F*C, T, 2]` out.

THE DIRECTML GLU FIX IS IN THE GRAPH, NOT HERE. The port found that
`Split(2) -> Sigmoid -> Mul` (what `F.glu` exports as) is miscomputed by the
DirectML EP -- it collapses to `a*sigmoid(a)`, so the audio is garbage on GPU
and correct on CPU. The fix replaces `nn.GLU` with two `Slice`s BEFORE tracing
(`toolkit/spec_models.py`), so it travels with the .onnx file, not with this
driver. The published graph records it: `manifest.json` of the `models-v1.0`
release carries `export_patches.glu_replaced = 60`, and the catalog pins the
graph's SHA-256, so a graph without the fix cannot be the one that gets loaded.
Nothing in this package can compensate for a graph exported without it.

Files are copied verbatim from the port repo's `driver/` package except for one
mechanical change: `from driver.X import ...` becomes
`from app.services.engines.roformer.X import ...`, because this repo has no
top-level `driver` package. No other logic was modified. Same vendoring
convention as `app/services/engines/gmfss/` (santiquiroz/port-gmfss-onnx),
`app/services/engines/audiosr/` (santiquiroz/port-audiosr-onnx) and
`app/services/engines/vr_deecho/` (santiquiroz/port-uvr-deecho-onnx).

CHANGES GO UPSTREAM. Anything wrong with the DSP is fixed in
santiquiroz/port-bs-roformer-onnx and re-vendored -- never patched here, or the
parity gates in that repo stop describing what Upflow actually runs.

To re-sync after a newer port commit: re-extract `driver/{chunking,stft,
pipeline}.py` at the desired commit (`git show <commit>:driver/<file>` -- never
the live working tree), reapply the same import rewrite, bump the pin in these
headers, and re-run `tests/test_roformer_separator.py` plus the port repo's own
`tests/` for parity.

Two properties of this architecture an integrator has to respect, both gated by
`tests/test_roformer_separator.py`:

* The time axis is FIXED at trace time (the rotary embedding caches its
  frequency table per sequence length), so `RoformerSpec.chunk_size` is not a
  tunable -- it has to match the graph that was exported. Read it out of
  `manifest.json` for any other export.
* These checkpoints predict ONE stem (`vocals`); the other one is `mix - vocals`
  with NO compensation factor, unlike MDX-Net.

The ONNX session itself is NOT vendored: Upflow builds it through
`app.services.ep_registry` like every other engine, and hands the driver a
`run_graph` callable. That seam is also where cancellation and per-chunk
progress hook in -- see `app/services/engines/roformer_separator.py`.
"""

from __future__ import annotations

from app.services.engines.roformer.pipeline import RoformerDriver, RoformerSpec

__all__ = ["RoformerDriver", "RoformerSpec"]
