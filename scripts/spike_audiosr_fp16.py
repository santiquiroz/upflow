"""Does the fp16 recipe that gave 12x on RoFormer do anything for AudioSR?

AudioSR is Upflow's heaviest audio model: four graphs, 2.6 GB, and the ddpm
UNet runs once per DDIM step per window. The RoFormer port showed fp16 pays off
exactly when a graph is memory bound, so this measures whether that applies
here -- speed AND how far the fp16 output drifts from the fp32 one, since there
is no ground truth to score against.

    python scripts/spike_audiosr_fp16.py convert <src_dir> <dst_dir>
    python scripts/spike_audiosr_fp16.py run <model_dir> <out.npy> [--steps N] [--device dml:0]
    python scripts/spike_audiosr_fp16.py compare <a.npy> <b.npy>
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

GRAPHS = ("vocoder", "vae_decoder", "vae_feature_extract", "ddpm")
# Same RMSNorm/GroupNorm block list the RoFormer port needed: Pow squares the
# activations and ReduceMean sums the squares past fp16's 65504 ceiling.
BLOCK = ["Pow", "ReduceMean", "Sqrt", "Div"]


def convert(src: Path, dst: Path) -> None:
    import onnx
    from onnxruntime.transformers import float16
    from onnxruntime.transformers.onnx_model import OnnxModel

    dst.mkdir(parents=True, exist_ok=True)
    for extra in ("manifest.json", "alphas_cumprod.npy", "mel_basis.npy"):
        shutil.copy2(src / extra, dst / extra)

    for name in GRAPHS:
        out = dst / f"{name}.onnx"
        # onnx.save appends to an existing external-data file instead of truncating.
        out.unlink(missing_ok=True)
        (dst / f"{name}.onnx.data").unlink(missing_ok=True)
        started = time.perf_counter()
        model = onnx.load(str(src / f"{name}.onnx"))
        converted = float16.convert_float_to_float16(
            model, keep_io_types=True, op_block_list=BLOCK
        )
        del converted.graph.value_info[:]
        wrapper = OnnxModel(converted)
        wrapper.topological_sort()
        onnx.save(
            wrapper.model,
            str(out),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=out.name + ".data",
        )
        before = _bytes_of(src, name)
        after = _bytes_of(dst, name)
        print(f"{name:22s} {before/2**20:8.1f} -> {after/2**20:8.1f} MiB "
              f"({100*(1-after/before):.1f}% smaller, {time.perf_counter()-started:.1f}s)")


def _bytes_of(folder: Path, name: str) -> int:
    total = (folder / f"{name}.onnx").stat().st_size
    data = folder / f"{name}.onnx.data"
    return total + (data.stat().st_size if data.exists() else 0)


def run(model_dir: Path, out_path: Path, steps: int, device: str) -> None:
    from app.config import Settings
    from app.services import ep_registry
    from app.services.engines.audiosr.assets import AudioSrAssets
    from app.services.engines.audiosr.driver import AudioSrDriver
    from app.services.engines.audiosr_restore import _session_runner

    settings = Settings(_env_file=None)
    assets = AudioSrAssets.load(model_dir)

    started = time.perf_counter()
    sessions = {
        name: ep_registry.create_session(str(model_dir / f"{name}.onnx"), device, settings)
        for name in GRAPHS
    }
    load_s = time.perf_counter() - started

    mono = _clip()
    driver = AudioSrDriver(assets, _session_runner(sessions))
    started = time.perf_counter()
    restored = driver.restore(mono, ddim_steps=steps)
    run_s = time.perf_counter() - started

    np.save(out_path, restored)
    print(f"sessions {load_s:6.1f}s | restore {run_s:6.1f}s for {len(mono)/24000:.1f}s of audio "
          f"({steps} steps) -> {out_path}")


def _clip() -> np.ndarray:
    """Deterministic 5.12 s at 24 kHz: harmonics plus noise bursts, band-limited.

    Not music, but AudioSR's job is inventing high band from a band-limited
    input, so the input has to actually lack the top octave for the comparison
    to exercise the model rather than a passthrough.
    """
    rng = np.random.default_rng(20260817)
    sr, seconds = 24000, 5.12
    t = np.arange(int(sr * seconds)) / sr
    tone = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate((220.0, 440.0, 660.0)))
    env = np.exp(-((t % 0.64) / 0.25))
    noise = rng.standard_normal(t.size) * 0.05
    wav = (tone * env + noise) * 0.2
    return wav.astype(np.float32)


def compare(a_path: Path, b_path: Path) -> None:
    a = np.load(a_path).astype(np.float64).ravel()
    b = np.load(b_path).astype(np.float64).ravel()
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    noise = a - b
    sisdr = 10 * np.log10(np.dot(a, a) / max(np.dot(noise, noise), 1e-20))
    print(f"samples={n} | fp16 vs fp32 SI-SDR = {sisdr:.1f} dB | "
          f"max abs diff = {np.abs(noise).max():.4f} | peak = {np.abs(a).max():.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert"); c.add_argument("src"); c.add_argument("dst")
    r = sub.add_parser("run"); r.add_argument("model_dir"); r.add_argument("out")
    r.add_argument("--steps", type=int, default=50); r.add_argument("--device", default="dml:0")
    m = sub.add_parser("compare"); m.add_argument("a"); m.add_argument("b")
    args = ap.parse_args()

    if args.cmd == "convert":
        convert(Path(args.src), Path(args.dst))
    elif args.cmd == "run":
        run(Path(args.model_dir), Path(args.out), args.steps, args.device)
    else:
        compare(Path(args.a), Path(args.b))


if __name__ == "__main__":
    main()
