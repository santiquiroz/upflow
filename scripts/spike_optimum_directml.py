"""Spike: valida optimum + onnxruntime-directml. Uso: python scripts/spike_optimum_directml.py <pipeline_dir> [device]"""
from __future__ import annotations

import sys
import time
from pathlib import Path


def main() -> int:
    pipeline_dir = Path(sys.argv[1])
    device = sys.argv[2] if len(sys.argv) > 2 else "dml:0"

    import onnxruntime as ort
    import torch

    print(f"onnxruntime {ort.__version__}, providers: {ort.get_available_providers()}")

    from optimum.onnxruntime import ORTStableDiffusionPipeline  # ajustar si Step 3 dijo otra clase

    if device.startswith("dml:"):
        kwargs = {"provider": "DmlExecutionProvider", "provider_options": {"device_id": int(device.split(":")[1])}}
    else:
        kwargs = {"provider": "CPUExecutionProvider"}

    t0 = time.perf_counter()
    pipe = ORTStableDiffusionPipeline.from_pretrained(str(pipeline_dir), **kwargs)
    print(f"load: {time.perf_counter() - t0:.1f}s")

    steps_seen: list[int] = []

    def on_step(step: int, timestep, latents) -> None:
        steps_seen.append(step)

    t0 = time.perf_counter()
    result = pipe(
        prompt="a red apple on a wooden table",
        negative_prompt="blurry",
        num_inference_steps=4,
        guidance_scale=7.5,
        width=256,
        height=256,
        callback=on_step,
        callback_steps=1,
        # diffusers' randn_tensor reads generator.device.type; np.random.RandomState lacks
        # that attribute and raises AttributeError (verified empirically against this pin).
        generator=torch.Generator(device="cpu").manual_seed(42),
    )
    print(f"infer: {time.perf_counter() - t0:.1f}s, callback steps: {steps_seen}")
    out = Path("spike_output.png")
    result.images[0].save(out)
    print(f"saved {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
