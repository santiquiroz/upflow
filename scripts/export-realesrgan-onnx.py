"""Export the builtin Real-ESRGAN models to uint8-in/out ONNX graphs (SP11).

Each exported graph bakes pre/post INTO the graph: input uint8 NHWC [1,H,W,3]
-> /255 -> NCHW -> network -> *255 -> clamp(0,255) -> round -> uint8 -> NHWC
[1,sH,sW,3]. This is what makes the ONNX video backend ~2.1x faster than NCNN
on DirectML: a frame is a raw uint8 array in and out, so there is no per-frame
numpy pre/post and no fp32 readback.

Weights come from the official xinntao/Real-ESRGAN PyTorch releases (download
them with scripts/download-realesrgan-onnx.ps1, which then runs this script).

Only realesr-animevideov3 ships x4 PyTorch weights officially; the x2/x3
variants are derived by baking an area-resample of the x4 output down to the
target ratio into the graph (this mirrors Real-ESRGAN's `outscale` behavior).

opset 17, fp32 (fp16 did not help in the benchmark).

Usage:
    python scripts/export-realesrgan-onnx.py \
        --weights-dir vendor/realesrgan-onnx/weights \
        --out-dir vendor/realesrgan-onnx
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

OPSET = 17


# ---------------------------------------------------------------------------
# Architectures (BasicSR-compatible, matching the official .pth state dicts)
# ---------------------------------------------------------------------------


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# ---------------------------------------------------------------------------
# uint8 pre/post wrapper (baked into the graph)
# ---------------------------------------------------------------------------


class Uint8Wrapper(nn.Module):
    """uint8 NHWC in -> uint8 NHWC out. `target_scale` differs from the base
    network scale only for the derived animevideov3 x2/x3 graphs, which resample
    the x4 output down to the requested ratio (Real-ESRGAN `outscale` behavior).

    `half` runs the network body in fp16. The body dominates runtime (it works at
    INPUT resolution regardless of scale), and on a GPU with double-rate fp16
    (RDNA3, Ampere+) that measured 1.33x faster at 1080p->4x with a max pixel
    delta of 3/255 vs fp32. I/O stays uint8 either way.
    """

    def __init__(self, net: nn.Module, base_scale: int, target_scale: int, half: bool = False):
        super().__init__()
        self.net = net
        self.base_scale = base_scale
        self.target_scale = target_scale
        self.half = half

    def forward(self, x_u8):
        x = x_u8.permute(0, 3, 1, 2)
        x = (x.half() if self.half else x.float()) / 255.0  # NHWC u8 -> NCHW 0..1
        y = self.net(x)  # NCHW f32 at base_scale
        if self.target_scale != self.base_scale:
            # bilinear (not area): area maps to adaptive pooling, whose output
            # size must be constant, which breaks the dynamic H/W axes.
            ratio = self.target_scale / self.base_scale
            y = F.interpolate(
                y, scale_factor=ratio, mode="bilinear", align_corners=False, recompute_scale_factor=False
            )
        # Back to fp32 before the 0..255 clamp so rounding matches the fp32 graph.
        y = torch.clamp(y.float() * 255.0, 0.0, 255.0).round()
        return y.permute(0, 2, 3, 1).to(torch.uint8)  # NCHW -> NHWC u8


# ---------------------------------------------------------------------------
# Export targets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportTarget:
    out_name: str
    weights: str
    arch: str  # "srvgg" | "rrdb"
    base_scale: int
    target_scale: int
    num_block: int = 23


TARGETS: list[ExportTarget] = [
    ExportTarget("realesr-animevideov3-x4-uint8.onnx", "realesr-animevideov3.pth", "srvgg", 4, 4),
    ExportTarget("realesr-animevideov3-x3-uint8.onnx", "realesr-animevideov3.pth", "srvgg", 4, 3),
    ExportTarget("realesr-animevideov3-x2-uint8.onnx", "realesr-animevideov3.pth", "srvgg", 4, 2),
    ExportTarget("realesrgan-x4plus-uint8.onnx", "RealESRGAN_x4plus.pth", "rrdb", 4, 4, num_block=23),
    ExportTarget(
        "realesrgan-x4plus-anime-uint8.onnx", "RealESRGAN_x4plus_anime_6B.pth", "rrdb", 4, 4, num_block=6
    ),
]


def _load_state_dict(weights_path: Path) -> dict:
    # weights_only=True: the official Real-ESRGAN checkpoints are pure tensor
    # state dicts, so refuse to unpickle arbitrary objects (RCE guard).
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict):
        return checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
    return checkpoint


def _build_network(target: ExportTarget, weights_path: Path) -> nn.Module:
    if target.arch == "srvgg":
        net = SRVGGNetCompact(num_feat=64, num_conv=16, upscale=target.base_scale)
    else:
        net = RRDBNet(num_feat=64, num_block=target.num_block, num_grow_ch=32, scale=target.base_scale)
    net.load_state_dict(_load_state_dict(weights_path), strict=True)
    net.eval()
    return net


def fp16_name(out_name: str) -> str:
    """`foo-uint8.onnx` -> `foo-uint8-fp16.onnx` (must match backend_registry)."""
    return out_name.replace(".onnx", "-fp16.onnx")


def _export_one(target: ExportTarget, weights_dir: Path, out_dir: Path) -> None:
    weights_path = weights_dir / target.weights
    if not weights_path.exists():
        print(f"[skip] {target.out_name}: weights not found ({weights_path})")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    # Both precisions from the same weights: fp32 is the portable/CPU graph, fp16
    # is the fast GPU graph (measured 1.33x at 1080p->4x, max pixel delta 3/255).
    _export_precision(target, weights_path, out_dir / target.out_name, half=False)
    _export_precision(target, weights_path, out_dir / fp16_name(target.out_name), half=True)


def _export_precision(target: ExportTarget, weights_path: Path, out_path: Path, *, half: bool) -> None:
    net = _build_network(target, weights_path)
    if half:
        net = net.half()
    model = Uint8Wrapper(net, target.base_scale, target.target_scale, half=half).eval()

    dummy = torch.randint(0, 256, (1, 64, 96, 3), dtype=torch.uint8)
    with torch.no_grad():
        out = model(dummy)
    expected = (64 * target.target_scale, 96 * target.target_scale)
    assert tuple(out.shape[1:3]) == expected, f"{out_path.name}: got {tuple(out.shape)}"
    assert out.dtype == torch.uint8

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["image"],
        output_names=["upscaled"],
        dynamic_axes={"image": {1: "h", 2: "w"}, "upscaled": {1: "H", 2: "W"}},
        opset_version=OPSET,
        dynamo=False,
    )
    precision = "fp16" if half else "fp32"
    print(f"[ok]   {out_path.name}  ({out_path.stat().st_size} bytes, x{target.target_scale}, {precision})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export builtin Real-ESRGAN models to uint8 ONNX")
    parser.add_argument("--weights-dir", default="vendor/realesrgan-onnx/weights")
    parser.add_argument("--out-dir", default="vendor/realesrgan-onnx")
    args = parser.parse_args()

    weights_dir = Path(args.weights_dir)
    out_dir = Path(args.out_dir)
    failures = 0
    for target in TARGETS:
        try:
            _export_one(target, weights_dir, out_dir)
        except Exception as exc:  # one bad export must not abort the rest
            failures += 1
            print(f"[fail] {target.out_name}: {exc}")
    if failures:
        raise SystemExit(f"{failures} model(s) failed to export")


if __name__ == "__main__":
    main()
