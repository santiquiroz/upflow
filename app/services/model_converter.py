from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SP1 Task 6 - model_converter: converts a downloaded .pth/.safetensors
# super-resolution checkpoint into a portable ONNX graph, so it can run
# through the SAME OnnxUpscaler inference path as a natively-published
# .onnx model (opset 17, dynamic H/W axes, fp32).
#
# Spandrel (`ModelLoader().load_from_file`) is the architecture-detection
# layer: it inspects the state dict's key names/shapes and picks the
# matching SR architecture from its own registry (ESRGAN, SwinIR,
# RealESRGAN, ...), handling .safetensors and .pth (via a restricted
# unpickler) uniformly -- this module never parses raw weights itself, only
# the resulting torch.nn.Module + metadata (scale, architecture name).
#
# `_load_spandrel_descriptor` is a monkeypatchable seam (mirrors
# ModelInstaller._create_validation_session): unit tests inject a fake
# descriptor wrapping a tiny REAL nn.Module instead of depending on Spandrel
# recognizing an architecture -- it only ever recognizes real published SR
# archs, never an arbitrary toy nn.Module -- while still exercising a REAL
# torch.onnx.export + REAL onnxruntime CPU validation end-to-end.
#
# dynamo=False (legacy TorchScript-based exporter) is explicit and
# deliberate: torch's newer dynamo-based exporter is the default in recent
# versions but additionally requires the `onnxscript` package. The legacy
# exporter only needs the `onnx` protobuf library (already a hard
# dependency of torch.onnx.export itself) and fully supports opset 17 +
# dynamic_axes, which is all this module needs.
# ---------------------------------------------------------------------------

ONNX_OPSET = 17
DUMMY_INPUT_SIZE = 64
INPUT_NAME = "input"
OUTPUT_NAME = "output"
DYNAMIC_AXES: dict[str, dict[int, str]] = {
    INPUT_NAME: {2: "height", 3: "width"},
    OUTPUT_NAME: {2: "out_height", 3: "out_width"},
}

STAGE_LOADING = "loading"
STAGE_EXPORTING = "exporting"
STAGE_VALIDATING = "validating"

ConversionProgressCallback = Callable[[str], None]


@dataclass(slots=True, frozen=True)
class ConversionResult:
    arch: str
    scale: int


def _load_spandrel_descriptor(weight_path: Path) -> Any:
    # Monkeypatchable seam: unit tests inject a fake descriptor wrapping a
    # tiny real nn.Module and never touch the real Spandrel registry.
    from spandrel import ModelLoader

    return ModelLoader().load_from_file(str(weight_path))


def _report(progress_cb: ConversionProgressCallback | None, stage: str) -> None:
    if progress_cb is not None:
        progress_cb(stage)


def _make_dummy_input() -> torch.Tensor:
    # Lazy import: torch takes ~4-5s to import and this module is pulled in
    # transitively at app startup (main.py -> ModelInstaller -> here), but
    # the vast majority of sessions never convert a .pth. Keeping torch out
    # of the module top level keeps startup sub-second -- same lazy-import
    # discipline already applied to spandrel and onnxruntime below.
    import torch

    return torch.zeros(1, 3, DUMMY_INPUT_SIZE, DUMMY_INPUT_SIZE, dtype=torch.float32)


def _load_descriptor_or_raise(weight_path: Path) -> Any:
    try:
        return _load_spandrel_descriptor(weight_path)
    except Exception as exc:  # Spandrel raises its own exception types
        raise RuntimeError(f"Failed to load weight file with Spandrel: {exc}") from exc


def _export_onnx(model: torch.nn.Module, out_onnx: Path) -> None:
    import torch

    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = _make_dummy_input()
    try:
        with torch.no_grad(), warnings.catch_warnings():
            # dynamo=False is deliberate (see module docstring); torch emits a
            # DeprecationWarning steering callers toward the dynamo exporter,
            # which is not actionable here and would otherwise be noisy on
            # every conversion.
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            torch.onnx.export(
                model,
                dummy_input,
                str(out_onnx),
                input_names=[INPUT_NAME],
                output_names=[OUTPUT_NAME],
                dynamic_axes=DYNAMIC_AXES,
                opset_version=ONNX_OPSET,
                dynamo=False,
            )
    except Exception as exc:  # torch raises many different native exception types
        out_onnx.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to export model to ONNX: {exc}") from exc


def _validate_exported_onnx(out_onnx: Path) -> None:
    import onnxruntime as ort

    try:
        session = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
        input_info = session.get_inputs()[0]
        dummy = _make_dummy_input().numpy()
        session.run(None, {input_info.name: dummy})
    except Exception as exc:  # onnxruntime raises its own native exception types
        raise RuntimeError(f"Exported ONNX model failed CPU validation: {exc}") from exc


def convert_to_onnx(
    weight_path: Path,
    out_onnx: Path,
    progress_cb: ConversionProgressCallback | None = None,
) -> ConversionResult:
    """Converts a .pth/.safetensors SR checkpoint to a dynamic-shape fp32 ONNX graph (sync; wrap in asyncio.to_thread)."""
    _report(progress_cb, STAGE_LOADING)
    descriptor = _load_descriptor_or_raise(weight_path)
    model = descriptor.model
    model.eval()

    _report(progress_cb, STAGE_EXPORTING)
    _export_onnx(model, out_onnx)

    _report(progress_cb, STAGE_VALIDATING)
    try:
        _validate_exported_onnx(out_onnx)
    except Exception:
        out_onnx.unlink(missing_ok=True)
        raise

    return ConversionResult(arch=descriptor.architecture.name, scale=descriptor.scale)
