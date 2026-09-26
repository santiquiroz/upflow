from __future__ import annotations

import logging
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SP1 Task 6 - model_converter: converts a downloaded .pth/.safetensors
# checkpoint into a portable ONNX graph, so it can run through the SAME
# OnnxUpscaler inference path as a natively-published .onnx model (dynamic
# H/W axes, fp32).
#
# Spandrel (`ModelLoader().load_from_file`) is the architecture-detection
# layer: it inspects the state dict's key names/shapes and picks the
# matching architecture from its own registry, handling .safetensors and
# .pth (via a restricted unpickler) uniformly -- this module never parses
# raw weights itself.
#
# MNT-02 - the exported graph is the descriptor's call path, not the bare
# `descriptor.model.forward`: `DescriptorGraph` runs Spandrel's `call_fn`
# (extra noise channel for DRUNet/DnCNN, `[0]` of the tuple for
# FBCNN/MMRealSR, `*0.5+0.5` for MixDehazeNet...) and clamps to [0, 1]
# exactly like `descriptor(x)`. The probe sizes honor the model's
# SizeRequirements (the graph itself never pads), and the export is only
# accepted when ONNX Runtime on CPU matches `descriptor(x)` at two
# different probe sizes: the second size catches shapes the tracer baked in
# as constants. Inpainting (image + mask) and FaceSR (needs detection and
# alignment) do not fit "one image in, one image out" and are rejected
# before exporting anything.
#
# `_load_spandrel_descriptor` is a monkeypatchable seam: unit tests inject a
# real Spandrel ImageModelDescriptor wrapping a tiny nn.Module, so the real
# call_fn/clamp/size-requirement semantics are exercised end to end.
#
# dynamo=False (legacy TorchScript-based exporter) is explicit and
# deliberate: the dynamo exporter additionally requires `onnxscript`. The
# legacy exporter tops out at opset 20.
# ---------------------------------------------------------------------------

DEFAULT_ONNX_OPSET = 17
MAX_ONNX_OPSET = 20
# F.fold in forward exports as Col2Im, which only exists from opset 18.
ARCH_OPSETS: dict[str, int] = {"IPT": 18}

MIN_PROBE_SIZE = 64
SECOND_PROBE_STEP = 8
PROBE_SEED = 0
PARITY_MAX_ABS = 2e-3
SUPPORTED_CHANNELS = (1, 3)

INPUT_NAME = "input"
OUTPUT_NAME = "output"
DYNAMIC_AXES: dict[str, dict[int, str]] = {
    INPUT_NAME: {2: "height", 3: "width"},
    OUTPUT_NAME: {2: "out_height", 3: "out_width"},
}

PURPOSE_INPAINTING = "Inpainting"
PURPOSE_FACE_SR = "FaceSR"

STAGE_LOADING = "loading"
STAGE_EXPORTING = "exporting"
STAGE_VALIDATING = "validating"

_REQUIRED_OPSET_PATTERN = re.compile(r"added in (?:opset )?version (\d+)")

ConversionProgressCallback = Callable[[str], None]


class UnsupportedModelError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class ConversionResult:
    arch: str
    scale: int
    purpose: str = "SR"
    channels_in: int = 3
    channels_out: int = 3
    size_minimum: int = 0
    size_multiple: int = 1
    size_square: bool = False
    tiling: str = "supported"
    supports_half: bool = False
    opset: int = DEFAULT_ONNX_OPSET
    probe_size: tuple[int, int] = (MIN_PROBE_SIZE, MIN_PROBE_SIZE)


def _load_spandrel_descriptor(weight_path: Path) -> Any:
    from spandrel import ModelLoader

    return ModelLoader().load_from_file(str(weight_path))


def _report(progress_cb: ConversionProgressCallback | None, stage: str) -> None:
    if progress_cb is not None:
        progress_cb(stage)


def _load_descriptor_or_raise(weight_path: Path) -> Any:
    try:
        return _load_spandrel_descriptor(weight_path)
    except Exception as exc:  # Spandrel raises its own exception types
        raise RuntimeError(f"Failed to load weight file with Spandrel: {exc}") from exc


def _architecture_id(descriptor: Any) -> str:
    architecture = descriptor.architecture
    return str(getattr(architecture, "id", architecture.name))


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def export_probe_size(size_requirements: Any) -> tuple[int, int]:
    base = max(MIN_PROBE_SIZE, size_requirements.minimum)
    side = _ceil_to_multiple(base, size_requirements.multiple_of)
    return side, side


def second_probe_size(size_requirements: Any) -> tuple[int, int]:
    side, _ = export_probe_size(size_requirements)
    step = _ceil_to_multiple(SECOND_PROBE_STEP, size_requirements.multiple_of)
    if size_requirements.square:
        return side + step, side + step
    return side + step, side + 2 * step


def opset_for_architecture(arch_id: str) -> int:
    return ARCH_OPSETS.get(arch_id, DEFAULT_ONNX_OPSET)


def required_opset_from_error(exc: BaseException) -> int | None:
    match = _REQUIRED_OPSET_PATTERN.search(str(exc))
    return int(match.group(1)) if match else None


def _next_opset(exc: BaseException, current: int) -> int | None:
    required = required_opset_from_error(exc)
    if required is None or required <= current or required > MAX_ONNX_OPSET:
        return None
    return required


def _reject_inpainting(descriptor: Any, arch_id: str) -> None:
    if descriptor.purpose == PURPOSE_INPAINTING:
        raise UnsupportedModelError(
            f"{arch_id} is an inpainting model (image + mask); only single-image models can be installed"
        )


def _reject_face_sr(descriptor: Any, arch_id: str) -> None:
    if descriptor.purpose == PURPOSE_FACE_SR:
        raise UnsupportedModelError(
            f"{arch_id} is a face restoration model that needs face detection and alignment; "
            "it cannot be installed as a whole-image model"
        )


def _reject_unsupported_channels(descriptor: Any, arch_id: str) -> None:
    channels = (descriptor.input_channels, descriptor.output_channels)
    if any(count not in SUPPORTED_CHANNELS for count in channels):
        raise UnsupportedModelError(
            f"{arch_id} uses {channels[0]} input / {channels[1]} output channels; "
            "only grayscale (1) or RGB (3) images are supported"
        )


def _require_convertible(descriptor: Any) -> None:
    arch_id = _architecture_id(descriptor)
    _reject_inpainting(descriptor, arch_id)
    _reject_face_sr(descriptor, arch_id)
    _reject_unsupported_channels(descriptor, arch_id)


def _make_probe(channels: int, size: tuple[int, int], seed: int) -> torch.Tensor:
    # Lazy import: torch takes seconds to import and this module is pulled
    # in at app startup, while most sessions never convert a checkpoint.
    import torch

    generator = torch.Generator().manual_seed(seed)
    return torch.rand(1, channels, size[0], size[1], generator=generator, dtype=torch.float32)


def _parity_probes(descriptor: Any) -> list[torch.Tensor]:
    requirements = descriptor.size_requirements
    sizes = (export_probe_size(requirements), second_probe_size(requirements))
    return [_make_probe(descriptor.input_channels, size, PROBE_SEED + index) for index, size in enumerate(sizes)]


def _run_export(graph: torch.nn.Module, probe: torch.Tensor, out_onnx: Path, opset: int) -> None:
    import torch

    with torch.no_grad(), warnings.catch_warnings():
        # The legacy exporter is deliberate (module header); tracer warnings
        # flag shapes baked as constants, which the second parity probe
        # already rejects with a clearer error.
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            graph,
            probe,
            str(out_onnx),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes=DYNAMIC_AXES,
            opset_version=opset,
            dynamo=False,
        )


def _export_with_opset_escalation(graph: torch.nn.Module, probe: torch.Tensor, out_onnx: Path, opset: int) -> int:
    while True:
        try:
            _run_export(graph, probe, out_onnx, opset)
            return opset
        except Exception as exc:  # torch raises many different native exception types
            next_opset = _next_opset(exc, opset)
            if next_opset is None:
                raise
            logger.info("Retrying ONNX export at opset %d (opset %d lacks an operator)", next_opset, opset)
            opset = next_opset


def _export_onnx(graph: torch.nn.Module, probe: torch.Tensor, out_onnx: Path, opset: int) -> int:
    out_onnx.parent.mkdir(parents=True, exist_ok=True)
    try:
        return _export_with_opset_escalation(graph, probe, out_onnx, opset)
    except Exception as exc:  # torch raises many different native exception types
        out_onnx.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to export model to ONNX: {exc}") from exc


def _reference_output(descriptor: Any, probe: torch.Tensor) -> np.ndarray:
    return descriptor(probe.clone()).float().cpu().numpy()


def _run_session(session: Any, probe: torch.Tensor) -> np.ndarray:
    try:
        return session.run(None, {session.get_inputs()[0].name: probe.numpy()})[0]
    except Exception as exc:  # onnxruntime raises its own native exception types
        raise RuntimeError(f"Exported ONNX model failed CPU validation: {exc}") from exc


def _probe_label(probe: torch.Tensor) -> str:
    return f"{probe.shape[-2]}x{probe.shape[-1]}"


def _require_same_shape(actual: np.ndarray, expected: np.ndarray, probe: torch.Tensor) -> None:
    if actual.shape != expected.shape:
        raise RuntimeError(
            f"Exported ONNX model returned shape {actual.shape} instead of {expected.shape} "
            f"for a {_probe_label(probe)} input"
        )


def _require_parity(actual: np.ndarray, expected: np.ndarray, probe: torch.Tensor) -> None:
    import numpy as np

    difference = float(np.max(np.abs(actual.astype(np.float32) - expected)))
    if not np.isfinite(difference) or difference > PARITY_MAX_ABS:
        raise RuntimeError(
            f"Exported ONNX model does not match the PyTorch model "
            f"(max |diff| {difference:.4g} for a {_probe_label(probe)} input)"
        )


def _check_probe(session: Any, descriptor: Any, probe: torch.Tensor) -> None:
    expected = _reference_output(descriptor, probe)
    actual = _run_session(session, probe)
    _require_same_shape(actual, expected, probe)
    _require_parity(actual, expected, probe)


def _create_cpu_session(out_onnx: Path) -> Any:
    import onnxruntime as ort

    try:
        return ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
    except Exception as exc:  # onnxruntime raises its own native exception types
        raise RuntimeError(f"Exported ONNX model failed CPU validation: {exc}") from exc


def _validate_exported_onnx(out_onnx: Path, descriptor: Any) -> None:
    session = _create_cpu_session(out_onnx)
    for probe in _parity_probes(descriptor):
        _check_probe(session, descriptor, probe)


def _tiling_name(descriptor: Any) -> str:
    return descriptor.tiling.name.lower()


def _conversion_result(descriptor: Any, opset: int) -> ConversionResult:
    requirements = descriptor.size_requirements
    return ConversionResult(
        arch=descriptor.architecture.name,
        scale=descriptor.scale,
        purpose=descriptor.purpose,
        channels_in=descriptor.input_channels,
        channels_out=descriptor.output_channels,
        size_minimum=requirements.minimum,
        size_multiple=requirements.multiple_of,
        size_square=requirements.square,
        tiling=_tiling_name(descriptor),
        supports_half=bool(descriptor.supports_half),
        opset=opset,
        probe_size=export_probe_size(requirements),
    )


def _load_convertible_descriptor(weight_path: Path) -> Any:
    descriptor = _load_descriptor_or_raise(weight_path)
    _require_convertible(descriptor)
    descriptor.eval()
    return descriptor


def convert_to_onnx(
    weight_path: Path,
    out_onnx: Path,
    progress_cb: ConversionProgressCallback | None = None,
) -> ConversionResult:
    """Converts a .pth/.safetensors checkpoint to a dynamic-shape fp32 ONNX graph (sync; wrap in asyncio.to_thread)."""
    from app.services.descriptor_graph import graph_from_descriptor

    _report(progress_cb, STAGE_LOADING)
    descriptor = _load_convertible_descriptor(weight_path)

    _report(progress_cb, STAGE_EXPORTING)
    probe = _parity_probes(descriptor)[0]
    start_opset = opset_for_architecture(_architecture_id(descriptor))
    opset = _export_onnx(graph_from_descriptor(descriptor), probe, out_onnx, start_opset)

    _report(progress_cb, STAGE_VALIDATING)
    try:
        _validate_exported_onnx(out_onnx, descriptor)
    except Exception:
        out_onnx.unlink(missing_ok=True)
        raise

    return _conversion_result(descriptor, opset)
