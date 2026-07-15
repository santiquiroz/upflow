from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from app.services import model_converter
from app.services.model_converter import ConversionResult, convert_to_onnx

# ---------------------------------------------------------------------------
# SP1 Task 6 - model_converter: .pth/.safetensors -> ONNX via Spandrel +
# torch.onnx.export (opset 17, dynamic H/W axes, fp32 CPU).
#
# Spandrel's architecture registry only recognizes real published SR
# architectures (ESRGAN, SwinIR, ...) by inspecting state dict key
# names/shapes -- it will never match an arbitrary toy nn.Module. So instead
# of shipping a real checkpoint fixture, these tests monkeypatch the single
# loader seam (`_load_spandrel_descriptor`) to return a fake descriptor that
# wraps a tiny REAL nn.Module (Conv2d + PixelShuffle, a genuine 2x
# sub-pixel-conv upscaler). Everything downstream of that seam --
# torch.onnx.export and the onnxruntime CPU validation -- runs for real.
# ---------------------------------------------------------------------------


class _FakeArchitecture:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeDescriptor:
    def __init__(self, model: nn.Module, scale: int, arch_name: str) -> None:
        self.model = model
        self.scale = scale
        self.architecture = _FakeArchitecture(arch_name)


class _TinyUpscaler2x(nn.Module):
    """A genuine (tiny) 2x super-resolution model: sub-pixel conv upscaling."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3 * (2**2), kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))


class _BrokenExportModel(nn.Module):
    """Raises inside forward() so torch.onnx.export fails deterministically."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("boom: unsupported op")


def _install_fake_loader(
    monkeypatch: pytest.MonkeyPatch, descriptor: _FakeDescriptor
) -> list[Path]:
    calls: list[Path] = []

    def fake_loader(weight_path: Path) -> _FakeDescriptor:
        calls.append(weight_path)
        return descriptor

    monkeypatch.setattr(model_converter, "_load_spandrel_descriptor", fake_loader)
    return calls


# ---------------------------------------------------------------------------
# happy path: real export + real onnxruntime validation
# ---------------------------------------------------------------------------


def test_convert_to_onnx_produces_valid_onnx_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = _FakeDescriptor(_TinyUpscaler2x(), scale=2, arch_name="TinySubPixel")
    _install_fake_loader(monkeypatch, descriptor)
    weight_path = tmp_path / "weights.safetensors"
    weight_path.write_bytes(b"fake-weights-not-actually-read")
    out_onnx = tmp_path / "out" / "model.onnx"

    result = convert_to_onnx(weight_path, out_onnx, progress_cb=None)

    assert isinstance(result, ConversionResult)
    assert result.arch == "TinySubPixel"
    assert result.scale == 2
    assert out_onnx.exists()
    assert out_onnx.stat().st_size > 0


def test_convert_to_onnx_calls_loader_with_given_weight_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = _FakeDescriptor(_TinyUpscaler2x(), scale=2, arch_name="TinySubPixel")
    calls = _install_fake_loader(monkeypatch, descriptor)
    weight_path = tmp_path / "weights.pth"
    weight_path.write_bytes(b"fake-weights")
    out_onnx = tmp_path / "model.onnx"

    convert_to_onnx(weight_path, out_onnx)

    assert calls == [weight_path]


def test_convert_to_onnx_exported_graph_has_dynamic_hw_and_runs_in_onnxruntime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = _FakeDescriptor(_TinyUpscaler2x(), scale=2, arch_name="TinySubPixel")
    _install_fake_loader(monkeypatch, descriptor)
    weight_path = tmp_path / "weights.safetensors"
    weight_path.write_bytes(b"fake-weights")
    out_onnx = tmp_path / "model.onnx"

    convert_to_onnx(weight_path, out_onnx)

    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(out_onnx), providers=["CPUExecutionProvider"])
    input_info = session.get_inputs()[0]
    assert len(input_info.shape) == 4
    assert "float" in str(input_info.type).lower()

    # A non-square, non-training-size input must scale correctly (2x here),
    # proving the exported axes are genuinely dynamic, not baked in at 64x64.
    tile = np.zeros((1, 3, 32, 48), dtype=np.float32)
    output = session.run(None, {input_info.name: tile})[0]
    assert output.shape == (1, 3, 64, 96)


def test_convert_to_onnx_reports_progress_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = _FakeDescriptor(_TinyUpscaler2x(), scale=2, arch_name="TinySubPixel")
    _install_fake_loader(monkeypatch, descriptor)
    weight_path = tmp_path / "weights.safetensors"
    weight_path.write_bytes(b"fake-weights")
    out_onnx = tmp_path / "model.onnx"
    stages: list[str] = []

    convert_to_onnx(weight_path, out_onnx, progress_cb=stages.append)

    assert stages == ["loading", "exporting", "validating"]


# ---------------------------------------------------------------------------
# failure paths: clear, wrapped errors
# ---------------------------------------------------------------------------


def test_convert_to_onnx_raises_clear_error_when_spandrel_load_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_loader(weight_path: Path):
        raise ValueError("unrecognized architecture")

    monkeypatch.setattr(model_converter, "_load_spandrel_descriptor", failing_loader)
    weight_path = tmp_path / "weights.pth"
    weight_path.write_bytes(b"garbage")
    out_onnx = tmp_path / "model.onnx"

    with pytest.raises(RuntimeError, match="[Ll]oad"):
        convert_to_onnx(weight_path, out_onnx)

    assert not out_onnx.exists()


def test_convert_to_onnx_raises_clear_error_when_export_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = _FakeDescriptor(_BrokenExportModel(), scale=2, arch_name="Broken")
    _install_fake_loader(monkeypatch, descriptor)
    weight_path = tmp_path / "weights.safetensors"
    weight_path.write_bytes(b"fake-weights")
    out_onnx = tmp_path / "model.onnx"

    with pytest.raises(RuntimeError, match="[Ee]xport"):
        convert_to_onnx(weight_path, out_onnx)

    assert not out_onnx.exists()


def test_convert_to_onnx_raises_clear_error_when_onnxruntime_validation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Patches the seam itself (not onnxruntime internals) to simulate a
    # validation failure; the real `_validate_exported_onnx` already wraps
    # errors with an "Exported ONNX model failed CPU validation" message
    # (exercised implicitly by the happy-path tests running it for real).
    # What this test asserts is convert_to_onnx's own contract: propagate
    # the failure and delete the now-invalid out_onnx file rather than
    # leaving a broken artifact behind.
    descriptor = _FakeDescriptor(_TinyUpscaler2x(), scale=2, arch_name="TinySubPixel")
    _install_fake_loader(monkeypatch, descriptor)
    weight_path = tmp_path / "weights.safetensors"
    weight_path.write_bytes(b"fake-weights")
    out_onnx = tmp_path / "model.onnx"

    def fake_validate(path: Path) -> None:
        raise RuntimeError("simulated onnxruntime rejection")

    monkeypatch.setattr(model_converter, "_validate_exported_onnx", fake_validate)

    with pytest.raises(RuntimeError, match="simulated onnxruntime rejection"):
        convert_to_onnx(weight_path, out_onnx)


def test_convert_to_onnx_creates_parent_directory_for_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = _FakeDescriptor(_TinyUpscaler2x(), scale=2, arch_name="TinySubPixel")
    _install_fake_loader(monkeypatch, descriptor)
    weight_path = tmp_path / "weights.safetensors"
    weight_path.write_bytes(b"fake-weights")
    out_onnx = tmp_path / "nested" / "dirs" / "model.onnx"

    convert_to_onnx(weight_path, out_onnx)

    assert out_onnx.exists()
