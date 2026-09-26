from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn.functional as F
from spandrel import (
    Architecture,
    ImageModelDescriptor,
    MaskedImageModelDescriptor,
    ModelTiling,
    SizeRequirements,
)
from torch import nn

from app.services import model_converter
from app.services.model_converter import (
    ConversionResult,
    UnsupportedModelError,
    convert_to_onnx,
    export_probe_size,
    opset_for_architecture,
    required_opset_from_error,
    second_probe_size,
)

# ---------------------------------------------------------------------------
# SP1 Task 6 / MNT-02 - model_converter: .pth/.safetensors -> ONNX via
# Spandrel + torch.onnx.export (dynamic H/W, fp32 CPU).
#
# Spandrel's registry only recognizes real published architectures, so most
# tests monkeypatch the loader seam (`_load_spandrel_descriptor`) to return a
# REAL Spandrel ImageModelDescriptor wrapping a tiny nn.Module. That keeps
# Spandrel's own call_fn / clamp / size-requirement semantics (and the private
# `_call_fn` attribute the converter depends on) under test. One test loads a
# real, tiny DRUNet through Spandrel's detection end to end.
# ---------------------------------------------------------------------------


class _ToyArchitecture(Architecture[nn.Module]):
    def __init__(self, arch_id: str) -> None:
        super().__init__(id=arch_id, detect=lambda state_dict: False)

    def load(self, state_dict):  # pragma: no cover - never detected
        raise NotImplementedError


def _descriptor(
    model: nn.Module,
    *,
    arch_id: str = "Toy",
    scale: int = 1,
    purpose: str = "Restoration",
    call_fn=None,
    size_requirements: SizeRequirements | None = None,
    channels: tuple[int, int] = (3, 3),
    tiling: ModelTiling = ModelTiling.SUPPORTED,
    supports_half: bool = False,
) -> ImageModelDescriptor:
    return ImageModelDescriptor(
        model,
        model.state_dict(),
        architecture=_ToyArchitecture(arch_id),
        purpose=purpose,
        tags=[],
        supports_half=supports_half,
        supports_bfloat16=False,
        scale=scale,
        input_channels=channels[0],
        output_channels=channels[1],
        size_requirements=size_requirements,
        tiling=tiling,
        call_fn=call_fn,
    )


class _TinyUpscaler2x(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3 * (2**2), kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.conv(x))


class _SignedRangeRestorer(nn.Module):
    """Outputs in [-1, 1], like MixDehazeNet; its call_fn maps back to [0, 1]."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.conv(x * 2 - 1))


class _NoiseMapRestorer(nn.Module):
    """Expects image + noise-level channel, like DRUNet/DnCNN."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(4, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv(x))


class _TupleRestorer(nn.Module):
    """Returns (image, quality factor), like FBCNN."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.sigmoid(self.conv(x)), x.mean(dim=(2, 3))


class _MultipleOf16Restorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 16 or x.shape[-2] % 16:
            raise RuntimeError("input must be a multiple of 16")
        return torch.sigmoid(self.conv(x))


class _Col2ImRestorer(nn.Module):
    """Uses F.fold (ONNX Col2Im, opset 18+) on a constant-size parameter."""

    def __init__(self) -> None:
        super().__init__()
        self.cols = nn.Parameter(torch.rand(1, 4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gain = F.fold(self.cols, output_size=(3, 3), kernel_size=2).mean()
        return x * gain


class _BrokenExportModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_tracing():
            raise RuntimeError("boom: unsupported op")
        return self.conv(x)


def _noise_map_call(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    _, _, height, width = image.shape
    noise_map = torch.zeros(1, 1, height, width).to(image) + 15 / 255
    return model(torch.cat([image, noise_map], dim=1))


def _first_pixel_gain_call(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    # .item() is baked as a constant by the tracer: the export only matches
    # descriptor(x) on the probe it was traced with.
    gain = float(image[0, 0, 0, 0].item())
    return model(torch.cat([image, image[:, :1]], dim=1)) * gain


def _install_fake_loader(monkeypatch: pytest.MonkeyPatch, descriptor) -> list[Path]:
    calls: list[Path] = []

    def fake_loader(weight_path: Path):
        calls.append(weight_path)
        return descriptor

    monkeypatch.setattr(model_converter, "_load_spandrel_descriptor", fake_loader)
    return calls


def _weight_file(tmp_path: Path, name: str = "weights.safetensors") -> Path:
    weight_path = tmp_path / name
    weight_path.write_bytes(b"fake-weights-not-actually-read")
    return weight_path


def _session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _run(path: Path, array: np.ndarray) -> np.ndarray:
    session = _session(path)
    return session.run(None, {session.get_inputs()[0].name: array})[0]


def _random_image(height: int, width: int, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, height, width, generator=generator)


def _max_abs_vs_descriptor(path: Path, descriptor, image: torch.Tensor) -> float:
    expected = descriptor(image.clone()).numpy()
    actual = _run(path, image.numpy())
    return float(np.max(np.abs(actual - expected)))


def _onnx_opset(path: Path) -> int:
    return next(entry.version for entry in onnx.load(str(path)).opset_import if entry.domain in ("", "ai.onnx"))


# ---------------------------------------------------------------------------
# happy path: real export + real onnxruntime validation
# ---------------------------------------------------------------------------


def test_convert_to_onnx_produces_valid_onnx_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_loader(monkeypatch, _descriptor(_TinyUpscaler2x(), scale=2, purpose="SR", arch_id="TinySubPixel"))
    out_onnx = tmp_path / "out" / "model.onnx"

    result = convert_to_onnx(_weight_file(tmp_path), out_onnx, progress_cb=None)

    assert isinstance(result, ConversionResult)
    assert result.arch == "TinySubPixel"
    assert result.scale == 2
    assert out_onnx.exists()
    assert out_onnx.stat().st_size > 0


def test_convert_to_onnx_calls_loader_with_given_weight_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_loader(monkeypatch, _descriptor(_TinyUpscaler2x(), scale=2, purpose="SR"))
    weight_path = _weight_file(tmp_path, "weights.pth")

    convert_to_onnx(weight_path, tmp_path / "model.onnx")

    assert calls == [weight_path]


def test_convert_to_onnx_exported_graph_has_dynamic_hw_and_runs_in_onnxruntime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_loader(monkeypatch, _descriptor(_TinyUpscaler2x(), scale=2, purpose="SR"))
    out_onnx = tmp_path / "model.onnx"

    convert_to_onnx(_weight_file(tmp_path), out_onnx)

    input_info = _session(out_onnx).get_inputs()[0]
    assert len(input_info.shape) == 4
    assert "float" in str(input_info.type).lower()
    output = _run(out_onnx, np.zeros((1, 3, 32, 48), dtype=np.float32))
    assert output.shape == (1, 3, 64, 96)


def test_convert_to_onnx_reports_progress_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_loader(monkeypatch, _descriptor(_TinyUpscaler2x(), scale=2, purpose="SR"))
    stages: list[str] = []

    convert_to_onnx(_weight_file(tmp_path), tmp_path / "model.onnx", progress_cb=stages.append)

    assert stages == ["loading", "exporting", "validating"]


def test_convert_to_onnx_creates_parent_directory_for_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_loader(monkeypatch, _descriptor(_TinyUpscaler2x(), scale=2, purpose="SR"))
    out_onnx = tmp_path / "nested" / "dirs" / "model.onnx"

    convert_to_onnx(_weight_file(tmp_path), out_onnx)

    assert out_onnx.exists()


# ---------------------------------------------------------------------------
# call_fn: the exported graph is descriptor(x), not model.forward
# ---------------------------------------------------------------------------


def test_signed_range_call_fn_is_part_of_the_exported_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = _descriptor(_SignedRangeRestorer(), call_fn=lambda model, image: model(image) * 0.5 + 0.5)
    _install_fake_loader(monkeypatch, descriptor)
    out_onnx = tmp_path / "model.onnx"

    convert_to_onnx(_weight_file(tmp_path), out_onnx)

    image = _random_image(40, 56)
    assert _run(out_onnx, image.numpy()).min() >= 0.0
    assert _max_abs_vs_descriptor(out_onnx, descriptor, image) < 1e-4


def test_noise_channel_call_fn_exports_a_three_channel_graph(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = _descriptor(_NoiseMapRestorer(), call_fn=_noise_map_call)
    _install_fake_loader(monkeypatch, descriptor)
    out_onnx = tmp_path / "model.onnx"

    convert_to_onnx(_weight_file(tmp_path), out_onnx)

    assert _session(out_onnx).get_inputs()[0].shape[1] == 3
    assert _max_abs_vs_descriptor(out_onnx, descriptor, _random_image(48, 80)) < 1e-4


def test_tuple_returning_model_exports_only_the_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = _descriptor(_TupleRestorer(), call_fn=lambda model, image: model(image)[0])
    _install_fake_loader(monkeypatch, descriptor)
    out_onnx = tmp_path / "model.onnx"

    convert_to_onnx(_weight_file(tmp_path), out_onnx)

    assert len(_session(out_onnx).get_outputs()) == 1
    assert _max_abs_vs_descriptor(out_onnx, descriptor, _random_image(33, 45)) < 1e-4


def test_output_is_clamped_like_the_descriptor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = _descriptor(_SignedRangeRestorer())
    _install_fake_loader(monkeypatch, descriptor)
    out_onnx = tmp_path / "model.onnx"

    convert_to_onnx(_weight_file(tmp_path), out_onnx)

    output = _run(out_onnx, _random_image(32, 32).numpy())
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_real_spandrel_drunet_converts_end_to_end(tmp_path: Path) -> None:
    from spandrel.architectures.DRUNet.__arch.network_unet import DRUNet

    torch.manual_seed(0)
    weight_path = tmp_path / "drunet_tiny.pth"
    torch.save(DRUNet(in_nc=4, out_nc=3, nc=[8, 16, 32, 64], nb=2).state_dict(), weight_path)
    out_onnx = tmp_path / "drunet.onnx"

    result = convert_to_onnx(weight_path, out_onnx)

    assert result.arch == "DRUNet"
    assert result.purpose == "Restoration"
    assert (result.scale, result.channels_in, result.channels_out) == (1, 3, 3)
    assert result.size_multiple == 8
    assert _run(out_onnx, _random_image(72, 80).numpy()).shape == (1, 3, 72, 80)


# ---------------------------------------------------------------------------
# probe sizes follow SizeRequirements
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("requirements", "first", "second"),
    [
        (SizeRequirements(), (64, 64), (72, 80)),
        (SizeRequirements(minimum=40), (64, 64), (72, 80)),
        (SizeRequirements(multiple_of=8), (64, 64), (72, 80)),
        (SizeRequirements(multiple_of=16), (64, 64), (80, 96)),
        (SizeRequirements(minimum=100, multiple_of=16), (112, 112), (128, 144)),
        (SizeRequirements(multiple_of=128, square=True), (128, 128), (256, 256)),
        (SizeRequirements(minimum=512), (512, 512), (520, 528)),
    ],
)
def test_probe_sizes_satisfy_size_requirements(requirements, first, second) -> None:
    assert export_probe_size(requirements) == first
    assert second_probe_size(requirements) == second
    for height, width in (first, second):
        assert requirements.check(width, height)


def test_model_with_size_multiple_converts_with_matching_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    descriptor = _descriptor(_MultipleOf16Restorer(), size_requirements=SizeRequirements(multiple_of=16))
    _install_fake_loader(monkeypatch, descriptor)

    result = convert_to_onnx(_weight_file(tmp_path), tmp_path / "model.onnx")

    assert result.size_multiple == 16
    assert result.probe_size == (64, 64)


# ---------------------------------------------------------------------------
# parity: an export that diverges from descriptor(x) is never installed
# ---------------------------------------------------------------------------


def test_export_that_does_not_match_descriptor_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = _descriptor(_NoiseMapRestorer(), call_fn=_first_pixel_gain_call)
    _install_fake_loader(monkeypatch, descriptor)
    out_onnx = tmp_path / "model.onnx"

    with pytest.raises(RuntimeError, match="does not match"):
        convert_to_onnx(_weight_file(tmp_path), out_onnx)

    assert not out_onnx.exists()


def test_validation_uses_the_descriptor_as_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = _descriptor(_TinyUpscaler2x(), scale=2, purpose="SR")
    _install_fake_loader(monkeypatch, descriptor)
    references: list[tuple[int, ...]] = []
    real_reference = model_converter._reference_output

    def recording_reference(desc, probe):
        references.append(tuple(probe.shape))
        return real_reference(desc, probe)

    monkeypatch.setattr(model_converter, "_reference_output", recording_reference)

    convert_to_onnx(_weight_file(tmp_path), tmp_path / "model.onnx")

    assert references == [(1, 3, 64, 64), (1, 3, 72, 80)]


# ---------------------------------------------------------------------------
# rejected model families
# ---------------------------------------------------------------------------


def _inpainting_descriptor() -> MaskedImageModelDescriptor:
    model = nn.Conv2d(4, 3, kernel_size=1)
    return MaskedImageModelDescriptor(
        model,
        model.state_dict(),
        architecture=_ToyArchitecture("LaMa"),
        purpose="Inpainting",
        tags=[],
        supports_half=False,
        supports_bfloat16=False,
        input_channels=3,
        output_channels=3,
    )


def test_inpainting_model_is_rejected_before_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_loader(monkeypatch, _inpainting_descriptor())
    out_onnx = tmp_path / "model.onnx"
    stages: list[str] = []

    with pytest.raises(UnsupportedModelError, match="inpainting"):
        convert_to_onnx(_weight_file(tmp_path), out_onnx, progress_cb=stages.append)

    assert stages == ["loading"]
    assert not out_onnx.exists()


def test_face_restoration_model_is_rejected_before_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = _descriptor(
        _TinyUpscaler2x(), arch_id="GFPGAN", scale=2, purpose="FaceSR", size_requirements=SizeRequirements(minimum=512)
    )
    _install_fake_loader(monkeypatch, descriptor)
    out_onnx = tmp_path / "model.onnx"

    with pytest.raises(UnsupportedModelError, match="face restoration"):
        convert_to_onnx(_weight_file(tmp_path), out_onnx)

    assert not out_onnx.exists()


def test_model_with_unsupported_channel_count_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = _descriptor(nn.Conv2d(4, 3, kernel_size=1), channels=(4, 3))
    _install_fake_loader(monkeypatch, descriptor)

    with pytest.raises(UnsupportedModelError, match="channels"):
        convert_to_onnx(_weight_file(tmp_path), tmp_path / "model.onnx")


def test_grayscale_model_converts_with_single_channel_probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = _descriptor(nn.Conv2d(1, 1, kernel_size=3, padding=1), channels=(1, 1))
    _install_fake_loader(monkeypatch, descriptor)
    out_onnx = tmp_path / "model.onnx"

    result = convert_to_onnx(_weight_file(tmp_path), out_onnx)

    assert (result.channels_in, result.channels_out) == (1, 1)
    assert _run(out_onnx, np.zeros((1, 1, 20, 24), dtype=np.float32)).shape == (1, 1, 20, 24)


# ---------------------------------------------------------------------------
# opset per architecture
# ---------------------------------------------------------------------------


def test_opset_defaults_to_17_and_follows_the_architecture_table() -> None:
    assert opset_for_architecture("ESRGAN") == 17
    assert opset_for_architecture("IPT") == 18


def test_architecture_in_opset_table_exports_at_its_opset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_loader(monkeypatch, _descriptor(_Col2ImRestorer(), arch_id="IPT"))
    out_onnx = tmp_path / "model.onnx"

    result = convert_to_onnx(_weight_file(tmp_path), out_onnx)

    assert result.opset == 18
    assert _onnx_opset(out_onnx) == 18


def test_export_retries_at_the_opset_the_exporter_asks_for(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_loader(monkeypatch, _descriptor(_Col2ImRestorer(), arch_id="Toy"))
    out_onnx = tmp_path / "model.onnx"

    result = convert_to_onnx(_weight_file(tmp_path), out_onnx)

    assert result.opset == 18
    assert _onnx_opset(out_onnx) == 18


def test_required_opset_is_read_from_the_exporter_error() -> None:
    error = RuntimeError(
        "Exporting the operator 'aten::col2im' to ONNX opset version 17 is not supported. "
        "Support for this operator was added in version 18, try exporting with this version"
    )

    assert required_opset_from_error(error) == 18
    assert required_opset_from_error(RuntimeError("boom")) is None


def test_opset_escalation_stops_at_the_legacy_exporter_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_loader(monkeypatch, _descriptor(_SignedRangeRestorer()))
    attempted: list[int] = []

    def fake_export(graph, probe, out_onnx, opset):
        attempted.append(opset)
        raise RuntimeError("Support for this operator was added in version 21")

    monkeypatch.setattr(model_converter, "_run_export", fake_export)

    with pytest.raises(RuntimeError, match="Failed to export"):
        convert_to_onnx(_weight_file(tmp_path), tmp_path / "model.onnx")

    assert attempted == [17]


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------


def test_conversion_result_carries_descriptor_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = _descriptor(
        _SignedRangeRestorer(),
        arch_id="SCUNet",
        size_requirements=SizeRequirements(minimum=40, multiple_of=8),
        tiling=ModelTiling.DISCOURAGED,
        supports_half=True,
    )
    _install_fake_loader(monkeypatch, descriptor)

    result = convert_to_onnx(_weight_file(tmp_path), tmp_path / "model.onnx")

    assert result == ConversionResult(
        arch="SCUNet",
        scale=1,
        purpose="Restoration",
        channels_in=3,
        channels_out=3,
        size_minimum=40,
        size_multiple=8,
        size_square=False,
        tiling="discouraged",
        supports_half=True,
        opset=17,
        probe_size=(64, 64),
    )


# ---------------------------------------------------------------------------
# failure paths: clear, wrapped errors
# ---------------------------------------------------------------------------


def test_convert_to_onnx_raises_clear_error_when_spandrel_load_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_loader(weight_path: Path):
        raise ValueError("unrecognized architecture")

    monkeypatch.setattr(model_converter, "_load_spandrel_descriptor", failing_loader)
    out_onnx = tmp_path / "model.onnx"

    with pytest.raises(RuntimeError, match="[Ll]oad"):
        convert_to_onnx(_weight_file(tmp_path, "weights.pth"), out_onnx)

    assert not out_onnx.exists()


def test_convert_to_onnx_raises_clear_error_when_export_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_loader(monkeypatch, _descriptor(_BrokenExportModel()))
    out_onnx = tmp_path / "model.onnx"

    with pytest.raises(RuntimeError, match="[Ee]xport"):
        convert_to_onnx(_weight_file(tmp_path), out_onnx)

    assert not out_onnx.exists()


def test_convert_to_onnx_removes_output_when_validation_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_loader(monkeypatch, _descriptor(_TinyUpscaler2x(), scale=2, purpose="SR"))
    out_onnx = tmp_path / "model.onnx"

    def fake_validate(path: Path, descriptor) -> None:
        raise RuntimeError("simulated onnxruntime rejection")

    monkeypatch.setattr(model_converter, "_validate_exported_onnx", fake_validate)

    with pytest.raises(RuntimeError, match="simulated onnxruntime rejection"):
        convert_to_onnx(_weight_file(tmp_path), out_onnx)

    assert not out_onnx.exists()
