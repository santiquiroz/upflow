from __future__ import annotations

from pathlib import Path

import pytest

import app.services.engines.sdcpp_engine as sdcpp_module
from app.config import Settings
from app.services.engines.generation_onnx import GenerationRequest
from app.services.engines.sdcpp_engine import SDCPP_MODEL_ID, SdcppEngine


def make_settings(tmp_path: Path, enabled: bool = True, with_files: bool = True) -> Settings:
    binary = tmp_path / "sd.exe"
    model = tmp_path / "model.safetensors"
    if with_files:
        binary.write_bytes(b"exe")
        model.write_bytes(b"weights")
    return Settings(
        _env_file=None,
        RUNTIME_DIR=str(tmp_path / "runtime"),
        ENABLE_SDCPP=enabled,
        SDCPP_BINARY=str(binary),
        SDCPP_MODEL=str(model),
    )


def make_request(**overrides) -> GenerationRequest:
    base = dict(
        prompt="un zorro", negative_prompt=None, steps=20, guidance=7.0,
        width=512, height=512, seed=None,
    )
    base.update(overrides)
    return GenerationRequest(**base)


def test_available_requires_flag_binary_and_model(tmp_path: Path) -> None:
    on = tmp_path / "on"
    off = tmp_path / "off"
    empty = tmp_path / "empty"
    for sub in (on, off, empty):
        sub.mkdir()
    assert SdcppEngine(make_settings(on)).available() is True
    assert SdcppEngine(make_settings(off, enabled=False)).available() is False
    assert SdcppEngine(make_settings(empty, with_files=False)).available() is False


def test_build_command_includes_all_parameters(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    engine = SdcppEngine(settings)
    request = make_request(negative_prompt="borroso", seed=42)

    command = engine.build_command(request, tmp_path / "out.png")

    assert command[0] == str(settings.sdcpp_binary_path)
    assert command[command.index("-m") + 1] == str(settings.sdcpp_model_path)
    assert command[command.index("--steps") + 1] == "20"
    assert command[command.index("-n") + 1] == "borroso"
    assert command[command.index("-s") + 1] == "42"


@pytest.mark.asyncio
async def test_run_fails_clearly_when_lane_is_off(tmp_path: Path) -> None:
    engine = SdcppEngine(make_settings(tmp_path, enabled=False))
    with pytest.raises(RuntimeError, match="download-sdcpp"):
        await engine.run(make_request(), tmp_path / "out.png")


@pytest.mark.asyncio
async def test_run_raises_on_nonzero_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_process(command, timeout):
        return b"", b"vulkan device not found", 1

    monkeypatch.setattr(sdcpp_module, "run_guarded_process", fake_process)
    engine = SdcppEngine(make_settings(tmp_path))

    with pytest.raises(RuntimeError, match="exit code 1"):
        await engine.run(make_request(), tmp_path / "out.png")


@pytest.mark.asyncio
async def test_run_raises_when_no_output_is_produced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_process(command, timeout):
        return b"", b"", 0

    monkeypatch.setattr(sdcpp_module, "run_guarded_process", fake_process)
    engine = SdcppEngine(make_settings(tmp_path))

    with pytest.raises(RuntimeError, match="no output"):
        await engine.run(make_request(), tmp_path / "out.png")


@pytest.mark.asyncio
async def test_run_returns_the_output_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out.png"

    async def fake_process(command, timeout):
        output.write_bytes(b"png")
        return b"", b"", 0

    monkeypatch.setattr(sdcpp_module, "run_guarded_process", fake_process)
    engine = SdcppEngine(make_settings(tmp_path))

    assert await engine.run(make_request(), output) == output


def test_model_id_constant_is_stable() -> None:
    # El id viaja en la API (picker + create_job): cambiarlo rompe clientes.
    assert SDCPP_MODEL_ID == "experimental-sdcpp"
