from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import app.services.engines.generation_onnx as generation_onnx_module
from app.config import Settings
from app.services.engines.generation_onnx import (
    CUDA_ONLY_MESSAGE,
    GenerationEngine,
    GenerationRequest,
    PIPELINE_CLASS_NAMES,
    VRAM_MESSAGE,
    _load_pipeline_class,
    _read_declared_class_name,
    _wrap_generation_error,
)


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)


def make_request(**overrides: Any) -> GenerationRequest:
    defaults: dict[str, Any] = {
        "prompt": "a red apple",
        "negative_prompt": None,
        "steps": 4,
        "guidance": 7.5,
        "width": 256,
        "height": 256,
        "seed": None,
    }
    defaults.update(overrides)
    return GenerationRequest(**defaults)


@pytest.mark.parametrize(
    ("declared", "expected_ort_name"),
    [
        ("OnnxStableDiffusionPipeline", "ORTStableDiffusionPipeline"),
        ("StableDiffusionXLPipeline", "ORTStableDiffusionXLPipeline"),
        ("StableDiffusion3Pipeline", "ORTStableDiffusion3Pipeline"),
    ],
)
def test_pipeline_class_map_covers_known_variants(
    declared: str, expected_ort_name: str
) -> None:
    # Turbo NO tiene entrada propia: es un checkpoint de la MISMA clase SDXL.
    assert PIPELINE_CLASS_NAMES[declared] == expected_ort_name


@pytest.mark.parametrize(
    ("declared", "expected_ort_name"),
    [
        ("ORTStableDiffusionXLPipeline", "ORTStableDiffusionXLPipeline"),
        ("ORTStableDiffusion3Pipeline", "ORTStableDiffusion3Pipeline"),
    ],
)
def test_load_pipeline_class_passthrough_for_ort_names(
    declared: str, expected_ort_name: str
) -> None:
    assert _load_pipeline_class(declared).__name__ == expected_ort_name


@pytest.mark.parametrize(
    "unknown_class_name",
    ["KandinskyV22Pipeline", "ORTKandinskyPipeline"],
)
def test_load_pipeline_class_unknown_class_lists_supported(
    unknown_class_name: str,
) -> None:
    with pytest.raises(RuntimeError) as excinfo:
        _load_pipeline_class(unknown_class_name)
    message = str(excinfo.value)
    assert unknown_class_name in message
    for supported in set(PIPELINE_CLASS_NAMES) | set(PIPELINE_CLASS_NAMES.values()):
        assert supported in message


def test_read_declared_class_name_reads_model_index(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLPipeline"}), encoding="utf-8"
    )
    assert _read_declared_class_name(tmp_path) == "StableDiffusionXLPipeline"


def test_read_declared_class_name_missing_class_is_actionable(tmp_path: Path) -> None:
    (tmp_path / "model_index.json").write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="_class_name"):
        _read_declared_class_name(tmp_path)


class FakeImage:
    def save(self, path: Path) -> None:
        Path(path).write_bytes(b"png")


class FakeResult:
    def __init__(self) -> None:
        self.images = [FakeImage()]


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> FakeResult:
        self.calls.append(kwargs)
        callback = kwargs.get("callback")
        if callback is not None:
            for step in range(kwargs["num_inference_steps"]):
                callback(step, None, None)
        return FakeResult()


class RecordingCoordinator:
    def __init__(self) -> None:
        self.acquired: list[tuple[str, Any]] = []

    def acquire(self, device: str, owner: Any) -> None:
        self.acquired.append((device, owner))


def make_engine(tmp_path: Path, pipeline: Any | None = None) -> tuple[GenerationEngine, RecordingCoordinator, FakePipeline]:
    coordinator = RecordingCoordinator()
    engine = GenerationEngine(make_settings(tmp_path), coordinator)  # type: ignore[arg-type]
    fake = pipeline or FakePipeline()
    engine._create_pipeline = lambda pipeline_dir, device: fake  # type: ignore[method-assign]
    return engine, coordinator, fake


@pytest.mark.anyio
async def test_run_generates_png_and_reports_progress(tmp_path: Path) -> None:
    engine, coordinator, fake = make_engine(tmp_path)
    output = tmp_path / "out.png"
    progress: list[tuple[int, int]] = []

    result = await engine.run(
        model_id="gen--amd--sd15",
        pipeline_dir=tmp_path,
        request=make_request(),
        device="dml:0",
        output_path=output,
        progress_cb=lambda done, total: progress.append((done, total)),
    )

    assert result == output
    assert output.read_bytes() == b"png"
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert coordinator.acquired == [("dml:0", engine)]
    call = fake.calls[0]
    assert call["prompt"] == "a red apple"
    assert call["num_inference_steps"] == 4
    assert call["width"] == 256 and call["height"] == 256


@pytest.mark.anyio
async def test_run_passes_seeded_generator(tmp_path: Path) -> None:
    engine, _coordinator, fake = make_engine(tmp_path)

    await engine.run(
        model_id="m",
        pipeline_dir=tmp_path,
        request=make_request(seed=42),
        device="cpu",
        output_path=tmp_path / "o.png",
        progress_cb=lambda *_: None,
    )

    assert "generator" in fake.calls[0]


@pytest.mark.anyio
async def test_pipeline_cache_is_lru_one_across_devices(tmp_path: Path) -> None:
    engine, _coordinator, _fake = make_engine(tmp_path)
    created: list[str] = []
    engine._create_pipeline = lambda pipeline_dir, device: created.append(device) or FakePipeline()  # type: ignore[method-assign]

    common = dict(
        model_id="m", pipeline_dir=tmp_path, request=make_request(),
        output_path=tmp_path / "o.png", progress_cb=lambda *_: None,
    )
    await engine.run(device="dml:0", **common)
    await engine.run(device="dml:0", **common)  # cache hit
    await engine.run(device="dml:1", **common)  # evicts dml:0
    await engine.run(device="dml:0", **common)  # rebuilt

    assert created == ["dml:0", "dml:1", "dml:0"]


@pytest.mark.anyio
async def test_release_device_drops_cached_pipeline(tmp_path: Path) -> None:
    engine, _coordinator, _fake = make_engine(tmp_path)
    created: list[str] = []
    engine._create_pipeline = lambda pipeline_dir, device: created.append(device) or FakePipeline()  # type: ignore[method-assign]
    common = dict(
        model_id="m", pipeline_dir=tmp_path, request=make_request(),
        output_path=tmp_path / "o.png", progress_cb=lambda *_: None,
    )
    await engine.run(device="dml:0", **common)

    engine.release_device("dml:0")
    await engine.run(device="dml:0", **common)

    assert created == ["dml:0", "dml:0"]


def test_wrap_generation_error_maps_vram() -> None:
    wrapped = _wrap_generation_error(RuntimeError("DML allocation failed: out of memory"))
    assert VRAM_MESSAGE in str(wrapped)


def test_wrap_generation_error_maps_cuda_only() -> None:
    wrapped = _wrap_generation_error(RuntimeError("CUDAExecutionProvider is not available"))
    assert CUDA_ONLY_MESSAGE in str(wrapped)


def test_create_pipeline_builds_expected_from_pretrained_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Any, dict[str, Any]]] = []
    declared_names: list[str] = []
    tuned: list[tuple[Any, str]] = []

    class FakePipelineClass:
        @staticmethod
        def from_pretrained(path: Any, **kwargs: Any) -> Any:
            calls.append((path, kwargs))
            return object()

    def fake_tune(sess_options: Any, device: str) -> None:
        tuned.append((sess_options, device))

    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusionXLPipeline"}), encoding="utf-8"
    )

    def fake_load_pipeline_class(declared_class_name: str) -> type[FakePipelineClass]:
        declared_names.append(declared_class_name)
        return FakePipelineClass

    monkeypatch.setattr(
        generation_onnx_module, "_load_pipeline_class", fake_load_pipeline_class
    )
    monkeypatch.setattr(generation_onnx_module, "_tune_session_options_for_device", fake_tune)

    engine = GenerationEngine(make_settings(tmp_path), RecordingCoordinator())  # type: ignore[arg-type]

    engine._create_pipeline(tmp_path, "dml:1")
    path, kwargs = calls[0]
    assert path == str(tmp_path)
    assert kwargs["use_io_binding"] is False
    assert kwargs["provider"] == "DmlExecutionProvider"
    assert kwargs["provider_options"] == {"device_id": 1}
    assert kwargs["session_options"] is tuned[0][0]
    assert tuned[0][1] == "dml:1"

    engine._create_pipeline(tmp_path, "cpu")
    _path_cpu, kwargs_cpu = calls[1]
    assert kwargs_cpu["provider"] == "CPUExecutionProvider"
    assert "provider_options" not in kwargs_cpu
    assert declared_names == [
        "StableDiffusionXLPipeline",
        "StableDiffusionXLPipeline",
    ]


@pytest.mark.anyio
async def test_run_wraps_pipeline_errors(tmp_path: Path) -> None:
    class ExplodingPipeline:
        def __call__(self, **kwargs: Any) -> Any:
            raise RuntimeError("CUDAExecutionProvider is not available")

    engine, _coordinator, _fake = make_engine(tmp_path, pipeline=ExplodingPipeline())
    with pytest.raises(RuntimeError, match="requiere GPU NVIDIA"):
        await engine.run(
            model_id="m", pipeline_dir=tmp_path, request=make_request(),
            device="dml:0", output_path=tmp_path / "o.png", progress_cb=lambda *_: None,
        )
