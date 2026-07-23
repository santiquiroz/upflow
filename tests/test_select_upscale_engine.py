from __future__ import annotations

import pytest
from app.models import UpscaleJob
from app.services.engines.base import UpscaleEngine
from app.services.job_manager import select_upscale_engine
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus
from pathlib import Path
import tempfile
from app.config import Settings


class FakeUpscaleEngine(UpscaleEngine):
    """Minimal fake engine for testing."""

    def available(self) -> bool:
        return True

    async def run(self, job: UpscaleJob) -> Path:
        raise NotImplementedError


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Create settings pointing to a temporary directory."""
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))


@pytest.fixture
def registry(settings: Settings) -> ModelRegistry:
    """Create a real ModelRegistry with temporary storage."""
    return ModelRegistry(settings)


@pytest.fixture
def builtin_engine() -> UpscaleEngine:
    """Create a fake builtin engine."""
    return FakeUpscaleEngine()


@pytest.fixture
def onnx_engine() -> UpscaleEngine:
    """Create a fake ONNX engine."""
    return FakeUpscaleEngine()


@pytest.fixture
def source_image(tmp_path: Path) -> Path:
    """Create a dummy source image."""
    image_path = tmp_path / "source.png"
    image_path.write_bytes(b"fake-png")
    return image_path


def test_select_upscale_engine_prefers_onnx_for_onnx_model(
    registry: ModelRegistry,
    builtin_engine: UpscaleEngine,
    onnx_engine: UpscaleEngine,
    source_image: Path,
) -> None:
    """When job has an ONNX model_id, select_upscale_engine returns onnx_engine."""
    # Arrange: Register an ONNX model
    onnx_entry = ModelEntry(
        id="test-onnx-model",
        name="Test ONNX Model",
        kind=ModelKind.onnx,
        source="https://example.com/model.onnx",
        size_bytes=1000,
        scale=2,
        arch="test",
        file_path="models/test-onnx-model.onnx",
        status=ModelStatus.installed,
    )
    registry.register(onnx_entry)

    job = UpscaleJob(
        source_path=source_image,
        original_filename="test.png",
        model_name="test-model",
        scale=2,
        output_format="png",
        model_id="test-onnx-model",
        device="cpu",
    )

    # Act
    result = select_upscale_engine(job, registry, builtin_engine, onnx_engine)

    # Assert
    assert result is onnx_engine, "Should return onnx_engine for ONNX model"


def test_select_upscale_engine_uses_builtin_when_no_model_id(
    registry: ModelRegistry,
    builtin_engine: UpscaleEngine,
    onnx_engine: UpscaleEngine,
    source_image: Path,
) -> None:
    """When job has no model_id, select_upscale_engine returns builtin_engine."""
    # Arrange: Create job without model_id
    job = UpscaleJob(
        source_path=source_image,
        original_filename="test.png",
        model_name="test-model",
        scale=2,
        output_format="png",
        model_id=None,
        device="cpu",
    )

    # Act
    result = select_upscale_engine(job, registry, builtin_engine, onnx_engine)

    # Assert
    assert result is builtin_engine, "Should return builtin_engine when no model_id"


def test_select_upscale_engine_raises_when_onnx_model_without_onnx_engine(
    registry: ModelRegistry,
    builtin_engine: UpscaleEngine,
    source_image: Path,
) -> None:
    """When job has ONNX model_id but onnx_engine is None, raises RuntimeError."""
    # Arrange: Register an ONNX model
    onnx_entry = ModelEntry(
        id="test-onnx-model",
        name="Test ONNX Model",
        kind=ModelKind.onnx,
        source="https://example.com/model.onnx",
        size_bytes=1000,
        scale=2,
        arch="test",
        file_path="models/test-onnx-model.onnx",
        status=ModelStatus.installed,
    )
    registry.register(onnx_entry)

    job = UpscaleJob(
        source_path=source_image,
        original_filename="test.png",
        model_name="test-model",
        scale=2,
        output_format="png",
        model_id="test-onnx-model",
        device="cpu",
    )

    # Act & Assert
    with pytest.raises(
        RuntimeError,
        match=r"Model 'test-onnx-model' requires the ONNX engine, which is not configured",
    ):
        select_upscale_engine(job, registry, builtin_engine, onnx_engine=None)


def test_select_upscale_engine_uses_builtin_when_registry_is_none(
    builtin_engine: UpscaleEngine,
    onnx_engine: UpscaleEngine,
    source_image: Path,
) -> None:
    """When registry is None, select_upscale_engine returns builtin_engine."""
    # Arrange: Create job with model_id but no registry
    job = UpscaleJob(
        source_path=source_image,
        original_filename="test.png",
        model_name="test-model",
        scale=2,
        output_format="png",
        model_id="some-model",
        device="cpu",
    )

    # Act
    result = select_upscale_engine(job, registry=None, builtin_engine=builtin_engine, onnx_engine=onnx_engine)

    # Assert
    assert result is builtin_engine, "Should return builtin_engine when registry is None"
