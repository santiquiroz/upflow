from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.models import JobStatus
from app.services.device_semaphores import DeviceSemaphores
from app.services.generation_job_manager import GenerationJobManager
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)


class FakeGenerationEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def run(self, **kwargs: Any) -> Path:
        self.calls.append(kwargs)
        output_path: Path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"png")
        return output_path


class FakeUpscaleEngine:
    def __init__(self, tmp_path: Path) -> None:
        self.calls: list[Any] = []
        self.tmp_path = tmp_path

    async def run(self, job: Any) -> Path:
        self.calls.append(job)
        out = self.tmp_path / f"upscaled-{job.id}.png"
        out.write_bytes(b"bigpng")
        return out


def register_generation_model(registry: ModelRegistry, settings: Settings, model_id: str = "gen--amd--sd15") -> None:
    model_dir = settings.models_path / "generation" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    registry.register(
        ModelEntry(
            id=model_id, name="amd/sd15", kind=ModelKind.diffusion_onnx,
            source="hf:amd/sd15", size_bytes=1, scale=None,
            file_path=f"generation/{model_id}",
        )
    )


def make_manager(tmp_path: Path) -> tuple[GenerationJobManager, FakeGenerationEngine, FakeUpscaleEngine, ModelRegistry, Settings]:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    engine = FakeGenerationEngine()
    upscaler = FakeUpscaleEngine(tmp_path)
    manager = GenerationJobManager(
        settings, engine, DeviceSemaphores(settings),
        registry=registry, upscale_engine=upscaler, onnx_upscale_engine=None,
    )
    register_generation_model(registry, settings)
    return manager, engine, upscaler, registry, settings


async def drain(manager: GenerationJobManager) -> None:
    # Same drain mechanism as JobManager/VideoJobManager/AudioJobManager tests
    # (tests/test_cleanup.py, tests/test_job_cancel.py): managers expose no
    # `_process_next` seam, so tests run the real worker loop and wait for the
    # queue to empty instead.
    await manager.start()
    try:
        await asyncio.wait_for(manager.queue.join(), timeout=2.0)
    finally:
        await manager.stop()


async def test_generation_job_completes_and_sets_output(tmp_path: Path) -> None:
    manager, engine, _up, _reg, _settings = make_manager(tmp_path)
    job = await manager.create_job(prompt="a red apple", model_id="gen--amd--sd15", device="cpu")

    await drain(manager)

    final = manager.get_job(job.id)
    assert final is not None
    assert final.status == JobStatus.completed
    assert final.output_path is not None and final.output_path.exists()
    assert final.finished_at is not None
    assert engine.calls[0]["device"] == "cpu"


async def test_auto_upscale_runs_two_stages_in_one_job(tmp_path: Path) -> None:
    manager, engine, upscaler, _reg, _settings = make_manager(tmp_path)
    job = await manager.create_job(
        prompt="a red apple", model_id="gen--amd--sd15", device="cpu",
        auto_upscale=True, upscale_model_name="realesrgan-x4plus", upscale_scale=4,
    )

    await drain(manager)

    final = manager.get_job(job.id)
    assert final.status == JobStatus.completed
    assert len(upscaler.calls) == 1
    assert upscaler.calls[0].scale == 4
    assert final.output_path.name.startswith("upscaled-")
    stage_keys = [s["key"] for s in final.metadata["stages"]]
    assert stage_keys == ["generating", "upscaling"]
    generated_intermediate = engine.calls[0]["output_path"]
    assert not generated_intermediate.exists()  # intermedio borrado tras upscale OK


async def test_create_job_rejects_unknown_model(tmp_path: Path) -> None:
    manager, *_ = make_manager(tmp_path)
    with pytest.raises(ValueError, match="model"):
        await manager.create_job(prompt="x", model_id="nope", device="cpu")


async def test_create_job_rejects_upscaler_model_id_as_generation_model(tmp_path: Path) -> None:
    manager, _e, _u, registry, _s = make_manager(tmp_path)
    registry.register(
        ModelEntry(id="up1", name="up", kind=ModelKind.onnx, source="hf:x/y", size_bytes=1, scale=2, file_path="up1.onnx")
    )
    with pytest.raises(ValueError):
        await manager.create_job(prompt="x", model_id="up1", device="cpu")


async def test_create_job_rejects_auto_device(tmp_path: Path) -> None:
    manager, *_ = make_manager(tmp_path)
    with pytest.raises(ValueError, match="auto"):
        await manager.create_job(prompt="x", model_id="gen--amd--sd15", device="auto")


async def test_create_job_requires_upscale_params_when_auto_upscale(tmp_path: Path) -> None:
    manager, *_ = make_manager(tmp_path)
    with pytest.raises(ValueError, match="upscale"):
        await manager.create_job(prompt="x", model_id="gen--amd--sd15", device="cpu", auto_upscale=True)


async def test_cancel_queued_job_skips_processing(tmp_path: Path) -> None:
    manager, engine, *_ = make_manager(tmp_path)
    job = await manager.create_job(prompt="x", model_id="gen--amd--sd15", device="cpu")

    assert manager.cancel_job(job.id) is True
    await drain(manager)

    assert manager.get_job(job.id).status == JobStatus.cancelled
    assert engine.calls == []
