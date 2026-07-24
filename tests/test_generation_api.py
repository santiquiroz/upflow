from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import app.main as main_module
from app.api.routes import (
    cancel_generation_job,
    create_generation_job,
    download_generation_job,
    generation_capabilities,
    generation_job_to_response,
    get_generation_install_status,
    get_generation_job,
)
from app.config import Settings
from app.main import app
from app.models import GenerationJob, JobStatus
from app.schemas import CreateGenerationJobRequest, InstallModelRequest
from app.services.device_semaphores import DeviceSemaphores
from app.services.generation_installer import GenerationModelInstaller
from app.services.generation_job_manager import GenerationJobManager
from app.services.model_installer import InstallJob, InstallStatus
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry

# ---------------------------------------------------------------------------
# generation module Task 9 - API: /api/v1/generation/*. Route-level tests call
# the route coroutines directly with fake service doubles (mirrors
# tests/test_job_cancel.py, tests/test_job_status_routes.py, tests/test_models_api.py);
# a handful of TestClient(app) tests verify main.py's lifespan actually wires
# GenerationJobManager/GenerationModelInstaller into app.state and that
# request/response bodies serialize to the documented camelCase shape.
# ---------------------------------------------------------------------------

MODEL_ID = "gen--amd--sd15"


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


class FakeDevicesService:
    def __init__(self, devices: list[dict] | None = None) -> None:
        self._devices = devices or [{"id": "cpu", "kind": "cpu", "name": "CPU", "backend": "cpu"}]

    def list_devices(self) -> list[dict]:
        return self._devices

    def validate(self, device_id: str) -> dict:
        for device in self._devices:
            if device["id"] == device_id:
                return device
        raise ValueError(f"Unknown device id: {device_id!r}")


def register_generation_model(registry: ModelRegistry, settings: Settings, model_id: str = MODEL_ID) -> None:
    model_dir = settings.models_path / "generation" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    registry.register(
        ModelEntry(
            id=model_id, name="amd/sd15", kind=ModelKind.diffusion_onnx,
            source="hf:amd/sd15", size_bytes=1, scale=None,
            file_path=f"generation/{model_id}",
        )
    )


def make_manager(tmp_path: Path) -> tuple[GenerationJobManager, FakeGenerationEngine, ModelRegistry, Settings]:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    engine = FakeGenerationEngine()
    manager = GenerationJobManager(
        settings, engine, DeviceSemaphores(settings),
        registry=registry, upscale_engine=None, onnx_upscale_engine=None,
    )
    register_generation_model(registry, settings)
    return manager, engine, registry, settings


def make_generation_job(**overrides: object) -> GenerationJob:
    fields: dict[str, object] = dict(prompt="a red apple", model_id=MODEL_ID)
    fields.update(overrides)
    return GenerationJob(**fields)


# ---------------------------------------------------------------------------
# generation_job_to_response
# ---------------------------------------------------------------------------


def test_generation_job_to_response_maps_fields() -> None:
    job = make_generation_job(negative_prompt="blurry", steps=30, guidance=8.0, width=576, height=576, seed=7)

    response = generation_job_to_response(job)

    assert response.id == job.id
    assert response.prompt == "a red apple"
    assert response.negative_prompt == "blurry"
    assert response.model_id == MODEL_ID
    assert response.steps == 30
    assert response.guidance == 8.0
    assert response.width == 576
    assert response.height == 576
    assert response.seed == 7
    assert response.download_url is None


def test_generation_job_to_response_sets_download_url_only_when_completed() -> None:
    job = make_generation_job()
    job.status = JobStatus.completed

    response = generation_job_to_response(job)

    assert response.download_url == f"/api/v1/generation/jobs/{job.id}/download"


def test_generation_job_response_serializes_camel_case() -> None:
    job = make_generation_job(negative_prompt="blurry", auto_upscale=True)

    serialized = generation_job_to_response(job).model_dump(by_alias=True)

    assert serialized["negativePrompt"] == "blurry"
    assert serialized["modelId"] == MODEL_ID
    assert serialized["autoUpscale"] is True
    assert "createdAt" in serialized


# ---------------------------------------------------------------------------
# POST /generation/jobs
# ---------------------------------------------------------------------------


async def test_create_generation_job_route_creates_job(tmp_path: Path) -> None:
    manager, engine, _registry, _settings = make_manager(tmp_path)
    payload = CreateGenerationJobRequest(prompt="a red apple", model_id=MODEL_ID, device="cpu")

    response = await create_generation_job(payload=payload, generation_jobs=manager)

    assert response.prompt == "a red apple"
    assert response.model_id == MODEL_ID
    assert response.status == JobStatus.queued
    assert manager.get_job(response.id) is not None


async def test_create_generation_job_returns_400_for_unknown_model(tmp_path: Path) -> None:
    manager, *_ = make_manager(tmp_path)
    payload = CreateGenerationJobRequest(prompt="x", model_id="nope")

    with pytest.raises(HTTPException) as exc_info:
        await create_generation_job(payload=payload, generation_jobs=manager)

    assert exc_info.value.status_code == 400


async def test_create_generation_job_returns_429_when_queue_full(tmp_path: Path) -> None:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None, MAX_QUEUE_SIZE=1)
    registry = ModelRegistry(settings)
    engine = FakeGenerationEngine()
    manager = GenerationJobManager(
        settings, engine, DeviceSemaphores(settings),
        registry=registry, upscale_engine=None, onnx_upscale_engine=None,
    )
    register_generation_model(registry, settings)

    await create_generation_job(
        payload=CreateGenerationJobRequest(prompt="a", model_id=MODEL_ID), generation_jobs=manager
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_generation_job(
            payload=CreateGenerationJobRequest(prompt="b", model_id=MODEL_ID), generation_jobs=manager
        )

    assert exc_info.value.status_code == 429


# ---------------------------------------------------------------------------
# GET /generation/jobs/{id}
# ---------------------------------------------------------------------------


async def test_get_generation_job_returns_404_when_unknown(tmp_path: Path) -> None:
    manager, *_ = make_manager(tmp_path)

    with pytest.raises(HTTPException) as exc_info:
        await get_generation_job(job_id="missing", generation_jobs=manager)

    assert exc_info.value.status_code == 404


async def test_get_generation_job_returns_response_and_download_url_tracks_status(tmp_path: Path) -> None:
    manager, *_ = make_manager(tmp_path)
    job = make_generation_job()
    manager.jobs[job.id] = job

    queued_response = await get_generation_job(job_id=job.id, generation_jobs=manager)
    job.status = JobStatus.completed
    completed_response = await get_generation_job(job_id=job.id, generation_jobs=manager)

    assert queued_response.id == job.id
    assert queued_response.download_url is None
    assert completed_response.download_url == f"/api/v1/generation/jobs/{job.id}/download"


# ---------------------------------------------------------------------------
# POST /generation/jobs/{id}/cancel
# ---------------------------------------------------------------------------


async def test_cancel_generation_job_endpoint_returns_updated_response(tmp_path: Path) -> None:
    manager, *_ = make_manager(tmp_path)
    job = make_generation_job()
    manager.jobs[job.id] = job
    await manager.queue.put(job)

    response = await cancel_generation_job(job_id=job.id, generation_jobs=manager)

    assert response.id == job.id
    assert response.status == JobStatus.cancelled


async def test_cancel_generation_job_endpoint_404_for_missing(tmp_path: Path) -> None:
    manager, *_ = make_manager(tmp_path)

    with pytest.raises(HTTPException) as exc_info:
        await cancel_generation_job(job_id="missing", generation_jobs=manager)

    assert exc_info.value.status_code == 404


async def test_cancel_generation_job_endpoint_409_for_finished(tmp_path: Path) -> None:
    manager, *_ = make_manager(tmp_path)
    job = make_generation_job()
    job.status = JobStatus.completed
    manager.jobs[job.id] = job

    with pytest.raises(HTTPException) as exc_info:
        await cancel_generation_job(job_id=job.id, generation_jobs=manager)

    assert exc_info.value.status_code == 409


# ---------------------------------------------------------------------------
# GET /generation/jobs/{id}/download
# ---------------------------------------------------------------------------


async def test_download_generation_job_returns_404_when_unknown(tmp_path: Path) -> None:
    manager, *_ = make_manager(tmp_path)

    with pytest.raises(HTTPException) as exc_info:
        await download_generation_job(job_id="missing", generation_jobs=manager)

    assert exc_info.value.status_code == 404


async def test_download_generation_job_returns_409_when_not_completed(tmp_path: Path) -> None:
    manager, *_ = make_manager(tmp_path)
    job = make_generation_job()
    manager.jobs[job.id] = job

    with pytest.raises(HTTPException) as exc_info:
        await download_generation_job(job_id=job.id, generation_jobs=manager)

    assert exc_info.value.status_code == 409


async def test_download_generation_job_returns_file_response_for_completed_job(tmp_path: Path) -> None:
    manager, *_ = make_manager(tmp_path)
    output_path = manager.settings.outputs_path / "out.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"png")
    job = make_generation_job(status=JobStatus.completed, output_path=output_path)
    manager.jobs[job.id] = job

    response = await download_generation_job(job_id=job.id, generation_jobs=manager)

    assert str(response.path) == str(output_path)
    assert response.media_type == "image/png"


# ---------------------------------------------------------------------------
# GET /generation/capabilities
# ---------------------------------------------------------------------------


async def test_generation_capabilities_reports_unavailable_without_deps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.api.routes as routes_module

    monkeypatch.setattr(routes_module, "generation_dependencies_available", lambda: (False, "optimum missing"))
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)

    response = await generation_capabilities(registry=registry, devices_service=FakeDevicesService())

    assert response.available is False
    assert response.reason == "optimum missing"
    assert response.cpu_only is True
    assert response.models == []


async def test_generation_capabilities_lists_models_and_cpu_only_flag(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    register_generation_model(registry, settings)

    response = await generation_capabilities(
        registry=registry,
        devices_service=FakeDevicesService([{"id": "cpu", "kind": "cpu", "name": "CPU", "backend": "cpu"}]),
    )

    assert response.available is True
    assert [m.model_dump() for m in response.models] == [{"id": MODEL_ID, "name": "amd/sd15"}]
    assert response.cpu_only is True


async def test_generation_capabilities_cpu_only_false_when_gpu_present(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)

    response = await generation_capabilities(
        registry=registry,
        devices_service=FakeDevicesService(
            [
                {"id": "cpu", "kind": "cpu", "name": "CPU", "backend": "cpu"},
                {"id": "dml:0", "kind": "gpu", "name": "Fake GPU", "backend": "directml"},
            ]
        ),
    )

    assert response.cpu_only is False


# ---------------------------------------------------------------------------
# POST /generation/models (install) + GET /generation/models/install/{id}
# ---------------------------------------------------------------------------


class FakeGenerationInstaller:
    def __init__(self) -> None:
        self.install_calls: list[str] = []
        self.install_id = "install-123"
        self.install_error: Exception | None = None
        self._jobs: dict[str, InstallJob] = {}

    def seed_job(self, job: InstallJob) -> None:
        self._jobs[job.id] = job

    async def install_from_hf(self, repo_id: str) -> str:
        self.install_calls.append(repo_id)
        if self.install_error:
            raise self.install_error
        return self.install_id

    def status(self, install_id: str) -> InstallJob | None:
        return self._jobs.get(install_id)


async def test_install_generation_model_returns_install_id_and_status_url() -> None:
    from app.api.routes import install_generation_model

    installer = FakeGenerationInstaller()

    response = await install_generation_model(payload=InstallModelRequest(repo_id="amd/sd15"), installer=installer)

    assert installer.install_calls == ["amd/sd15"]
    assert response.install_id == "install-123"
    assert response.status_url == "/api/v1/generation/models/install/install-123"


async def test_install_generation_model_returns_400_for_invalid_repo_id() -> None:
    from app.api.routes import install_generation_model

    installer = FakeGenerationInstaller()
    installer.install_error = ValueError("optimum missing")

    with pytest.raises(HTTPException) as exc_info:
        await install_generation_model(payload=InstallModelRequest(repo_id="amd/sd15"), installer=installer)

    assert exc_info.value.status_code == 400


async def test_get_generation_install_status_returns_404_for_unknown_id() -> None:
    installer = FakeGenerationInstaller()

    with pytest.raises(HTTPException) as exc_info:
        await get_generation_install_status(install_id="missing", installer=installer)

    assert exc_info.value.status_code == 404


async def test_get_generation_install_status_maps_job_fields() -> None:
    installer = FakeGenerationInstaller()
    installer.seed_job(
        InstallJob(
            id="install-123", repo_id="amd/sd15", status=InstallStatus.downloading,
            progress_pct=12.5, model_id=None, error=None,
        )
    )

    response = await get_generation_install_status(install_id="install-123", installer=installer)

    assert response.install_id == "install-123"
    assert response.repo_id == "amd/sd15"
    assert response.status == "downloading"
    assert response.progress_pct == 12.5


# ---------------------------------------------------------------------------
# Schema validation (pydantic-level, no manager involved)
# ---------------------------------------------------------------------------


def test_create_generation_job_request_accepts_camel_case_fields() -> None:
    payload = CreateGenerationJobRequest.model_validate(
        {"prompt": "x", "modelId": MODEL_ID, "negativePrompt": "y", "autoUpscale": True}
    )

    assert payload.model_id == MODEL_ID
    assert payload.negative_prompt == "y"
    assert payload.auto_upscale is True


# ---------------------------------------------------------------------------
# Wiring: TestClient(app) exercises the real main.py lifespan
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_with_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        main_module, "GenerationEngine", lambda settings, gpu_coordinator: FakeGenerationEngine()
    )
    with TestClient(app) as test_client:
        registry = app.state.model_registry
        register_generation_model(registry, registry.settings)
        yield test_client


def test_create_generation_job_validates_steps_cap(client) -> None:
    response = client.post("/api/v1/generation/jobs", json={
        "prompt": "x", "modelId": "gen--amd--sd15", "steps": 101,
    })
    assert response.status_code == 422


def test_create_generation_job_validates_dimension_multiple(client) -> None:
    response = client.post("/api/v1/generation/jobs", json={
        "prompt": "x", "modelId": "gen--amd--sd15", "width": 500,
    })
    assert response.status_code == 422


def test_get_generation_job_unknown_returns_404(client) -> None:
    assert client.get("/api/v1/generation/jobs/nope").status_code == 404


def test_capabilities_reports_unavailable_without_deps(client, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.routes.generation_dependencies_available", lambda: (False, "optimum missing")
    )
    payload = client.get("/api/v1/generation/capabilities").json()
    assert payload["available"] is False
    assert "optimum" in payload["reason"]


def test_capabilities_lists_installed_models_and_cpu_only_flag(client_with_model) -> None:
    payload = client_with_model.get("/api/v1/generation/capabilities").json()
    assert payload["available"] is True
    assert payload["models"] == [{"id": "gen--amd--sd15", "name": "amd/sd15"}]
    assert isinstance(payload["cpuOnly"], bool)


def test_create_and_poll_generation_job_roundtrip(client_with_model) -> None:
    created = client_with_model.post("/api/v1/generation/jobs", json={
        "prompt": "a red apple", "modelId": "gen--amd--sd15", "device": "cpu",
    })
    assert created.status_code in (200, 201)
    job_id = created.json()["id"]
    polled = client_with_model.get(f"/api/v1/generation/jobs/{job_id}").json()
    assert polled["prompt"] == "a red apple"
    assert polled["status"] in ("queued", "running", "completed")


def test_install_endpoint_wired_accepts_camel_case_body() -> None:
    from app.api.routes import get_generation_installer

    fake_installer = FakeGenerationInstaller()
    app.dependency_overrides[get_generation_installer] = lambda: fake_installer
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/generation/models", json={"repoId": "amd/sd15"})
    finally:
        app.dependency_overrides.pop(get_generation_installer, None)

    assert response.status_code == 202
    body = response.json()
    assert body["installId"] == "install-123"
    assert body["statusUrl"] == "/api/v1/generation/models/install/install-123"
    assert fake_installer.install_calls == ["amd/sd15"]


def test_lifespan_wires_generation_managers_into_app_state() -> None:
    with TestClient(app):
        assert isinstance(app.state.generation_job_manager, GenerationJobManager)
        assert isinstance(app.state.generation_installer, GenerationModelInstaller)
        assert app.state.retention_sweeper.generation_job_manager is app.state.generation_job_manager
