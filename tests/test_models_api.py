from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.routes import (
    delete_model,
    get_hf_client,
    get_install_status,
    get_model_installer,
    install_model,
    list_models,
    search_models,
)
from app.config import Settings
from app.exceptions import ModelNotFoundError, ModelProtectedError
from app.main import app
from app.schemas import InstallModelRequest
from app.services.hf_client import HfModelSummary
from app.services.model_installer import InstallJob, InstallStatus
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus

# ---------------------------------------------------------------------------
# SP1 Task 5 - API: /api/v1/models*. Route-level tests call the route
# coroutines directly with fake service doubles (mirrors tests/test_queue_full.py),
# so no real network/onnxruntime is ever touched; a handful of TestClient(app)
# tests verify main.py's lifespan actually wires ModelRegistry/HfClient/
# ModelInstaller into app.state and that request/response bodies serialize
# to the documented camelCase shape.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    kwargs: dict[str, object] = {"RUNTIME_DIR": str(tmp_path / "runtime")}
    kwargs.update(overrides)
    return Settings(_env_file=None, **kwargs)


def make_onnx_entry(**overrides: object) -> ModelEntry:
    defaults: dict[str, object] = {
        "id": "swinir-real-sr-x4",
        "name": "SwinIR Real SR x4",
        "kind": ModelKind.onnx,
        "source": "https://huggingface.co/example/swinir-real-sr-x4",
        "size_bytes": 12_345,
        "scale": 4,
        "arch": "swinir",
        "file_path": "onnx/swinir-real-sr-x4.onnx",
        "status": ModelStatus.installed,
    }
    defaults.update(overrides)
    return ModelEntry(**defaults)


class FakeHfClient:
    def __init__(self, results: list[HfModelSummary] | None = None, error: Exception | None = None) -> None:
        self.results = results or []
        self.error = error
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, limit: int = 20) -> list[HfModelSummary]:
        self.calls.append((query, limit))
        if self.error:
            raise self.error
        return self.results


class FakeInstaller:
    def __init__(self) -> None:
        self.install_calls: list[str] = []
        self.delete_calls: list[str] = []
        self.install_id = "install-123"
        self.install_error: Exception | None = None
        self.delete_error: Exception | None = None
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

    async def delete(self, model_id: str) -> None:
        self.delete_calls.append(model_id)
        if self.delete_error:
            raise self.delete_error


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------


async def test_list_models_returns_all_registry_entries(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    registry.register(make_onnx_entry())

    response = await list_models(registry=registry)

    ids = {item.id for item in response.models}
    assert "swinir-real-sr-x4" in ids
    assert "realesrgan-x4plus" in ids
    onnx_item = next(item for item in response.models if item.id == "swinir-real-sr-x4")
    assert onnx_item.kind == "onnx"
    assert onnx_item.status == "installed"
    assert onnx_item.scale == 4


# ---------------------------------------------------------------------------
# GET /models/search
# ---------------------------------------------------------------------------


async def test_search_models_maps_results_to_camel_case() -> None:
    hf_client = FakeHfClient(
        results=[
            HfModelSummary(
                id="Kim2091/2x-AnimeSharpV4",
                author="Kim2091",
                pipeline_tag="image-to-image",
                downloads=10,
                likes=5,
                tags=("onnx", "super-resolution"),
            )
        ]
    )

    response = await search_models(q="anime", hf_client=hf_client)

    assert hf_client.calls == [("anime", 20)]
    assert len(response.results) == 1
    result = response.results[0]
    assert result.id == "Kim2091/2x-AnimeSharpV4"
    assert result.pipeline_tag == "image-to-image"
    assert result.tags == ["onnx", "super-resolution"]


async def test_search_models_raises_502_when_hf_client_fails() -> None:
    hf_client = FakeHfClient(error=RuntimeError("hub unreachable"))

    with pytest.raises(HTTPException) as exc_info:
        await search_models(q="anime", hf_client=hf_client)

    assert exc_info.value.status_code == 502


# ---------------------------------------------------------------------------
# POST /models/install
# ---------------------------------------------------------------------------


async def test_install_model_returns_install_id_and_status_url() -> None:
    installer = FakeInstaller()

    response = await install_model(
        payload=InstallModelRequest(repo_id="org/repo"), installer=installer
    )

    assert installer.install_calls == ["org/repo"]
    assert response.install_id == "install-123"
    assert response.status_url == "/api/v1/models/install/install-123"


async def test_install_model_accepts_camel_case_repo_id_field() -> None:
    payload = InstallModelRequest.model_validate({"repoId": "org/repo"})

    assert payload.repo_id == "org/repo"


async def test_install_model_returns_400_for_invalid_repo_id() -> None:
    installer = FakeInstaller()
    installer.install_error = ValueError("repo_id must look like 'owner/name'")

    with pytest.raises(HTTPException) as exc_info:
        await install_model(payload=InstallModelRequest(repo_id="../../etc"), installer=installer)

    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# GET /models/install/{install_id}
# ---------------------------------------------------------------------------


async def test_get_install_status_returns_404_for_unknown_id() -> None:
    installer = FakeInstaller()

    with pytest.raises(HTTPException) as exc_info:
        await get_install_status(install_id="does-not-exist", installer=installer)

    assert exc_info.value.status_code == 404


async def test_get_install_status_maps_job_fields() -> None:
    installer = FakeInstaller()
    installer.seed_job(
        InstallJob(
            id="install-123",
            repo_id="org/repo",
            status=InstallStatus.validating,
            progress_pct=42.5,
            model_id=None,
            error=None,
        )
    )

    response = await get_install_status(install_id="install-123", installer=installer)

    assert response.install_id == "install-123"
    assert response.repo_id == "org/repo"
    assert response.status == "validating"
    assert response.progress_pct == 42.5
    assert response.model_id is None


# ---------------------------------------------------------------------------
# DELETE /models/{model_id}
# ---------------------------------------------------------------------------


async def test_delete_model_returns_404_for_unknown_model() -> None:
    installer = FakeInstaller()
    installer.delete_error = ModelNotFoundError("Unknown model id: 'ghost'")

    with pytest.raises(HTTPException) as exc_info:
        await delete_model(model_id="ghost", installer=installer)

    assert exc_info.value.status_code == 404


async def test_delete_model_returns_409_for_builtin_model() -> None:
    installer = FakeInstaller()
    installer.delete_error = ModelProtectedError("Cannot remove builtin model: 'realesrgan-x4plus'")

    with pytest.raises(HTTPException) as exc_info:
        await delete_model(model_id="realesrgan-x4plus", installer=installer)

    assert exc_info.value.status_code == 409


async def test_delete_model_returns_204_on_success() -> None:
    installer = FakeInstaller()

    response = await delete_model(model_id="swinir-real-sr-x4", installer=installer)

    assert installer.delete_calls == ["swinir-real-sr-x4"]
    assert response.status_code == 204


# ---------------------------------------------------------------------------
# Wiring: TestClient(app) exercises the real main.py lifespan
# ---------------------------------------------------------------------------


def test_models_endpoint_wired_through_app_state_lists_builtins() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/models")

    assert response.status_code == 200
    ids = {item["id"] for item in response.json()["models"]}
    assert "realesrgan-x4plus" in ids


def test_search_endpoint_wired_and_returns_camel_case() -> None:
    fake_hf_client = FakeHfClient(
        results=[
            HfModelSummary(
                id="org/repo",
                author="org",
                pipeline_tag="image-to-image",
                downloads=1,
                likes=1,
                tags=("onnx",),
            )
        ]
    )
    app.dependency_overrides[get_hf_client] = lambda: fake_hf_client
    try:
        with TestClient(app) as client:
            response = client.get("/api/v1/models/search", params={"q": "anime"})
    finally:
        app.dependency_overrides.pop(get_hf_client, None)

    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["pipelineTag"] == "image-to-image"


def test_install_endpoint_wired_accepts_camel_case_body() -> None:
    fake_installer = FakeInstaller()
    app.dependency_overrides[get_model_installer] = lambda: fake_installer
    try:
        with TestClient(app) as client:
            response = client.post("/api/v1/models/install", json={"repoId": "org/repo"})
    finally:
        app.dependency_overrides.pop(get_model_installer, None)

    assert response.status_code == 202
    body = response.json()
    assert body["installId"] == "install-123"
    assert body["statusUrl"] == "/api/v1/models/install/install-123"
    assert fake_installer.install_calls == ["org/repo"]


def test_delete_endpoint_wired_returns_409_for_builtin() -> None:
    fake_installer = FakeInstaller()
    fake_installer.delete_error = ModelProtectedError("Cannot remove builtin model: 'realesrgan-x4plus'")
    app.dependency_overrides[get_model_installer] = lambda: fake_installer
    try:
        with TestClient(app) as client:
            response = client.delete("/api/v1/models/realesrgan-x4plus")
    finally:
        app.dependency_overrides.pop(get_model_installer, None)

    assert response.status_code == 409
