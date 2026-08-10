from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api.routes import optimize_generation_model
from app.config import Settings
from app.services.generation_optimize import GIB, optimized_model_id
from app.services.hf_client import HfFile
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus

MODEL_ID = "gen--owner--sdxl"
REPO_ID = "owner/sdxl"


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)


class _RamProbe:
    def __init__(self, free_mb: int | None) -> None:
        self._free_mb = free_mb

    def free_capacity_mb(self, _device_id: str) -> int | None:
        return self._free_mb

    def own_usage_mb(self, _device_id: str) -> int | None:
        return None


def make_request(free_ram_bytes: int | None = 64 * GIB):
    free_mb = None if free_ram_bytes is None else free_ram_bytes // (1024 * 1024)
    state = types.SimpleNamespace(resource_probes={"cpu": _RamProbe(free_mb)})
    return types.SimpleNamespace(app=types.SimpleNamespace(state=state))


class FakeConverter:
    def __init__(self, files: list[HfFile]) -> None:
        self.hf_client = types.SimpleNamespace(repo_files=self._repo_files)
        self.files = files
        self.calls: list[dict] = []

    async def _repo_files(self, _repo_id: str) -> list[HfFile]:
        return self.files

    async def optimize_installed(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return "conv-optimize-1"


def torch_repo_files() -> list[HfFile]:
    return [
        HfFile(path="model_index.json", size=10),
        HfFile(path="unet/diffusion_pytorch_model.safetensors", size=1000),
    ]


def onnx_only_repo_files() -> list[HfFile]:
    return [
        HfFile(path="model_index.json", size=10),
        HfFile(path="unet/model.onnx", size=1000),
    ]


def register_installed(
    registry: ModelRegistry,
    settings: Settings,
    *,
    model_id: str = MODEL_ID,
    declared_class: str = "StableDiffusionXLPipeline",
    source: str = f"hf:{REPO_ID}",
    status: ModelStatus = ModelStatus.installed,
    checkpoint_path: str | None = None,
) -> ModelEntry:
    model_dir = settings.models_path / "generation" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": declared_class}), encoding="utf-8"
    )
    entry = ModelEntry(
        id=model_id,
        name=REPO_ID,
        kind=ModelKind.diffusion_onnx,
        source=source,
        size_bytes=1,
        file_path=f"generation/{model_id}",
        checkpoint_path=checkpoint_path,
        status=status,
    )
    registry.register(entry)
    return entry


def make_case(tmp_path: Path, files: list[HfFile] | None = None, **register_kwargs):
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    register_installed(registry, settings, **register_kwargs)
    return FakeConverter(files or torch_repo_files()), registry, settings


async def test_optimize_enqueues_with_the_installed_model_as_source(tmp_path: Path) -> None:
    converter, registry, settings = make_case(tmp_path)

    response = await optimize_generation_model(
        MODEL_ID, make_request(), converter, registry, settings
    )

    assert response.conversion_id == "conv-optimize-1"
    assert response.status_url.endswith("/generation/models/convert/conv-optimize-1")
    call = converter.calls[0]
    assert call["source_model_id"] == MODEL_ID
    assert call["repo_id"] == REPO_ID
    assert call["installed_dir"] == settings.models_path / "generation" / MODEL_ID


async def test_optimize_rejects_an_unknown_model(tmp_path: Path) -> None:
    converter, registry, settings = make_case(tmp_path)

    with pytest.raises(HTTPException) as excinfo:
        await optimize_generation_model(
            "gen--nope", make_request(), converter, registry, settings
        )

    assert excinfo.value.status_code == 404
    assert converter.calls == []


async def test_optimize_rejects_a_model_that_is_still_converting(tmp_path: Path) -> None:
    converter, registry, settings = make_case(tmp_path, status=ModelStatus.converting)

    with pytest.raises(HTTPException) as excinfo:
        await optimize_generation_model(
            MODEL_ID, make_request(), converter, registry, settings
        )

    assert excinfo.value.status_code == 400
    assert "instalarse" in excinfo.value.detail


async def test_optimize_rejects_a_model_without_a_hugging_face_origin(
    tmp_path: Path,
) -> None:
    # Sin repo de origen no hay pesos torch que re-exportar en fp32, y fusionar
    # sobre el ONNX fp16 instalado no fusiona nada (medido: 0.96x).
    converter, registry, settings = make_case(tmp_path, source="local:disk")

    with pytest.raises(HTTPException) as excinfo:
        await optimize_generation_model(
            MODEL_ID, make_request(), converter, registry, settings
        )

    assert excinfo.value.status_code == 400
    assert "origen" in excinfo.value.detail


async def test_optimize_rejects_a_repo_that_only_publishes_onnx_weights(
    tmp_path: Path,
) -> None:
    converter, registry, settings = make_case(tmp_path, files=onnx_only_repo_files())

    with pytest.raises(HTTPException) as excinfo:
        await optimize_generation_model(
            MODEL_ID, make_request(), converter, registry, settings
        )

    assert excinfo.value.status_code == 400
    assert "PyTorch" in excinfo.value.detail


async def test_optimize_admits_a_single_file_install_without_repo_torch_weights(
    tmp_path: Path,
) -> None:
    # Instalado desde un checkpoint suelto: los pesos torch son ESE archivo, así
    # que el listado del repo no tiene por qué traer un árbol diffusers.
    converter, registry, settings = make_case(
        tmp_path, files=onnx_only_repo_files(), checkpoint_path="model.safetensors"
    )

    await optimize_generation_model(MODEL_ID, make_request(), converter, registry, settings)

    assert converter.calls[0]["checkpoint_path"] == "model.safetensors"


async def test_optimize_rejects_an_architecture_the_fusion_cannot_guarantee(
    tmp_path: Path,
) -> None:
    converter, registry, settings = make_case(
        tmp_path, declared_class="StableDiffusion3Pipeline"
    )

    with pytest.raises(HTTPException) as excinfo:
        await optimize_generation_model(
            MODEL_ID, make_request(), converter, registry, settings
        )

    assert excinfo.value.status_code == 400
    assert "StableDiffusion3Pipeline" in excinfo.value.detail


async def test_optimize_rejects_before_enqueuing_when_ram_is_short(tmp_path: Path) -> None:
    converter, registry, settings = make_case(tmp_path)

    with pytest.raises(HTTPException) as excinfo:
        await optimize_generation_model(
            MODEL_ID, make_request(free_ram_bytes=8 * GIB), converter, registry, settings
        )

    assert excinfo.value.status_code == 400
    assert "50.0 GiB" in excinfo.value.detail
    assert "8.0 GiB" in excinfo.value.detail
    assert converter.calls == []


async def test_optimize_admits_when_ram_cannot_be_measured(tmp_path: Path) -> None:
    converter, registry, settings = make_case(tmp_path)

    await optimize_generation_model(
        MODEL_ID, make_request(free_ram_bytes=None), converter, registry, settings
    )

    assert len(converter.calls) == 1


async def test_optimize_rejects_a_model_that_already_has_an_optimized_variant(
    tmp_path: Path,
) -> None:
    converter, registry, settings = make_case(tmp_path)
    register_installed(registry, settings, model_id=optimized_model_id(MODEL_ID))

    with pytest.raises(HTTPException) as excinfo:
        await optimize_generation_model(
            MODEL_ID, make_request(), converter, registry, settings
        )

    assert excinfo.value.status_code == 400
    assert "optimizada" in excinfo.value.detail


async def test_optimize_rejects_optimizing_the_optimized_variant_itself(
    tmp_path: Path,
) -> None:
    converter, registry, settings = make_case(tmp_path, model_id=optimized_model_id(MODEL_ID))

    with pytest.raises(HTTPException) as excinfo:
        await optimize_generation_model(
            optimized_model_id(MODEL_ID), make_request(), converter, registry, settings
        )

    assert excinfo.value.status_code == 400
    assert "ya es la versión optimizada" in excinfo.value.detail


async def test_optimize_rejects_an_inpainting_merge_variant(tmp_path: Path) -> None:
    # Sus pesos son el resultado del merge: re-exportar desde el repo daría el
    # UNet SIN mergear y la "optimizada" sería otro modelo disfrazado.
    inpaint_id = f"{MODEL_ID}--inpainting"
    converter, registry, settings = make_case(tmp_path, model_id=inpaint_id)

    with pytest.raises(HTTPException) as excinfo:
        await optimize_generation_model(
            inpaint_id, make_request(), converter, registry, settings
        )

    assert excinfo.value.status_code == 400
    assert "inpainting" in excinfo.value.detail
    assert converter.calls == []


async def test_optimize_retries_after_a_previous_failure(tmp_path: Path) -> None:
    converter, registry, settings = make_case(tmp_path)
    register_installed(
        registry, settings, model_id=optimized_model_id(MODEL_ID), status=ModelStatus.error
    )

    await optimize_generation_model(MODEL_ID, make_request(), converter, registry, settings)

    assert len(converter.calls) == 1
