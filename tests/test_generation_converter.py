from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import app.services.generation_converter as generation_converter_module
from app.models import JobStatus
from app.services.device_semaphores import DeviceSemaphores
from app.services.generation_converter import (
    GenerationModelConverter,
    _parse_submodel_line,
    _select_conversion_files,
)
from app.services.generation_installer import (
    GenerationModelInstaller,
    _generation_model_id,
)
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.hf_client import HfFile
from app.services.model_registry import ModelRegistry
from test_generation_installer import (
    FakeHfClient,
    FakeValidationPipeline,
    make_settings,
)


SOURCE_MODEL_INDEX = json.dumps(
    {
        "_class_name": "StableDiffusionXLPipeline",
        "unet": ["diffusers", "UNet2DConditionModel"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
)


def _pytorch_repo_files() -> list[HfFile]:
    return [
        HfFile(path="model_index.json", size=100),
        HfFile(path="unet/diffusion_pytorch_model.safetensors", size=1000),
        HfFile(path="unet/config.json", size=10),
        HfFile(path="vae/diffusion_pytorch_model.safetensors", size=500),
        HfFile(path="vae/config.json", size=10),
    ]


def make_converter(tmp_path: Path, export_fn):
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    hf_client = FakeHfClient(files=_pytorch_repo_files())
    hf_client.download_bytes_by_path = {
        "model_index.json": SOURCE_MODEL_INDEX.encode("utf-8")
    }
    installer = GenerationModelInstaller(
        settings,
        registry,
        hf_client,
        GpuSessionCoordinator(),
        DeviceSemaphores(settings),
    )
    installer._create_validation_pipeline = (  # type: ignore[method-assign]
        lambda pipeline_dir: FakeValidationPipeline()
    )
    converter = GenerationModelConverter(
        settings,
        installer,
        hf_client,
        export_fn=export_fn,
    )
    return converter, installer, settings, registry


def convert_and_drain(converter: GenerationModelConverter, repo_id: str) -> str:
    async def _run() -> str:
        conversion_id = await converter.convert_from_hf(repo_id)
        await converter._process_next()
        return conversion_id

    return asyncio.run(_run())


async def test_convert_from_hf_rejects_missing_generation_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    converter, _installer, _settings, _registry = make_converter(
        tmp_path, export_fn=fake_export_ok
    )
    monkeypatch.setattr(
        generation_converter_module,
        "generation_dependencies_available",
        lambda: (False, "optimum y torch no están disponibles"),
    )

    with pytest.raises(
        ValueError, match="optimum y torch no están disponibles"
    ):
        await converter.convert_from_hf("amd/sdxl-torch")

    assert converter._queue.empty()


def test_parse_submodel_line_uses_index_to_keep_duplicate_classes_unique() -> None:
    assert (
        _parse_submodel_line(
            "***** Exporting submodel 3/4: AutoencoderKL *****"
        )
        == "3-AutoencoderKL"
    )


def test_parse_submodel_line_ignores_unrelated_output() -> None:
    assert _parse_submodel_line("Validating ONNX model unet/model.onnx") is None


def test_select_conversion_files_keeps_torch_weights_in_declared_dirs() -> None:
    files = _pytorch_repo_files() + [
        HfFile(path="unet/duplicate.ckpt", size=999),
        HfFile(path="undeclared/x.safetensors", size=999),
    ]
    kept = _select_conversion_files(files, ["unet", "vae"])
    paths = {f.path for f in kept}
    assert "unet/diffusion_pytorch_model.safetensors" in paths
    assert "vae/config.json" in paths
    assert "unet/duplicate.ckpt" not in paths
    assert "undeclared/x.safetensors" not in paths
    assert "model_index.json" not in paths


def test_select_conversion_files_prefers_safetensors_over_bin_in_same_dir() -> None:
    files = [
        HfFile(path="unet/diffusion_pytorch_model.safetensors", size=1000),
        HfFile(path="unet/diffusion_pytorch_model.bin", size=1000),
    ]
    kept = _select_conversion_files(files, ["unet"])
    assert [f.path for f in kept] == [
        "unet/diffusion_pytorch_model.safetensors"
    ]


EXPORTED_LOG: list[str] = []


def fake_export_ok(src_dir: Path, out_dir: Path, on_component) -> list[str]:
    assert (src_dir / "model_index.json").exists()
    (out_dir / "unet").mkdir(parents=True)
    (out_dir / "model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "StableDiffusionXLPipeline",
                "unet": ["diffusers", "x"],
            }
        ),
        encoding="utf-8",
    )
    for name in ("unet", "vae"):
        on_component(name)
        EXPORTED_LOG.append(name)
    return ["unet", "vae"]


def test_convert_happy_path_exports_and_promotes(tmp_path, monkeypatch) -> None:
    EXPORTED_LOG.clear()
    converter, installer, settings, registry = make_converter(
        tmp_path, export_fn=fake_export_ok
    )
    job_id = convert_and_drain(converter, "amd/sdxl-torch")
    job = converter.status(job_id)
    assert job is not None
    assert job.status == JobStatus.completed
    assert EXPORTED_LOG == ["unet", "vae"]
    assert registry.get(job.model_id) is not None
    assert job.metadata["progress"] == 1.0


def test_convert_export_failure_leaves_no_orphans(tmp_path) -> None:
    def failing_export(src_dir, out_dir, on_component):
        raise RuntimeError("export reventó a mitad del unet")

    converter, installer, settings, registry = make_converter(
        tmp_path, export_fn=failing_export
    )
    job_id = convert_and_drain(converter, "amd/sdxl-torch")
    job = converter.status(job_id)
    assert job is not None
    assert job.status == JobStatus.failed
    assert "export reventó" in (job.error or "")
    assert registry.get(_generation_model_id("amd/sdxl-torch")) is None
    leftovers = (
        [path for path in settings.temp_path.iterdir()]
        if settings.temp_path.exists()
        else []
    )
    assert leftovers == []


def test_converted_result_goes_through_real_validation(
    tmp_path, monkeypatch
) -> None:
    def explode(pipeline_dir):
        raise RuntimeError("pipeline inválido")

    EXPORTED_LOG.clear()
    converter, installer, settings, registry = make_converter(
        tmp_path, export_fn=fake_export_ok
    )
    monkeypatch.setattr(installer, "_create_validation_pipeline", explode)
    job_id = convert_and_drain(converter, "amd/sdxl-torch")
    job = converter.status(job_id)
    assert job is not None
    assert job.status == JobStatus.failed
    assert registry.get(_generation_model_id("amd/sdxl-torch")) is None
