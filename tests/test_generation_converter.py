from __future__ import annotations

import asyncio
import errno
import json
from pathlib import Path

import pytest

import app.services.generation_converter as generation_converter_module
from app.models import JobStatus
from app.services.device_semaphores import DeviceSemaphores
from app.services.generation_converter import (
    GenerationModelConverter,
    _parse_submodel_line,
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


def _pytorch_repo_files_with_fp16() -> list[HfFile]:
    return _pytorch_repo_files() + [
        HfFile(
            path="unet/diffusion_pytorch_model.fp16.safetensors",
            size=500,
        ),
        HfFile(
            path="vae/diffusion_pytorch_model.fp16.safetensors",
            size=250,
        ),
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


EXPORTED_LOG: list[str] = []


def fake_export_ok(
    src_dir: Path,
    out_dir: Path,
    on_component,
    dtype=None,
    atol=None,
) -> list[str]:
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
    def failing_export(
        src_dir,
        out_dir,
        on_component,
        dtype=None,
        atol=None,
    ):
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


def test_conversion_maps_disk_full_to_actionable_error(tmp_path: Path) -> None:
    converter, _installer, _settings, _registry = make_converter(
        tmp_path,
        export_fn=fake_export_ok,
    )
    disk_full = OSError(errno.ENOSPC, "No space left on device")
    disk_full.filename = str(tmp_path / "unet" / "model.safetensors.part")
    converter.hf_client.download_error = disk_full

    job_id = convert_and_drain(converter, "amd/sdxl-torch")
    job = converter.status(job_id)

    assert job is not None
    assert job.status == JobStatus.failed
    assert "espacio" in (job.error or "").lower()


def test_conversion_downloads_without_a_generation_ceiling(
    tmp_path: Path,
) -> None:
    converter, _installer, _settings, _registry = make_converter(
        tmp_path,
        export_fn=fake_export_ok,
    )

    job_id = convert_and_drain(converter, "amd/sdxl-torch")
    job = converter.status(job_id)

    assert job is not None
    assert job.status == JobStatus.completed
    assert all(converter.hf_client.download_unlimited)


def _recording_export(captured: dict[str, object]):
    def fake_export(
        src_dir,
        out_dir,
        on_component,
        dtype=None,
        atol=None,
    ):
        captured["dtype"] = dtype
        captured["atol"] = atol
        (out_dir / "model_index.json").write_text(
            SOURCE_MODEL_INDEX,
            encoding="utf-8",
        )
        for component in ("unet", "vae"):
            (out_dir / component).mkdir(parents=True, exist_ok=True)
            (out_dir / component / "model.onnx").write_bytes(b"onnx")
        return ["unet", "vae"]

    return fake_export


async def test_fp16_weights_are_staged_under_the_canonical_name(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    converter, _installer, _settings, _registry = make_converter(
        tmp_path,
        _recording_export(captured),
    )
    converter.hf_client.files = _pytorch_repo_files_with_fp16()
    staged: list[str] = []
    original_download = converter.hf_client.download

    async def recording_download(
        repo_id,
        filename,
        dest,
        progress_cb=None,
        max_bytes=None,
        unlimited=False,
    ):
        staged.append(dest.name)
        return await original_download(
            repo_id,
            filename,
            dest,
            progress_cb,
            max_bytes,
            unlimited,
        )

    converter.hf_client.download = recording_download  # type: ignore[method-assign]

    conversion_id = await converter.convert_from_hf(
        "owner/name",
        precision="fp16",
    )
    await converter._process_next()

    assert converter.status(conversion_id).status is JobStatus.completed
    assert "diffusion_pytorch_model.safetensors" in staged
    assert not any(".fp16." in name for name in staged)


async def test_export_receives_fp16_dtype_and_relaxed_atol(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    converter, *_ = make_converter(
        tmp_path,
        _recording_export(captured),
    )
    converter.hf_client.files = _pytorch_repo_files_with_fp16()

    await converter.convert_from_hf("owner/name", precision="fp16")
    await converter._process_next()

    assert captured["dtype"] == "fp16"
    assert captured["atol"] == pytest.approx(1e-2)


async def test_fp32_export_passes_no_dtype_and_no_atol_override(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    converter, *_ = make_converter(
        tmp_path,
        _recording_export(captured),
    )
    converter.hf_client.files = _pytorch_repo_files_with_fp16()

    await converter.convert_from_hf("owner/name", precision="fp32")
    await converter._process_next()

    assert captured["dtype"] is None
    assert captured["atol"] is None


async def test_precision_falls_back_when_repo_lacks_requested_variant(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    converter, *_ = make_converter(
        tmp_path,
        _recording_export(captured),
    )
    converter.hf_client.files = _pytorch_repo_files()

    conversion_id = await converter.convert_from_hf(
        "owner/name",
        precision="fp16",
    )
    await converter._process_next()

    assert converter.status(conversion_id).status is JobStatus.completed
    assert captured["dtype"] is None
