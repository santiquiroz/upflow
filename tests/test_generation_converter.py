from __future__ import annotations

import asyncio
import errno
import json
import os
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
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus
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
        "model_index.json": SOURCE_MODEL_INDEX.encode("utf-8"),
        "vae/config.json": json.dumps(
            {
                "_class_name": "AutoencoderKL",
                "sample_size": 1024,
                "latent_channels": 4,
            }
        ).encode("utf-8"),
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


def _write_vae_config(src_root: Path, content: str) -> Path:
    config_path = src_root / "vae" / "config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(content, encoding="utf-8")
    return config_path


def test_cap_vae_trace_resolution_rewrites_large_sample_size(
    tmp_path: Path,
) -> None:
    config_path = _write_vae_config(
        tmp_path,
        json.dumps(
            {
                "_class_name": "AutoencoderKL",
                "sample_size": 1024,
                "latent_channels": 4,
            }
        ),
    )

    generation_converter_module._cap_vae_trace_resolution(tmp_path)

    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "_class_name": "AutoencoderKL",
        "sample_size": 512,
        "latent_channels": 4,
    }


def test_cap_vae_trace_resolution_leaves_512_config_byte_identical(
    tmp_path: Path,
) -> None:
    original = '{\n  "sample_size": 512,\n  "latent_channels": 4\n}\n'
    config_path = _write_vae_config(tmp_path, original)

    generation_converter_module._cap_vae_trace_resolution(tmp_path)

    assert config_path.read_text(encoding="utf-8") == original


def test_cap_vae_trace_resolution_without_vae_is_noop(tmp_path: Path) -> None:
    generation_converter_module._cap_vae_trace_resolution(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_cap_vae_trace_resolution_without_sample_size_is_byte_identical(
    tmp_path: Path,
) -> None:
    original = '{\n  "_class_name": "AutoencoderKL",\n  "latent_channels": 4\n}\n'
    config_path = _write_vae_config(tmp_path, original)

    generation_converter_module._cap_vae_trace_resolution(tmp_path)

    assert config_path.read_text(encoding="utf-8") == original


def test_cap_vae_trace_resolution_with_non_numeric_sample_size_is_byte_identical(
    tmp_path: Path,
) -> None:
    original = '{\n  "sample_size": "1024",\n  "latent_channels": 4\n}\n'
    config_path = _write_vae_config(tmp_path, original)

    generation_converter_module._cap_vae_trace_resolution(tmp_path)

    assert config_path.read_text(encoding="utf-8") == original


def test_disable_vae_force_upcast_rewrites_truthy_value(
    tmp_path: Path,
) -> None:
    config_path = _write_vae_config(
        tmp_path,
        json.dumps(
            {
                "_class_name": "AutoencoderKL",
                "force_upcast": True,
                "latent_channels": 4,
            }
        ),
    )

    generation_converter_module._disable_vae_force_upcast(tmp_path)

    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "_class_name": "AutoencoderKL",
        "force_upcast": False,
        "latent_channels": 4,
    }


def test_disable_vae_force_upcast_false_config_is_not_rewritten(
    tmp_path: Path,
) -> None:
    original = '{\n  "force_upcast": false,\n  "latent_channels": 4\n}\n'
    config_path = _write_vae_config(tmp_path, original)
    preserved_mtime_ns = 1_700_000_000_000_000_000
    os.utime(
        config_path,
        ns=(preserved_mtime_ns, preserved_mtime_ns),
    )

    generation_converter_module._disable_vae_force_upcast(tmp_path)

    assert config_path.read_text(encoding="utf-8") == original
    assert config_path.stat().st_mtime_ns == preserved_mtime_ns


def test_disable_vae_force_upcast_without_vae_is_noop(tmp_path: Path) -> None:
    generation_converter_module._disable_vae_force_upcast(tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_disable_vae_force_upcast_without_flag_is_byte_identical(
    tmp_path: Path,
) -> None:
    original = '{\n  "_class_name": "AutoencoderKL",\n  "latent_channels": 4\n}\n'
    config_path = _write_vae_config(tmp_path, original)

    generation_converter_module._disable_vae_force_upcast(tmp_path)

    assert config_path.read_text(encoding="utf-8") == original


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


def test_conversion_caps_staged_vae_config_before_export(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def inspect_vae_config_then_export(
        src_dir: Path,
        out_dir: Path,
        on_component,
        dtype=None,
        atol=None,
    ) -> list[str]:
        config = json.loads(
            (src_dir / "vae" / "config.json").read_text(encoding="utf-8")
        )
        captured["sample_size"] = config["sample_size"]
        captured["latent_channels"] = config["latent_channels"]
        return fake_export_ok(src_dir, out_dir, on_component, dtype, atol)

    converter, *_ = make_converter(
        tmp_path,
        export_fn=inspect_vae_config_then_export,
    )
    converter.hf_client.download_bytes_by_path["vae/config.json"] = json.dumps(
        {
            "_class_name": "AutoencoderKL",
            "sample_size": 1024,
            "latent_channels": 4,
        }
    ).encode("utf-8")

    job_id = convert_and_drain(converter, "amd/sdxl-torch")

    assert converter.status(job_id).status is JobStatus.completed
    assert captured == {"sample_size": 512, "latent_channels": 4}


def test_conversion_disables_staged_vae_force_upcast_before_export(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def inspect_vae_config_then_export(
        src_dir: Path,
        out_dir: Path,
        on_component,
        dtype=None,
        atol=None,
    ) -> list[str]:
        config = json.loads(
            (src_dir / "vae" / "config.json").read_text(encoding="utf-8")
        )
        captured["force_upcast"] = config["force_upcast"]
        captured["latent_channels"] = config["latent_channels"]
        return fake_export_ok(src_dir, out_dir, on_component, dtype, atol)

    converter, *_ = make_converter(
        tmp_path,
        export_fn=inspect_vae_config_then_export,
    )
    converter.hf_client.download_bytes_by_path["vae/config.json"] = json.dumps(
        {
            "_class_name": "AutoencoderKL",
            "sample_size": 1024,
            "force_upcast": True,
            "latent_channels": 4,
        }
    ).encode("utf-8")

    job_id = convert_and_drain(converter, "amd/sdxl-torch")

    assert converter.status(job_id).status is JobStatus.completed
    assert captured == {"force_upcast": False, "latent_channels": 4}


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
    # El fallo deja rastro VISIBLE (entrada en error, reintentable) pero jamás
    # un modelo roto como instalado.
    entry = registry.get(_generation_model_id("amd/sdxl-torch"))
    assert entry is not None
    assert entry.status == ModelStatus.error
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
    entry = registry.get(_generation_model_id("amd/sdxl-torch"))
    assert entry is not None
    assert entry.status == ModelStatus.error


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


async def test_plain_fp16_weights_keep_requested_fp16_export_dtype(
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    converter, *_ = make_converter(
        tmp_path,
        _recording_export(captured),
    )

    async def read_header(repo_id: str, path: str):
        assert repo_id == "owner/name"
        assert path == "unet/diffusion_pytorch_model.safetensors"
        header = {
            "weight": {
                "dtype": "F16",
                "shape": [1],
                "data_offsets": [0, 2],
            }
        }
        return header, len(json.dumps(header).encode("utf-8"))

    converter.hf_client.read_safetensors_header = read_header  # type: ignore[attr-defined]

    conversion_id = await converter.convert_from_hf(
        "owner/name",
        precision="fp16",
    )
    await converter._process_next()

    assert converter.status(conversion_id).status is JobStatus.completed
    assert captured["dtype"] == "fp16"


def _single_file_header(name: str) -> dict:
    fixture_path = Path(__file__).parent / "assets" / "single_file_fixtures.json"
    entry = json.loads(fixture_path.read_text(encoding="utf-8"))[name]
    header: dict[str, dict] = {}
    for group in ("watched_keys", "role_sample_keys", "lora_sample_keys"):
        for key, shape in entry[group].items():
            header[key] = {"shape": shape, "dtype": "F16"}
    return header


@pytest.mark.asyncio
async def test_single_file_conversion_downloads_materializes_and_promotes_exact_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    converter, _installer, _settings, registry = make_converter(
        tmp_path,
        _recording_export(captured),
    )
    checkpoint = HfFile(path="pony.safetensors", size=6616 * 1024 * 1024)
    converter.hf_client.files = [
        checkpoint,
        HfFile(path="other.safetensors", size=100),
    ]

    async def read_header(repo_id: str, path: str):
        assert path == checkpoint.path
        header = _single_file_header("pony_sdxl")
        return header, len(json.dumps(header).encode("utf-8"))

    converter.hf_client.read_safetensors_header = read_header  # type: ignore[attr-defined]

    def fake_materialize(
        checkpoint_path: Path,
        out_dir: Path,
        architecture: str,
    ) -> None:
        captured["checkpoint_path"] = checkpoint_path
        captured["architecture"] = architecture
        captured["stage"] = converter.status(conversion_id).metadata["stage"]
        captured["stages"] = converter.status(conversion_id).metadata["stages"]
        (out_dir / "model_index.json").write_text(
            SOURCE_MODEL_INDEX,
            encoding="utf-8",
        )

    monkeypatch.setattr(generation_converter_module, "materialize", fake_materialize)

    conversion_id = await converter.convert_from_hf(
        "owner/checkpoints",
        checkpoint_path=checkpoint.path,
    )
    await converter._process_next()

    job = converter.status(conversion_id)
    assert job is not None
    assert job.status is JobStatus.completed, job.error
    assert job.checkpoint_path == checkpoint.path
    assert captured["architecture"] == "xl_base"
    assert Path(captured["checkpoint_path"]).name == checkpoint.path
    assert captured["stage"] == "materializing"
    assert any(
        stage["key"] == "materializing"
        and stage["label"] == "Materializing checkpoint"
        for stage in captured["stages"]
    )
    downloaded = [call[1] for call in converter.hf_client.download_calls]
    assert downloaded == [checkpoint.path]
    assert all(converter.hf_client.download_unlimited)
    expected_model_id = _generation_model_id(
        "owner/checkpoints",
        checkpoint.path,
    )
    assert job.model_id == expected_model_id
    assert registry.get(expected_model_id) is not None


@pytest.mark.asyncio
async def test_single_file_remote_header_failure_reclassifies_downloaded_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    converter, _installer, _settings, _registry = make_converter(
        tmp_path,
        _recording_export(captured),
    )
    header = _single_file_header("pony_sdxl")
    header_bytes = json.dumps(header).encode("utf-8")
    checkpoint = HfFile(path="pony.safetensors", size=len(header_bytes) + 8)
    converter.hf_client.files = [checkpoint]
    converter.hf_client.download_bytes_by_path[checkpoint.path] = (
        len(header_bytes).to_bytes(8, byteorder="little") + header_bytes
    )

    async def fail_remote_header(repo_id: str, path: str):
        raise RuntimeError("range unavailable")

    converter.hf_client.read_safetensors_header = fail_remote_header  # type: ignore[attr-defined]

    def fake_materialize(
        checkpoint_path: Path,
        out_dir: Path,
        architecture: str,
    ) -> None:
        captured["architecture"] = architecture
        (out_dir / "model_index.json").write_text(
            SOURCE_MODEL_INDEX,
            encoding="utf-8",
        )

    monkeypatch.setattr(generation_converter_module, "materialize", fake_materialize)

    conversion_id = await converter.convert_from_hf(
        "owner/checkpoints",
        checkpoint_path=checkpoint.path,
    )
    await converter._process_next()

    job = converter.status(conversion_id)
    assert job is not None
    assert job.status is JobStatus.completed, job.error
    assert captured["architecture"] == "xl_base"


@pytest.mark.asyncio
async def test_single_file_non_installable_header_fails_before_download(
    tmp_path: Path,
) -> None:
    converter, _installer, _settings, _registry = make_converter(
        tmp_path,
        _recording_export({}),
    )
    checkpoint = HfFile(path="vae.safetensors", size=335 * 1024 * 1024)
    converter.hf_client.files = [checkpoint]

    async def read_header(repo_id: str, path: str):
        header = _single_file_header("vae_only")
        return header, len(json.dumps(header).encode("utf-8"))

    converter.hf_client.read_safetensors_header = read_header  # type: ignore[attr-defined]

    conversion_id = await converter.convert_from_hf(
        "owner/checkpoints",
        checkpoint_path=checkpoint.path,
    )
    await converter._process_next()

    job = converter.status(conversion_id)
    assert job is not None
    assert job.status is JobStatus.failed
    assert "pipeline completo" in (job.error or "")
    assert converter.hf_client.download_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "allocation_error",
    [
        MemoryError("cannot allocate checkpoint"),
        RuntimeError("DefaultCPUAllocator: not enough memory"),
    ],
)
async def test_single_file_allocation_failure_mentions_checkpoint_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allocation_error: Exception,
) -> None:
    converter, _installer, settings, _registry = make_converter(
        tmp_path,
        _recording_export({}),
    )
    checkpoint = HfFile(path="huge.safetensors", size=7_123_456_789)
    converter.hf_client.files = [checkpoint]

    async def read_header(repo_id: str, path: str):
        header = _single_file_header("pony_sdxl")
        return header, len(json.dumps(header).encode("utf-8"))

    converter.hf_client.read_safetensors_header = read_header  # type: ignore[attr-defined]

    def fail_materialize(*args, **kwargs) -> None:
        raise allocation_error

    monkeypatch.setattr(generation_converter_module, "materialize", fail_materialize)

    conversion_id = await converter.convert_from_hf(
        "owner/checkpoints",
        checkpoint_path=checkpoint.path,
    )
    await converter._process_next()

    job = converter.status(conversion_id)
    assert job is not None
    assert job.status is JobStatus.failed
    assert "RAM" in (job.error or "")
    assert str(checkpoint.size) in (job.error or "")
    leftovers = (
        list(settings.temp_path.iterdir())
        if settings.temp_path.exists()
        else []
    )
    assert leftovers == []


# ---------------------------------------------------------------------------
# Visibilidad: la conversion existe en el registro DESDE que se encola
#
# Bug de UX real (2026-07-31): las conversiones disparadas por los packs del
# instalador eran invisibles — el modelo aparecia en la UI recien al promover,
# ~40 min despues, y el usuario concluia que "no vino".
# ---------------------------------------------------------------------------


async def test_enqueuing_a_conversion_registers_a_converting_entry(tmp_path: Path) -> None:
    converter, _installer, _settings, registry = make_converter(
        tmp_path, export_fn=fake_export_ok
    )

    await converter.convert_from_hf("amd/sdxl-torch")

    entry = registry.get("gen--amd--sdxl-torch")
    assert entry is not None
    assert entry.status.value == "converting"
    assert entry.kind == ModelKind.diffusion_onnx


def test_a_failed_conversion_marks_the_entry_error(tmp_path: Path) -> None:
    def failing_export(src_dir, out_dir, on_component, dtype=None, atol=None):
        raise RuntimeError("export roto a proposito")

    converter, _installer, _settings, registry = make_converter(
        tmp_path, export_fn=failing_export
    )

    convert_and_drain(converter, "amd/sdxl-torch")

    entry = registry.get("gen--amd--sdxl-torch")
    assert entry is not None
    assert entry.status.value == "error"
    assert "export roto" in (entry.error or "")


def test_promotion_flips_the_converting_entry_to_installed(tmp_path: Path) -> None:
    converter, _installer, _settings, registry = make_converter(
        tmp_path, export_fn=fake_export_ok
    )

    convert_and_drain(converter, "amd/sdxl-torch")

    entry = registry.get("gen--amd--sdxl-torch")
    assert entry is not None
    assert entry.status.value == "installed"


async def test_a_reconversion_never_downgrades_an_installed_entry(tmp_path: Path) -> None:
    # Reconvertir un modelo ya instalado no debe sacarlo del dropdown mientras
    # corre, ni marcarlo en error si la reconversion falla: el instalado sigue
    # siendo usable hasta que una promocion NUEVA lo reemplace.
    converter, _installer, _settings, registry = make_converter(
        tmp_path, export_fn=fake_export_ok
    )
    registry.register(
        ModelEntry(
            id="gen--amd--sdxl-torch",
            name="amd/sdxl-torch",
            kind=ModelKind.diffusion_onnx,
            source="hf:amd/sdxl-torch",
            size_bytes=1,
            scale=None,
            file_path=None,
            status=ModelStatus.installed,
        )
    )

    await converter.convert_from_hf("amd/sdxl-torch")

    assert registry.get("gen--amd--sdxl-torch").status.value == "installed"


async def test_inpaint_merge_conversion_registers_the_merged_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.services.generation_inpaint_merge as merge_module

    converter, _installer, _settings, registry = make_converter(
        tmp_path, export_fn=fake_export_ok
    )
    merge_calls: dict[str, tuple[Path, Path, Path]] = {}

    def fake_merge(user_dir: Path, base_dir: Path, inpaint_dir: Path, out_dir: Path, progress_cb=None) -> Path:
        merge_calls["dirs"] = (user_dir, base_dir, inpaint_dir)
        # El merge real deja un dir diffusers en src_root; el export fake solo
        # necesita el model_index.
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "model_index.json").write_text(
            json.dumps({"_class_name": "StableDiffusionXLInpaintPipeline"}), encoding="utf-8"
        )
        return out_dir

    monkeypatch.setattr(merge_module, "merge_to_inpaint_dir", fake_merge)

    conversion_id = await converter.convert_inpaint_merge("john/epic-xl")
    # La entrada convirtiendo existe DESDE el encolado, con nombre e id propios:
    # el merge no puede pisar el modelo original.
    converting = registry.get("gen--john--epic-xl--inpainting")
    assert converting is not None and converting.status == ModelStatus.converting
    assert converting.name == "john/epic-xl (inpainting)"

    await converter._process_next()

    job = converter.status(conversion_id)
    assert job is not None and job.status == JobStatus.completed
    entry = registry.get("gen--john--epic-xl--inpainting")
    assert entry is not None and entry.status == ModelStatus.installed
    assert entry.name == "john/epic-xl (inpainting)"
    assert registry.get("gen--john--epic-xl") is None  # el original no se toca
    user_dir, base_dir, inpaint_dir = merge_calls["dirs"]
    assert user_dir.name == "user" and base_dir.name == "base" and inpaint_dir.name == "inpaint"


async def test_inpaint_merge_failure_marks_the_merged_entry_not_the_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.services.generation_inpaint_merge as merge_module

    converter, _installer, _settings, registry = make_converter(
        tmp_path, export_fn=fake_export_ok
    )

    def broken_merge(*args, **kwargs) -> Path:
        raise RuntimeError("merge roto")

    monkeypatch.setattr(merge_module, "merge_to_inpaint_dir", broken_merge)

    conversion_id = await converter.convert_inpaint_merge("john/epic-xl")
    await converter._process_next()

    job = converter.status(conversion_id)
    assert job is not None and job.status == JobStatus.failed
    entry = registry.get("gen--john--epic-xl--inpainting")
    assert entry is not None and entry.status == ModelStatus.error
