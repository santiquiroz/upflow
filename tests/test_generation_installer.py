from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.services.device_semaphores import DeviceSemaphores
from app.services.generation_installer import (
    GenerationModelInstaller,
    _filter_to_declared,
    _generation_model_id,
    _patch_legacy_component_configs,
    _select_files,
)
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.hf_client import HfFile
from app.services.model_installer import InstallStatus
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry
from test_generation_engine import RecordingCoordinator
from test_model_installer import FakeHfClient as _BaseFakeHfClient

# ---------------------------------------------------------------------------
# SP1 Task 5 (generation module) - GenerationModelInstaller: installs a
# multi-file diffusers ONNX pipeline (model_index.json + component
# subdirectories) from a Hugging Face repo_id, parallel to the single-file
# ModelInstaller in model_installer.py. Two-phase download (model_index.json
# first, then only the components it declares) avoids pulling multi-GB
# torch checkpoints or vendor-specific binaries (e.g. MIGraphX `MXR/`)
# present alongside the ONNX pipeline in the `amd/` legacy repos.
#
# FakeHfClient (adapted from test_model_installer.FakeHfClient): the base
# fake's `download` doesn't accept `max_bytes` (added to the real HfClient in
# Task 3) and always writes the same `download_bytes` regardless of which
# file was requested. This subclass adds both: `max_bytes` as an accepted
# (ignored) kwarg, and `download_bytes_by_path` so a test can make
# `model_index.json` resolve to real JSON (needed for structural validation)
# while every other file gets an opaque placeholder.
# ---------------------------------------------------------------------------


class FakeHfClient(_BaseFakeHfClient):
    def __init__(self, files: list[HfFile], **kwargs: Any) -> None:
        super().__init__(files, **kwargs)
        self.download_bytes_by_path: dict[str, bytes] = {}

    async def download(self, repo_id, filename, dest, progress_cb=None, max_bytes=None):
        self.download_calls.append((repo_id, filename, dest))
        if self.download_error:
            raise self.download_error
        content = self.download_bytes_by_path.get(filename, b"onnx")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        if progress_cb is not None:
            progress_cb(len(content), len(content))
        return dest


MODEL_INDEX = json.dumps(
    {
        "_class_name": "OnnxStableDiffusionPipeline",
        "text_encoder": ["diffusers", "OnnxRuntimeModel"],
        "unet": ["diffusers", "OnnxRuntimeModel"],
        "vae_decoder": ["diffusers", "OnnxRuntimeModel"],
        "tokenizer": ["transformers", "CLIPTokenizer"],
        "scheduler": ["diffusers", "PNDMScheduler"],
    }
)

PIPELINE_FILES = [
    HfFile(path="model_index.json", size=len(MODEL_INDEX)),
    HfFile(path="text_encoder/model.onnx", size=10),
    HfFile(path="unet/model.onnx", size=10),
    HfFile(path="vae_decoder/model.onnx", size=10),
    HfFile(path="tokenizer/tokenizer_config.json", size=5),
    HfFile(path="scheduler/scheduler_config.json", size=5),
    HfFile(path="v1-5-pruned.ckpt", size=4_000_000_000),  # duplicado torch: debe saltearse
    HfFile(path="MXR/unet.mxr", size=5_000_000_000),  # binarios MIGraphX: carpeta NO declarada, debe saltearse
]


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), _env_file=None, **overrides)


def make_installer(
    tmp_path: Path,
    files: list[HfFile],
    *,
    gpu_coordinator: Any | None = None,
    device_semaphores: Any | None = None,
    **hf_kwargs: Any,
):
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    hf = FakeHfClient(files=files, **hf_kwargs)
    installer = GenerationModelInstaller(
        settings,
        registry,
        hf,
        gpu_coordinator if gpu_coordinator is not None else GpuSessionCoordinator(),
        device_semaphores if device_semaphores is not None else DeviceSemaphores(settings),
    )
    # el download fake debe escribir el model_index real para la validación estructural:
    hf.download_bytes_by_path = {"model_index.json": MODEL_INDEX.encode("utf-8")}
    return installer, registry, settings, hf


def install_and_drain(installer: GenerationModelInstaller, repo_id: str):
    async def _run():
        install_id = await installer.install_from_hf(repo_id)
        await installer._process_next()
        return installer.status(install_id)

    import asyncio

    return asyncio.run(_run())


def test_generation_model_id_is_prefixed_and_safe() -> None:
    assert _generation_model_id("amd/Stable-Diffusion-1.5") == "gen--amd--stable-diffusion-1.5"


def test_select_files_skips_torch_checkpoints() -> None:
    selected = _select_files(PIPELINE_FILES)
    assert all(not f.path.endswith((".ckpt", ".pth", ".safetensors", ".bin")) for f in selected)
    assert any(f.path == "unet/model.onnx" for f in selected)


def test_filter_to_declared_drops_undeclared_dirs_and_model_index() -> None:
    declared = ["text_encoder", "unet", "vae_decoder", "tokenizer", "scheduler"]
    kept = _filter_to_declared(_select_files(PIPELINE_FILES), declared)
    paths = [f.path for f in kept]
    assert "MXR/unet.mxr" not in paths          # carpeta no declarada
    assert "model_index.json" not in paths       # se baja aparte, en fase 1
    assert "unet/model.onnx" in paths
    assert "tokenizer/tokenizer_config.json" in paths


def test_install_happy_path_registers_diffusion_model(tmp_path: Path, monkeypatch) -> None:
    installer, registry, settings, hf = make_installer(tmp_path, files=PIPELINE_FILES)
    monkeypatch.setattr(installer, "_create_validation_pipeline", lambda pipeline_dir: FakeValidationPipeline())

    job = install_and_drain(installer, "amd/sd15")

    assert job.status == InstallStatus.installed
    entry = registry.get("gen--amd--sd15")
    assert entry is not None
    assert entry.kind == ModelKind.diffusion_onnx
    assert entry.scale is None
    assert entry.file_path == "generation/gen--amd--sd15"
    final_dir = settings.models_path / "generation" / "gen--amd--sd15"
    assert (final_dir / "model_index.json").is_file()
    # patch legacy: _class_name == OnnxStableDiffusionPipeline y los componentes
    # no traian config.json -> el installer los completa desde los vendorizados
    assert (final_dir / "unet" / "config.json").is_file()
    assert (final_dir / "text_encoder" / "config.json").is_file()
    # MXR/ no declarado: nunca se descargo
    assert not any("MXR" in call for call in map(str, hf.download_calls))


def test_install_validation_acquires_gpu_coordinator(tmp_path: Path, monkeypatch) -> None:
    # Item 1 (final whole-branch review): la validacion del installer es la
    # UNICA sesion DML del codebase invisible al GpuSessionCoordinator -- debe
    # anunciarse igual que los 6 engines (ver GenerationEngine._get_pipeline).
    coordinator = RecordingCoordinator()
    installer, registry, settings, hf = make_installer(
        tmp_path, files=PIPELINE_FILES, gpu_coordinator=coordinator
    )
    monkeypatch.setattr(installer, "_create_validation_pipeline", lambda pipeline_dir: FakeValidationPipeline())

    job = install_and_drain(installer, "amd/sd15")

    assert job.status == InstallStatus.installed
    assert len(coordinator.acquired) == 1
    device, owner = coordinator.acquired[0]
    assert device == settings.default_device
    assert hasattr(owner, "release_device")
    owner.release_device(device)  # no-op, protocol requires the method


def test_install_rejects_repo_without_model_index(tmp_path: Path) -> None:
    files = [HfFile(path="model.onnx", size=10)]
    installer, registry, _settings, hf = make_installer(tmp_path, files=files)

    job = install_and_drain(installer, "someone/upscaler")

    assert job.status == InstallStatus.error
    assert "model_index.json" in (job.error or "")
    assert hf.download_calls == []  # error ANTES de bajar gigas
    assert registry.get("gen--someone--upscaler") is None


def test_install_rejects_when_total_size_exceeds_cap(tmp_path: Path) -> None:
    big = [
        HfFile(path="model_index.json", size=len(MODEL_INDEX)),
        HfFile(path="unet/model.onnx", size=9 * 1024 * 1024 * 1024),
    ]
    installer, _registry, _settings, hf = make_installer(tmp_path, files=big)

    job = install_and_drain(installer, "amd/sdxl-huge")

    assert job.status == InstallStatus.error
    # el cap se chequea DESPUES de bajar model_index.json (fase 1, KBs) pero
    # ANTES de bajar cualquier peso:
    assert [str(c) for c in hf.download_calls if "model_index" not in str(c)] == []


def test_install_cuda_only_model_fails_with_friendly_message(tmp_path: Path, monkeypatch) -> None:
    installer, registry, _settings, _hf = make_installer(tmp_path, files=PIPELINE_FILES)

    def explode(pipeline_dir: Path) -> Any:
        raise RuntimeError("CUDAExecutionProvider is not in available providers")

    monkeypatch.setattr(installer, "_create_validation_pipeline", explode)

    job = install_and_drain(installer, "tlwu/sdxl-cuda-only")

    assert job.status == InstallStatus.error
    assert "requiere GPU NVIDIA" in (job.error or "")
    assert registry.get("gen--tlwu--sdxl-cuda-only") is None


class FakeValidationPipeline:
    def __call__(self, **kwargs: Any) -> Any:
        class _R:
            images = [object()]

        return _R()


# ---------------------------------------------------------------------------
# Review round 1 fixes:
#   1. _promote_staging_dir must roll back to the previous install if the
#      staging->final replace never succeeds (Windows file-lock PermissionError).
#   2. Repo-controlled paths (declared component names, file paths) must not
#      be able to escape staging_root when resolved on disk.
#   3. _patch_legacy_component_configs must never overwrite an existing
#      component config.json, and must no-op for a non-legacy pipeline class.
# ---------------------------------------------------------------------------


def _patch_staging_dir_replace(monkeypatch: pytest.MonkeyPatch, *, fail_times: int | None) -> list[int]:
    """Mirrors test_model_installer._patch_staging_replace: patches Path.replace
    so only the staging directory -> final directory move is affected, leaving
    ModelRegistry's own tmp-then-replace persistence (and the final_dir ->
    backup_dir move) unaffected.
    """
    original_replace = Path.replace
    call_count = [0]

    def fake_replace(self: Path, target: Path) -> Path:
        if self.name.startswith("gen-staging-"):
            call_count[0] += 1
            if fail_times is None or call_count[0] <= fail_times:
                raise PermissionError("[WinError 5] Access is denied")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fake_replace)
    return call_count


def test_promote_rolls_back_previous_install_when_replace_always_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installer, registry, settings, hf = make_installer(tmp_path, files=PIPELINE_FILES)
    monkeypatch.setattr(installer, "_create_validation_pipeline", lambda pipeline_dir: FakeValidationPipeline())

    final_dir = settings.models_path / "generation" / "gen--amd--sd15"
    final_dir.mkdir(parents=True)
    (final_dir / "marker.txt").write_text("previous-install-marker", encoding="utf-8")
    registry.register(
        ModelEntry(
            id="gen--amd--sd15",
            name="amd/sd15",
            kind=ModelKind.diffusion_onnx,
            source="kept-old-marker",
            size_bytes=1,
            scale=None,
            file_path="generation/gen--amd--sd15",
        )
    )

    _patch_staging_dir_replace(monkeypatch, fail_times=None)

    job = install_and_drain(installer, "amd/sd15")

    assert job.status == InstallStatus.error
    # rollback: la instalacion previa sigue intacta, ni se perdio ni quedo a medias
    assert final_dir.is_dir()
    assert (final_dir / "marker.txt").read_text(encoding="utf-8") == "previous-install-marker"
    entry = registry.get("gen--amd--sd15")
    assert entry is not None
    assert entry.source == "kept-old-marker"


MODEL_INDEX_TRAVERSAL = json.dumps(
    {
        "_class_name": "OnnxStableDiffusionPipeline",
        "..": ["diffusers", "OnnxRuntimeModel"],
    }
)


def test_install_rejects_files_that_escape_staging_dir(tmp_path: Path) -> None:
    files = [
        HfFile(path="model_index.json", size=len(MODEL_INDEX_TRAVERSAL)),
        HfFile(path="../evil.onnx", size=10),
    ]
    installer, registry, settings, hf = make_installer(tmp_path, files=files)
    hf.download_bytes_by_path = {"model_index.json": MODEL_INDEX_TRAVERSAL.encode("utf-8")}

    job = install_and_drain(installer, "amd/evil-repo")

    assert job.status == InstallStatus.error
    assert "escapa" in (job.error or "")
    assert not (settings.temp_path / "evil.onnx").exists()
    assert registry.get("gen--amd--evil-repo") is None


def test_patch_legacy_component_configs_skips_component_that_escapes_staging_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    escaping_dir = tmp_path / "outside-component"
    escaping_dir.mkdir()

    # El lookup de "vendorizado" real (LEGACY_CONFIGS_ASSETS_DIR) solo tiene
    # nombres legitimos (unet, text_encoder, ...), asi que un componente ".."
    # nunca resuelve a un vendored real -- eso por si solo ya evitaria el
    # copyfile, sin probar nada del guard nuevo. Para probar el guard de
    # verdad, se apunta LEGACY_CONFIGS_ASSETS_DIR a un directorio fake donde
    # SI existe un "vendorizado" para ese nombre exacto: sin el guard, esto
    # copiaria fuera de staging_root.
    fake_assets_dir = tmp_path / "somewhere" / "fake-assets"
    fake_assets_dir.mkdir(parents=True)
    monkeypatch.setattr(
        "app.services.generation_installer.LEGACY_CONFIGS_ASSETS_DIR", fake_assets_dir
    )
    vendored_escape_target = tmp_path / "somewhere" / "outside-component" / "config.json"
    vendored_escape_target.parent.mkdir(parents=True)
    vendored_escape_target.write_text('{"vendored": true}', encoding="utf-8")

    index = json.dumps(
        {
            "_class_name": "OnnxStableDiffusionPipeline",
            "../outside-component": ["diffusers", "OnnxRuntimeModel"],
        }
    )
    (staging_root / "model_index.json").write_text(index, encoding="utf-8")

    _patch_legacy_component_configs(staging_root)

    assert not (escaping_dir / "config.json").exists()


DISTINCTIVE_UNET_CONFIG = b'{"_class_name": "UNet2DConditionModel", "distinctive_marker": "keep-me"}'

PIPELINE_FILES_WITH_EXISTING_UNET_CONFIG = [
    HfFile(path="model_index.json", size=len(MODEL_INDEX)),
    HfFile(path="text_encoder/model.onnx", size=10),
    HfFile(path="unet/model.onnx", size=10),
    HfFile(path="unet/config.json", size=len(DISTINCTIVE_UNET_CONFIG)),
    HfFile(path="vae_decoder/model.onnx", size=10),
    HfFile(path="tokenizer/tokenizer_config.json", size=5),
    HfFile(path="scheduler/scheduler_config.json", size=5),
]


def test_install_does_not_overwrite_existing_component_config(tmp_path: Path, monkeypatch) -> None:
    installer, registry, settings, hf = make_installer(
        tmp_path, files=PIPELINE_FILES_WITH_EXISTING_UNET_CONFIG
    )
    hf.download_bytes_by_path["unet/config.json"] = DISTINCTIVE_UNET_CONFIG
    monkeypatch.setattr(installer, "_create_validation_pipeline", lambda pipeline_dir: FakeValidationPipeline())

    job = install_and_drain(installer, "amd/sd15")

    assert job.status == InstallStatus.installed
    final_dir = settings.models_path / "generation" / "gen--amd--sd15"
    assert (final_dir / "unet" / "config.json").read_bytes() == DISTINCTIVE_UNET_CONFIG


def test_patch_legacy_component_configs_skips_non_legacy_pipeline_class(tmp_path: Path) -> None:
    staging_root = tmp_path / "staging"
    staging_root.mkdir()
    unet_dir = staging_root / "unet"
    unet_dir.mkdir()
    index = json.dumps(
        {
            "_class_name": "StableDiffusionPipeline",
            "unet": ["diffusers", "UNet2DConditionModel"],
        }
    )
    (staging_root / "model_index.json").write_text(index, encoding="utf-8")

    _patch_legacy_component_configs(staging_root)

    assert not (unet_dir / "config.json").exists()
