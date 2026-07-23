from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.services.generation_installer import (
    GenerationModelInstaller,
    _filter_to_declared,
    _generation_model_id,
    _select_files,
)
from app.services.hf_client import HfFile
from app.services.model_installer import InstallStatus
from app.services.model_registry import ModelKind, ModelRegistry
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


def make_installer(tmp_path: Path, files: list[HfFile], **hf_kwargs: Any):
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    hf = FakeHfClient(files=files, **hf_kwargs)
    installer = GenerationModelInstaller(settings, registry, hf)
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
