from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.services.asr_installer import (
    ASR_MODELS_SUBDIR,
    AsrInstallStatus,
    AsrModelInstaller,
    asr_model_id,
)
from app.services.hf_client import HfFile
from app.services.model_registry import ModelKind, ModelRegistry
from app.services.storage import StorageService

MB = 1024 * 1024

COMPLETE_REPO = (
    HfFile(path="config.json", size=1024),
    HfFile(path="generation_config.json", size=1024),
    HfFile(path="preprocessor_config.json", size=1024),
    HfFile(path="tokenizer.json", size=2048),
    HfFile(path="vocab.json", size=2048),
    HfFile(path="merges.txt", size=512),
    HfFile(path="README.md", size=4096),
    HfFile(path="onnx/encoder_model.onnx", size=32 * MB),
    HfFile(path="onnx/decoder_model.onnx", size=100 * MB),
    HfFile(path="onnx/decoder_with_past_model.onnx", size=90 * MB),
    HfFile(path="onnx/decoder_model_merged.onnx", size=180 * MB),
    HfFile(path="onnx/encoder_model_fp16.onnx", size=16 * MB),
)


class FakeHf:
    def __init__(self, files: tuple[HfFile, ...], fail_on: str | None = None) -> None:
        self.files = files
        self.fail_on = fail_on
        self.downloaded: list[str] = []

    async def repo_files(self, _repo_id: str) -> list[HfFile]:
        return list(self.files)

    async def download(self, _repo_id: str, filename: str, dest: Path, **_kwargs) -> Path:
        if self.fail_on is not None and filename == self.fail_on:
            raise RuntimeError(f"fallo bajando {filename}")
        self.downloaded.append(filename)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return dest


def make_settings(tmp_path: Path) -> Settings:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    return settings


def make_installer(
    tmp_path: Path, files: tuple[HfFile, ...] = COMPLETE_REPO, fail_on: str | None = None
) -> tuple[AsrModelInstaller, FakeHf, Settings, ModelRegistry]:
    settings = make_settings(tmp_path)
    registry = ModelRegistry(settings)
    hf = FakeHf(files, fail_on)
    return AsrModelInstaller(settings, registry, hf), hf, settings, registry


# ---------------------------------------------------------------------------
# Camino feliz
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_installing_downloads_only_what_the_model_needs(tmp_path: Path):
    installer, hf, _s, _r = make_installer(tmp_path)
    install_id = await installer.install_from_hf("onnx-community/whisper-tiny.en")
    await installer._process_next()

    assert installer.status(install_id).status is AsrInstallStatus.installed
    # El merged y las variantes cuantizadas NO se bajan: peso muerto.
    assert not any("merged" in name for name in hf.downloaded)
    assert not any("fp16" in name for name in hf.downloaded)
    assert "README.md" not in hf.downloaded
    # Y si se baja el par no fusionado completo.
    for required in (
        "onnx/encoder_model.onnx",
        "onnx/decoder_model.onnx",
        "onnx/decoder_with_past_model.onnx",
    ):
        assert required in hf.downloaded


@pytest.mark.asyncio
async def test_the_model_lands_in_the_registry_with_its_own_kind(tmp_path: Path):
    installer, _hf, _s, registry = make_installer(tmp_path)
    await installer.install_from_hf("onnx-community/whisper-tiny.en")
    await installer._process_next()

    entry = registry.get(asr_model_id("onnx-community/whisper-tiny.en"))
    assert entry is not None
    assert entry.kind is ModelKind.asr_onnx
    assert entry.source == "hf:onnx-community/whisper-tiny.en"


@pytest.mark.asyncio
async def test_the_files_end_up_in_the_asr_folder(tmp_path: Path):
    installer, _hf, settings, _r = make_installer(tmp_path)
    await installer.install_from_hf("onnx-community/whisper-tiny.en")
    await installer._process_next()

    model_dir = (
        settings.models_path
        / ASR_MODELS_SUBDIR
        / asr_model_id("onnx-community/whisper-tiny.en")
    )
    assert (model_dir / "onnx" / "encoder_model.onnx").exists()
    assert (model_dir / "config.json").exists()


@pytest.mark.asyncio
async def test_progress_reaches_a_hundred(tmp_path: Path):
    installer, _hf, _s, _r = make_installer(tmp_path)
    install_id = await installer.install_from_hf("owner/model")
    await installer._process_next()

    assert installer.status(install_id).progress_pct == 100.0


@pytest.mark.asyncio
async def test_staging_is_cleaned_up(tmp_path: Path):
    installer, _hf, settings, _r = make_installer(tmp_path)
    await installer.install_from_hf("owner/model")
    await installer._process_next()

    assert list(settings.temp_path.glob("asr-staging-*")) == []


# ---------------------------------------------------------------------------
# Repos que no sirven
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_repo_without_the_encoder_fails_before_downloading(tmp_path: Path):
    # Descubrirlo al final dejaria cientos de megas en disco para despues fallar al
    # cargar el modelo.
    files = tuple(f for f in COMPLETE_REPO if f.path != "onnx/encoder_model.onnx")
    installer, hf, _s, _r = make_installer(tmp_path, files)
    install_id = await installer.install_from_hf("owner/incompleto")
    await installer._process_next()

    job = installer.status(install_id)
    assert job.status is AsrInstallStatus.error
    assert "encoder_model" in job.error
    assert hf.downloaded == []


@pytest.mark.asyncio
async def test_a_repo_with_only_the_merged_decoder_is_rejected(tmp_path: Path):
    # Existen repos asi. No sirven para este motor, que carga con use_merged=False.
    files = (
        HfFile(path="config.json", size=1024),
        HfFile(path="onnx/encoder_model.onnx", size=32 * MB),
        HfFile(path="onnx/decoder_model_merged.onnx", size=180 * MB),
    )
    installer, hf, _s, _r = make_installer(tmp_path, files)
    install_id = await installer.install_from_hf("owner/solo-merged")
    await installer._process_next()

    assert installer.status(install_id).status is AsrInstallStatus.error
    assert hf.downloaded == []


@pytest.mark.asyncio
async def test_nothing_is_registered_when_the_install_fails(tmp_path: Path):
    installer, _hf, _s, registry = make_installer(
        tmp_path, fail_on="onnx/decoder_model.onnx"
    )
    install_id = await installer.install_from_hf("owner/model")
    await installer._process_next()

    assert installer.status(install_id).status is AsrInstallStatus.error
    assert registry.get(asr_model_id("owner/model")) is None


@pytest.mark.asyncio
async def test_a_failed_download_leaves_no_staging_behind(tmp_path: Path):
    installer, _hf, settings, _r = make_installer(
        tmp_path, fail_on="onnx/decoder_model.onnx"
    )
    await installer.install_from_hf("owner/model")
    await installer._process_next()

    assert list(settings.temp_path.glob("asr-staging-*")) == []


@pytest.mark.asyncio
async def test_a_path_that_escapes_is_filtered_before_being_downloaded(tmp_path: Path):
    """Primera capa: la seleccion nunca lo pide.

    El listado viene de la red, asi que un `../` tiene dos barreras. Esta es la
    primera: is_required_asr_file solo acepta metadata de la RAIZ, y `../x.json`
    tiene separador, asi que queda afuera. La instalacion sigue bien porque el resto
    del repo esta completo.
    """
    files = (
        HfFile(path="config.json", size=1),
        HfFile(path="onnx/encoder_model.onnx", size=1),
        HfFile(path="onnx/decoder_model.onnx", size=1),
        HfFile(path="onnx/decoder_with_past_model.onnx", size=1),
        HfFile(path="../escapado.json", size=1),
    )
    installer, hf, _s, _r = make_installer(tmp_path, files)
    await installer.install_from_hf("owner/malicioso")
    await installer._process_next()

    assert "../escapado.json" not in hf.downloaded


def test_the_staging_guard_rejects_an_escaping_path(tmp_path: Path):
    """Segunda capa: el guard de destino, por si el filtro cambiara.

    Se prueba directo porque la primera capa impide llegar hasta aca, y un backstop
    que nadie ejercita es un backstop que no se sabe si funciona.
    """
    from app.services.asr_installer import _safe_staging_dest

    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(ValueError, match="escapa"):
        _safe_staging_dest(staging, "../afuera.json")
    with pytest.raises(ValueError, match="escapa"):
        _safe_staging_dest(staging, "onnx/../../afuera.onnx")

    # Y una ruta normal sigue resolviendo dentro.
    assert _safe_staging_dest(staging, "onnx/encoder_model.onnx").is_relative_to(
        staging.resolve()
    )


@pytest.mark.asyncio
async def test_reinstalling_replaces_the_previous_copy(tmp_path: Path):
    installer, _hf, settings, registry = make_installer(tmp_path)
    for _ in range(2):
        await installer.install_from_hf("owner/model")
        await installer._process_next()

    model_dir = settings.models_path / ASR_MODELS_SUBDIR / asr_model_id("owner/model")
    assert model_dir.exists()
    assert registry.get(asr_model_id("owner/model")) is not None


def test_the_model_id_is_derived_from_the_repo():
    assert asr_model_id("onnx-community/whisper-tiny.en") == (
        "asr--onnx-community--whisper-tiny.en"
    )


@pytest.mark.asyncio
async def test_status_of_an_unknown_install_is_none(tmp_path: Path):
    installer, _hf, _s, _r = make_installer(tmp_path)
    assert installer.status("nope") is None


def test_the_installer_and_engine_are_wired_into_the_app_state():
    """Sin esto la ruta tiraria AttributeError en la primera llamada real, que es
    exactamente como se rompio /audio/voice-catalog."""
    from fastapi.testclient import TestClient

    from app.main import app
    from app.services.engines.transcribe_onnx import TranscribeEngine

    with TestClient(app):
        assert isinstance(app.state.asr_installer, AsrModelInstaller)
        assert isinstance(app.state.transcribe_engine, TranscribeEngine)
