from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.config import Settings
from app.services.engines.sdcpp_models import SDCPP_MODEL_PREFIX
from app.services.missing_pack import PACK_LABELS
from app.services.vulkan_installer import VulkanInstallStatus, VulkanModelInstaller


def make_settings(tmp_path: Path, with_binary: bool = True, enabled: bool = True) -> Settings:
    binary = tmp_path / "sd.exe"
    if with_binary:
        binary.write_bytes(b"exe")
    return Settings(
        _env_file=None,
        RUNTIME_DIR=str(tmp_path / "runtime"),
        ENABLE_SDCPP=enabled,
        SDCPP_BINARY=str(binary),
        SDCPP_MODELS_DIR=str(tmp_path / "models"),
        SDCPP_MODEL=str(tmp_path / "no-existe.safetensors"),
    )


class FakeHfClient:
    def __init__(self, payload: bytes = b"checkpoint") -> None:
        self.payload = payload
        self.calls: list[tuple[str, str, Path]] = []

    async def download(self, repo_id, filename, dest, progress_cb=None, max_bytes=None):
        self.calls.append((repo_id, filename, dest))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.payload)
        if progress_cb is not None:
            progress_cb(len(self.payload), len(self.payload))


async def run_install(installer: VulkanModelInstaller, *args) -> str:
    job_id = await installer.install(*args)
    # El worker corre en su propia tarea: se espera al estado terminal.
    await installer.start()
    for _ in range(200):
        job = installer.status(job_id)
        if job.status in (VulkanInstallStatus.installed, VulkanInstallStatus.error):
            break
        await asyncio.sleep(0.01)
    await installer.stop()
    return job_id


@pytest.mark.asyncio
async def test_installing_a_checkpoint_is_just_a_download(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    client = FakeHfClient()
    installer = VulkanModelInstaller(settings, client)

    job_id = await run_install(installer, "Lykon/DreamShaper", "dreamshaper_8.safetensors")

    job = installer.status(job_id)
    assert job.status is VulkanInstallStatus.installed
    assert job.model_id == f"{SDCPP_MODEL_PREFIX}dreamshaper_8"
    assert (settings.sdcpp_models_dir_path / "dreamshaper_8.safetensors").is_file()
    # Sin conversión: una sola descarga y nada más.
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_the_installed_model_shows_up_as_usable(tmp_path: Path) -> None:
    from app.services.engines.sdcpp_models import list_sdcpp_models

    settings = make_settings(tmp_path)
    installer = VulkanModelInstaller(settings, FakeHfClient())

    await run_install(installer, "Lykon/DreamShaper", "dreamshaper_8.safetensors")

    assert [m.id for m in list_sdcpp_models(settings)] == [f"{SDCPP_MODEL_PREFIX}dreamshaper_8"]
    assert settings.sdcpp_available() is True


@pytest.mark.asyncio
async def test_a_non_checkpoint_file_is_rejected(tmp_path: Path) -> None:
    installer = VulkanModelInstaller(make_settings(tmp_path), FakeHfClient())

    with pytest.raises(ValueError, match="no es un checkpoint"):
        await installer.install("algun/repo", "README.md")


@pytest.mark.asyncio
async def test_a_path_in_the_filename_is_rejected(tmp_path: Path) -> None:
    # El nombre viene del repo remoto: sin esto escribiria fuera de la carpeta.
    installer = VulkanModelInstaller(make_settings(tmp_path), FakeHfClient())

    for evil in ("../fuera.safetensors", "sub/dir.safetensors", "..\\fuera.safetensors"):
        with pytest.raises(ValueError):
            await installer.install("algun/repo", evil)


@pytest.mark.asyncio
async def test_sin_el_motor_dice_que_paquete_falta_sin_dictar_comandos(tmp_path: Path) -> None:
    installer = VulkanModelInstaller(make_settings(tmp_path, with_binary=False), FakeHfClient())

    with pytest.raises(ValueError) as excinfo:
        await installer.install("algun/repo", "modelo.safetensors")

    mensaje = str(excinfo.value)
    assert PACK_LABELS["sdcpp"] in mensaje
    # La regresion a evitar: volver a mandar al usuario a la terminal.
    assert ".ps1" not in mensaje


@pytest.mark.asyncio
async def test_a_failed_download_reports_the_error_and_does_not_raise(tmp_path: Path) -> None:
    class BrokenClient(FakeHfClient):
        async def download(self, *args, **kwargs):
            raise RuntimeError("se corto la red")

    installer = VulkanModelInstaller(make_settings(tmp_path), BrokenClient())

    job_id = await run_install(installer, "algun/repo", "modelo.safetensors")

    job = installer.status(job_id)
    assert job.status is VulkanInstallStatus.error
    assert "se corto la red" in job.error


@pytest.mark.asyncio
async def test_an_empty_download_is_an_error_not_a_broken_model(tmp_path: Path) -> None:
    installer = VulkanModelInstaller(make_settings(tmp_path), FakeHfClient(payload=b""))

    job_id = await run_install(installer, "algun/repo", "modelo.safetensors")

    assert installer.status(job_id).status is VulkanInstallStatus.error
