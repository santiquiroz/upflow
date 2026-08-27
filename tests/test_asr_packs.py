from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.services.asr_installer import asr_model_id
from app.services.model_packs import (
    ASR_PACKS,
    RECOMMENDED_ASR_REPO,
    pending_asr_packs,
)
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry
from app.services.storage import StorageService


def make_registry(tmp_path: Path) -> ModelRegistry:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    StorageService(settings).ensure_directories()
    return ModelRegistry(settings)


def register_installed(registry: ModelRegistry, repo_id: str) -> None:
    registry.register(
        ModelEntry(
            id=asr_model_id(repo_id),
            name=repo_id,
            kind=ModelKind.asr_onnx,
            source=f"hf:{repo_id}",
            size_bytes=1,
            file_path="asr/x",
        )
    )


def test_el_recomendado_es_uno_de_los_packs_del_instalador():
    assert RECOMMENDED_ASR_REPO in ASR_PACKS.values()


def test_todos_los_packs_son_variantes_timestamped_multilingues():
    for repo in ASR_PACKS.values():
        # El karaoke necesita tiempos por palabra, y `.en` seria solo-ingles.
        assert repo.endswith("_timestamped")
        assert ".en" not in repo


def test_pendientes_ignora_claves_desconocidas(tmp_path: Path):
    registry = make_registry(tmp_path)
    assert pending_asr_packs(["rife", "whisper-small"], registry) == ("whisper-small",)


def test_pendientes_saltea_lo_ya_instalado(tmp_path: Path):
    registry = make_registry(tmp_path)
    register_installed(registry, ASR_PACKS["whisper-small"])

    assert pending_asr_packs(["whisper-small", "whisper-tiny"], registry) == (
        "whisper-tiny",
    )


@pytest.mark.asyncio
async def test_encolar_baja_solo_lo_pendiente(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from app.services import model_packs

    registry = make_registry(tmp_path)
    register_installed(registry, ASR_PACKS["whisper-tiny"])
    packs_file = tmp_path / "optional-packs.txt"
    packs_file.write_text("whisper-tiny\nwhisper-small\nrife\n", encoding="utf-8")
    monkeypatch.setattr(
        model_packs, "resolve_against_project_root", lambda _n: packs_file
    )

    class FakeInstaller:
        def __init__(self) -> None:
            self.repos: list[str] = []

        async def install_from_hf(self, repo_id: str) -> str:
            self.repos.append(repo_id)
            return f"install-{len(self.repos)}"

    installer = FakeInstaller()
    ids = await model_packs.enqueue_pending_asr_packs(registry, installer)

    assert installer.repos == [ASR_PACKS["whisper-small"]]
    assert ids == ["install-1"]
