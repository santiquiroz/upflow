from __future__ import annotations

from pathlib import Path

import pytest

from app.services import model_packs
from app.services.generation_installer import _generation_model_id
from app.services.model_packs import (
    enqueue_pending_model_packs,
    pending_model_packs,
    read_selected_packs,
)
from app.services.model_registry import ModelEntry, ModelKind, ModelStatus


class FakeRegistry:
    def __init__(self, entries: list[ModelEntry] | None = None) -> None:
        self._entries = entries or []

    def list(self) -> list[ModelEntry]:
        return list(self._entries)


class FakeConverter:
    def __init__(self, failing_repos: set[str] | None = None) -> None:
        self.failing_repos = failing_repos or set()
        self.calls: list[tuple[str, str]] = []

    async def convert_from_hf(self, repo_id: str, *, precision: str) -> str:
        self.calls.append((repo_id, precision))
        if repo_id in self.failing_repos:
            raise RuntimeError(f"cannot convert {repo_id}")
        return f"conversion-{len(self.calls)}"


def installed_model(repo_id: str, status: ModelStatus = ModelStatus.installed) -> ModelEntry:
    return ModelEntry(
        id=_generation_model_id(repo_id, None),
        name=repo_id,
        kind=ModelKind.diffusion_onnx,
        source=f"hf:{repo_id}",
        size_bytes=1,
        status=status,
    )


def select_file(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(
        model_packs,
        "resolve_against_project_root",
        lambda value: path,
    )


def test_read_selected_packs_filters_binary_and_unknown_keys(tmp_path: Path) -> None:
    packs_file = tmp_path / "optional-packs.txt"
    packs_file.write_text(
        "rife\n\nmodel-anime\nunknown-pack\nmodel-photo\n",
        encoding="utf-8",
    )

    assert read_selected_packs(packs_file) == ("model-anime", "model-photo")


def test_read_selected_packs_returns_empty_for_missing_file(tmp_path: Path) -> None:
    assert read_selected_packs(tmp_path / "missing.txt") == ()


def test_read_selected_packs_ignores_blank_lines(tmp_path: Path) -> None:
    packs_file = tmp_path / "optional-packs.txt"
    packs_file.write_text("\n \nmodel-photo\n\t\n", encoding="utf-8")

    assert read_selected_packs(packs_file) == ("model-photo",)


def test_pending_model_packs_excludes_installed_model() -> None:
    anime_repo = "John6666/hassaku-xl-illustrious-v31-sdxl"
    registry = FakeRegistry([installed_model(anime_repo)])

    assert pending_model_packs(("model-anime", "model-photo"), registry) == (
        "model-photo",
    )


def test_pending_model_packs_includes_model_not_in_installed_state() -> None:
    photo_repo = "John6666/epicrealism-xl-vxvi-lastfame-realism-sdxl"
    registry = FakeRegistry([installed_model(photo_repo, ModelStatus.error)])

    assert pending_model_packs(("model-photo",), registry) == ("model-photo",)


def test_pending_model_packs_returns_all_selected_for_empty_registry() -> None:
    assert pending_model_packs(
        ("model-photo", "model-anime"),
        FakeRegistry(),
    ) == ("model-photo", "model-anime")


@pytest.mark.asyncio
async def test_enqueue_pending_model_packs_uses_repo_and_precision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packs_file = tmp_path / "optional-packs.txt"
    packs_file.write_text("model-anime\nmodel-photo\n", encoding="utf-8")
    select_file(monkeypatch, packs_file)
    converter = FakeConverter()

    conversion_ids = await enqueue_pending_model_packs(
        FakeRegistry(),
        converter,
    )

    assert converter.calls == [
        ("John6666/hassaku-xl-illustrious-v31-sdxl", "fp16"),
        ("John6666/epicrealism-xl-vxvi-lastfame-realism-sdxl", "fp16"),
    ]
    assert conversion_ids == ["conversion-1", "conversion-2"]


@pytest.mark.asyncio
async def test_enqueue_pending_model_packs_continues_after_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packs_file = tmp_path / "optional-packs.txt"
    packs_file.write_text("model-anime\nmodel-photo\n", encoding="utf-8")
    select_file(monkeypatch, packs_file)
    converter = FakeConverter(
        failing_repos={"John6666/hassaku-xl-illustrious-v31-sdxl"}
    )

    conversion_ids = await enqueue_pending_model_packs(
        FakeRegistry(),
        converter,
    )

    assert converter.calls == [
        ("John6666/hassaku-xl-illustrious-v31-sdxl", "fp16"),
        ("John6666/epicrealism-xl-vxvi-lastfame-realism-sdxl", "fp16"),
    ]
    assert conversion_ids == ["conversion-2"]


@pytest.mark.asyncio
async def test_enqueue_pending_model_packs_is_noop_without_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    select_file(monkeypatch, tmp_path / "missing.txt")
    converter = FakeConverter()

    conversion_ids = await enqueue_pending_model_packs(
        FakeRegistry(),
        converter,
    )

    assert converter.calls == []
    assert conversion_ids == []
