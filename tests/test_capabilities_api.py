from __future__ import annotations

from pathlib import Path

import pytest

from app.api.routes import capability_tree
from app.config import Settings
from app.services.capabilities import CATALOG, DOMAIN_ORDER
from app.services.model_registry import ModelEntry, ModelKind, ModelStatus


class FakeRegistry:
    def __init__(self, entries: list[ModelEntry] | None = None) -> None:
        self._entries = entries or []

    def list(self) -> list[ModelEntry]:
        return list(self._entries)


def entry(kind: ModelKind) -> ModelEntry:
    return ModelEntry(
        id=f"m-{kind.value}",
        name="m",
        kind=kind,
        source="test",
        size_bytes=1,
        status=ModelStatus.installed,
    )


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), _env_file=None, **overrides)


def flat(response) -> dict:
    return {
        item.id: item
        for group in response.domains
        for item in (*group.capabilities, *group.roadmap)
    }


@pytest.mark.asyncio
async def test_the_tree_exposes_the_four_domains_in_order(tmp_path: Path):
    response = await capability_tree(make_settings(tmp_path), FakeRegistry())
    assert [group.domain for group in response.domains] == list(DOMAIN_ORDER)


@pytest.mark.asyncio
async def test_the_tree_exposes_every_capability_of_the_catalog(tmp_path: Path):
    response = await capability_tree(make_settings(tmp_path), FakeRegistry())
    assert sorted(flat(response)) == sorted(item.id for item in CATALOG)


@pytest.mark.asyncio
async def test_the_roadmap_is_separate_from_the_live_capabilities(tmp_path: Path):
    # Van separadas para que el frontend les pueda dar el encabezado de mapa de
    # ruta sin filtrar por su cuenta.
    response = await capability_tree(make_settings(tmp_path), FakeRegistry())
    for group in response.domains:
        for item in group.capabilities:
            assert item.status != "not_implemented"
        for item in group.roadmap:
            assert item.status == "not_implemented"
            assert item.unavailable_reason_key


@pytest.mark.asyncio
async def test_the_status_follows_the_disk(tmp_path: Path):
    binary = tmp_path / "rife.exe"
    settings = make_settings(tmp_path, RIFE_BINARY=str(binary))

    before = flat(await capability_tree(settings, FakeRegistry()))["video.interpolate"]
    binary.write_bytes(b"x")
    after = flat(await capability_tree(settings, FakeRegistry()))["video.interpolate"]

    assert before.status == "needs_setup"
    assert before.missing_packs == ["rife"]
    assert after.status == "available"
    assert after.missing_packs == []


@pytest.mark.asyncio
async def test_the_registry_decides_the_generation_capability(tmp_path: Path):
    settings = make_settings(tmp_path)

    empty = flat(await capability_tree(settings, FakeRegistry()))["generate.textToImage"]
    filled = flat(
        await capability_tree(settings, FakeRegistry([entry(ModelKind.diffusion_onnx)]))
    )["generate.textToImage"]

    assert empty.status == "needs_setup"
    assert filled.status == "available"


@pytest.mark.asyncio
async def test_the_response_serializes_camel_case_aliases(tmp_path: Path):
    response = await capability_tree(make_settings(tmp_path), FakeRegistry())
    dumped = response.model_dump(by_alias=True)

    group = dumped["domains"][0]
    assert "labelKey" in group
    item = group["roadmap"][0] if group["roadmap"] else group["capabilities"][0]
    assert "labelKey" in item
    assert "jobKind" in item
    assert "missingPacks" in item
    assert "unavailableReasonKey" in item
    assert "setupReasonKey" in item


@pytest.mark.asyncio
async def test_the_copy_travels_as_translation_keys(tmp_path: Path):
    response = await capability_tree(make_settings(tmp_path), FakeRegistry())
    for group in response.domains:
        assert group.label_key.startswith("capability.domain.")
        for item in (*group.capabilities, *group.roadmap):
            assert item.label_key.startswith("capability.")
            for key in (item.unavailable_reason_key, item.setup_reason_key):
                if key is not None:
                    assert key.startswith("capability.")


@pytest.mark.asyncio
async def test_every_capability_reports_its_strategies(tmp_path: Path):
    response = await capability_tree(make_settings(tmp_path), FakeRegistry())
    for item in flat(response).values():
        assert item.strategies
        assert set(item.strategies) <= {"dsp", "model"}
