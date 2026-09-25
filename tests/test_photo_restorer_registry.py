from __future__ import annotations

from dataclasses import replace

import pytest

from app.config import Settings
from app.services import capabilities
from app.services.photo_restore_chain import RESTORE_CHAIN
from app.services.photo_restorer_registry import (
    DSP_CAPABILITY,
    STEP_CAPABILITY,
    capability_for_step,
    validate_step_ready,
)


class FakeResolver:
    def __init__(self, missing: dict[str, tuple[str, ...]] | None = None) -> None:
        self.missing = missing or {}
        self.asked: list[tuple[str, object]] = []

    def __call__(self, capability_id: str, settings: Settings, registry: object | None):
        self.asked.append((capability_id, registry))
        return capabilities.ResolvedCapability(
            id=capability_id,
            domain="image",
            label_key=f"capability.{capability_id}",
            status="needs_setup" if self.missing.get(capability_id) else "available",
            provisioning="vendored_pack",
            job_kind="image",
            strategies=("model",),
            missing_packs=self.missing.get(capability_id, ()),
        )


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


def test_every_chain_step_has_a_capability():
    assert set(STEP_CAPABILITY) == {spec.id for spec in RESTORE_CHAIN}


def test_dsp_steps_map_to_the_builtin_capability():
    assert STEP_CAPABILITY["descreen"] == DSP_CAPABILITY
    assert STEP_CAPABILITY["tone"] == DSP_CAPABILITY


def test_model_steps_map_to_the_capability_of_their_pack():
    assert STEP_CAPABILITY["repair"] == "image.restoreModels"
    assert STEP_CAPABILITY["deblock"] == "image.restoreModels"
    assert STEP_CAPABILITY["denoise"] == "image.restoreModels"
    assert STEP_CAPABILITY["faces"] == "image.restoreFaces"
    assert STEP_CAPABILITY["colorize"] == "image.colorize"


def test_repair_without_a_model_needs_only_the_builtin_capability():
    assert capability_for_step("repair", uses_model=False) == DSP_CAPABILITY


@pytest.mark.parametrize("step_id", ["deblock", "faces", "descreen"])
def test_steps_without_a_dsp_alternative_ignore_uses_model(step_id):
    assert capability_for_step(step_id, uses_model=False) == STEP_CAPABILITY[step_id]


def test_ready_step_passes_and_resolves_without_the_model_registry(settings):
    resolver = FakeResolver()

    validate_step_ready(settings, "denoise", resolve=resolver)

    assert resolver.asked == [("image.restoreModels", None)]


def test_step_with_a_missing_pack_names_the_pack(settings):
    resolver = FakeResolver({"image.restoreFaces": ("restore-faces",)})

    with pytest.raises(ValueError, match="los modelos de caras") as raised:
        validate_step_ready(settings, "faces", resolve=resolver)

    assert "download" not in str(raised.value).lower()
    assert ".ps1" not in str(raised.value)


def test_classic_repair_does_not_need_the_core_pack(settings):
    resolver = FakeResolver({"image.restoreModels": ("restore-core",)})

    validate_step_ready(settings, "repair", uses_model=False, resolve=resolver)

    assert resolver.asked == [(DSP_CAPABILITY, None)]


def test_unknown_step_is_refused(settings):
    with pytest.raises(ValueError, match="sharpen"):
        validate_step_ready(settings, "sharpen", resolve=FakeResolver())


@pytest.mark.xfail(strict=True, raises=KeyError, reason="P1-14 agrega image.restore* e image.colorize al CATALOG")
def test_every_step_capability_is_in_the_catalog(settings):
    for capability_id in set(STEP_CAPABILITY.values()):
        capabilities.resolve_one(capability_id, settings, None)


def test_step_unavailable_for_another_reason_is_refused(settings):
    def not_implemented(capability_id, settings, registry):
        resolved = FakeResolver()(capability_id, settings, registry)
        return replace(resolved, status="not_implemented")

    with pytest.raises(ValueError, match="colorize"):
        validate_step_ready(settings, "colorize", resolve=not_implemented)
