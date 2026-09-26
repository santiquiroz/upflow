from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import restore_routes
from app.api.auth_deps import get_user_store
from app.config import Settings, get_settings
from app.headless_base import UsageError
from app.headless_restore import require_restore_enabled
from app.models import CctvOptions
from app.services.capabilities import (
    CATALOG,
    PENDING_MODEL_RELEASE_REASON,
    HostProbes,
    group_by_domain,
    resolve_capabilities,
    resolve_one,
)
from app.services.cctv_capability_view import cctv_capability_view
from app.services.cctv_chain import CctvChainError
from app.services.cctv_enhance_runner import enhance_runner_entry
from app.services.cctv_job_validation import AI_LANE_DISABLED, check_lane_released
from app.services.ffmpeg_capabilities import FfmpegProbeError
from app.services.release_gates import (
    CCTV_AI_DISABLED_MESSAGE,
    RESTORE_DISABLED_MESSAGE,
    FeatureDisabledError,
    ensure_cctv_ai_enabled,
    ensure_restore_enabled,
)
from test_job_manager_restore import create_restore_job, make_manager
from test_video_job_manager_cctv import TOKEN, rejected, write_fake_session
from test_video_job_manager_cctv import make_manager as make_cctv_manager

RESTORE_CAPABILITIES = ("image.restore", "image.restoreModels", "image.restoreFaces", "image.colorize")
CCTV_AI_CAPABILITY = "video.cctvAi"
GATED_CAPABILITIES = frozenset({*RESTORE_CAPABILITIES, CCTV_AI_CAPABILITY})
EVERYTHING_PRESENT = HostProbes(ffmpeg_cctv_build=lambda _s: True, dml_gpu=lambda _s: True)


class EmptyRegistry:
    def list(self) -> list:
        return []


def release_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides: object) -> Settings:
    monkeypatch.delenv("RESTORE_PHOTO_ENABLED", raising=False)
    monkeypatch.delenv("CCTV_AI_ENABLED", raising=False)
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"), **overrides)


def no_ffmpeg_build(_binary: Path):
    raise FfmpegProbeError("no ffmpeg in this test")


def test_the_release_ships_photo_restore_and_the_cctv_ai_lane_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = release_settings(tmp_path, monkeypatch)

    assert settings.restore_photo_enabled is False
    assert settings.cctv_ai_enabled is False


def test_only_the_features_waiting_for_model_packs_are_gated() -> None:
    gated = {capability.id for capability in CATALOG if capability.release_flag is not None}

    assert gated == GATED_CAPABILITIES


@pytest.mark.parametrize("capability_id", sorted(GATED_CAPABILITIES))
def test_a_gated_capability_goes_to_the_roadmap_without_offering_a_download(
    capability_id: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = release_settings(tmp_path, monkeypatch)

    resolved = resolve_one(capability_id, settings, EmptyRegistry(), EVERYTHING_PRESENT)

    assert resolved.status == "not_implemented"
    assert resolved.missing_packs == ()
    assert resolved.activatable_settings == ()
    assert resolved.unavailable_reason_key == PENDING_MODEL_RELEASE_REASON


def test_the_gated_capabilities_are_listed_in_the_roadmap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = release_settings(tmp_path, monkeypatch)

    domains = group_by_domain(resolve_capabilities(settings, EmptyRegistry(), EVERYTHING_PRESENT))

    roadmap = {item.id for domain in domains for item in domain.roadmap}
    live = {item.id for domain in domains for item in domain.capabilities}
    assert GATED_CAPABILITIES <= roadmap
    assert not GATED_CAPABILITIES & live
    assert "video.cctv" in live


def test_turning_the_flag_on_brings_the_builtin_restore_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = release_settings(tmp_path, monkeypatch, RESTORE_PHOTO_ENABLED=True)

    assert resolve_one("image.restore", settings, EmptyRegistry(), EVERYTHING_PRESENT).status == "available"


def test_the_cctv_view_explains_why_the_ai_lane_is_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = release_settings(tmp_path, monkeypatch)

    view = cctv_capability_view(settings, EVERYTHING_PRESENT, load=no_ffmpeg_build)

    assert view.ai_available is False
    assert view.ai_reason_key == PENDING_MODEL_RELEASE_REASON


def test_the_restore_api_answers_404_with_the_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = release_settings(tmp_path, monkeypatch)
    app = FastAPI()
    app.include_router(restore_routes.router)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_user_store] = lambda: None

    response = TestClient(app).get("/api/v1/restore/capabilities")

    assert response.status_code == 404
    assert response.json()["detail"] == RESTORE_DISABLED_MESSAGE


async def test_a_restore_job_is_refused_before_touching_the_photo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = make_manager(release_settings(tmp_path, monkeypatch))

    with pytest.raises(FeatureDisabledError, match="not published yet"):
        await create_restore_job(manager, tmp_path / "never-read.png")


async def test_a_plain_upscale_job_is_not_affected_by_the_restore_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = make_manager(release_settings(tmp_path, monkeypatch))

    assert manager._restore_selection(None, None, None, 2) is None  # noqa: SLF001


def test_the_headless_restore_is_a_usage_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(UsageError, match="not published yet"):
        require_restore_enabled(release_settings(tmp_path, monkeypatch))


def test_the_gate_helpers_pass_when_the_flags_are_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = release_settings(tmp_path, monkeypatch, RESTORE_PHOTO_ENABLED=True, CCTV_AI_ENABLED=True)

    ensure_restore_enabled(settings)
    ensure_cctv_ai_enabled(settings)


def test_the_ai_lane_has_no_runner_when_it_is_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = release_settings(tmp_path, monkeypatch)

    assert enhance_runner_entry(settings, object(), None, object()) == {}


def test_the_ai_lane_has_its_runner_when_it_is_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = release_settings(tmp_path, monkeypatch, CCTV_AI_ENABLED=True)

    assert set(enhance_runner_entry(settings, object(), None, object())) == {"enhance"}


def test_the_classic_lane_is_never_gated() -> None:
    check_lane_released("classic", ai_lane_enabled=False)


async def test_an_ai_job_is_refused_with_the_release_reason_not_a_pack_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CCTV_AI_ENABLED", raising=False)
    manager = make_cctv_manager(tmp_path, tasks=("clarify", "enhance", "roi_fusion"))
    write_fake_session(manager.settings)
    options = CctvOptions(task="enhance", session_token=TOKEN, steps=(), no_osd=True)

    error = await rejected(manager, options, device="dml:0")

    assert isinstance(error, CctvChainError)
    assert error.code == AI_LANE_DISABLED
    assert str(error) == CCTV_AI_DISABLED_MESSAGE
