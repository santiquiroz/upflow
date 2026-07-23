from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile
import io

from app.api.routes import create_video_job
from app.config import GMFSS_ENGINE, INTERP_ENGINES, RIFE_ENGINE, Settings
from app.models import VideoUpscaleJob
from app.services.device_semaphores import DeviceSemaphores
from app.services.engines.gmfss.assets import GRAPH_NAMES
from app.services.storage import StorageService
from app.services.video_job_manager import VideoJobManager

# ---------------------------------------------------------------------------
# Task 4.2 - interp_engine selector: config constants, per-job validation
# (mirrors _validate_audio_restore_mode's disabled-vs-not-installed split),
# and route wiring. Default is ALWAYS "rife" -- GMFSS is opt-in per job.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), **overrides)


def make_upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


def _fake_rife_install(tmp_path: Path, model_name: str = "rife-v4.25") -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    binary = tmp_path / "rife-ncnn-vulkan.exe"
    binary.write_bytes(b"fake")
    models_dir = tmp_path / "rife-models"
    (models_dir / model_name).mkdir(parents=True)
    return binary, models_dir


def make_settings_with_rife_available(tmp_path: Path) -> Settings:
    binary, models_dir = _fake_rife_install(tmp_path)
    return Settings(
        RUNTIME_DIR=str(tmp_path / "runtime"),
        ENABLE_INTERPOLATION=True,
        RIFE_BINARY=str(binary),
        RIFE_MODELS_DIR=str(models_dir),
    )


def _fake_gmfss_install(tmp_path: Path) -> Path:
    model_dir = tmp_path / "gmfss-models"
    model_dir.mkdir(parents=True)
    manifest = {
        "resolution": {"fixed_padded_hw": [16, 24]},
        "required_files": ["manifest.json"] + [f"{name}.onnx" for name in GRAPH_NAMES],
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest))
    for name in GRAPH_NAMES:
        (model_dir / f"{name}.onnx").write_bytes(b"fake")
    return model_dir


def make_settings_with_gmfss_available(tmp_path: Path) -> Settings:
    model_dir = _fake_gmfss_install(tmp_path)
    return Settings(
        RUNTIME_DIR=str(tmp_path / "runtime"),
        ENABLE_GMFSS=True,
        GMFSS_MODEL_DIR=str(model_dir),
    )


class FakeUpscaler:
    async def run(self, job: VideoUpscaleJob, fps_multiplier: int = 1) -> Path:
        return job.source_path


class FakeMediaTools:
    async def ffprobe_json(self, source_path: Path) -> dict:
        return {"streams": [{"codec_type": "video"}]}


class FakeDevicesService:
    def list_devices(self) -> list[dict]:
        return [{"id": "dml:0", "kind": "gpu", "name": "Fake GPU", "backend": "directml"}]

    def resolve_default(self, devices: list[dict]) -> dict:
        return devices[0]


# ---------------------------------------------------------------------------
# config constants
# ---------------------------------------------------------------------------


def test_interp_engines_constant() -> None:
    assert INTERP_ENGINES == frozenset({"rife", "gmfss"})
    assert RIFE_ENGINE == "rife"
    assert GMFSS_ENGINE == "gmfss"


def test_video_upscale_job_interp_engine_defaults_to_rife(tmp_path: Path) -> None:
    job = VideoUpscaleJob(
        source_path=tmp_path / "clip.mp4",
        original_filename="clip.mp4",
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
    )

    assert job.interp_engine == "rife"


# ---------------------------------------------------------------------------
# Settings.interp_engine_available
# ---------------------------------------------------------------------------


def test_interp_engine_available_rife_follows_interpolation_available(tmp_path: Path) -> None:
    available = make_settings_with_rife_available(tmp_path)
    unavailable = Settings(
        RUNTIME_DIR=str(tmp_path / "b"),
        ENABLE_INTERPOLATION=True,
        RIFE_BINARY=str(tmp_path / "missing-rife.exe"),
    )

    assert available.interp_engine_available("rife") is True
    assert unavailable.interp_engine_available("rife") is False


def test_interp_engine_available_gmfss_follows_gmfss_available(tmp_path: Path) -> None:
    available = make_settings_with_gmfss_available(tmp_path)
    disabled = Settings(RUNTIME_DIR=str(tmp_path / "b"), ENABLE_GMFSS=False)

    assert available.interp_engine_available("gmfss") is True
    assert disabled.interp_engine_available("gmfss") is False


def test_interp_engine_available_unknown_engine_is_false(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    assert settings.interp_engine_available("nope") is False


# ---------------------------------------------------------------------------
# VideoJobManager validation - disabled vs not-installed split (mirrors
# validate_restore_mode_ready), and unknown interp_engine rejection.
# ---------------------------------------------------------------------------


async def test_job_manager_rejects_unknown_interp_engine(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))
    source_path = settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    with pytest.raises(ValueError, match="interp_engine"):
        await video_jobs.create_job(
            source_path=source_path,
            original_filename="clip.mp4",
            model_name="realesr-animevideov3-x2",
            scale=2,
            output_container="mp4",
            video_codec="libx264",
            video_preset="medium",
            crf=18,
            keep_audio=False,
            interp_engine="bogus",
        )


async def test_job_manager_accepts_rife_default_without_interpolation_requested(tmp_path: Path) -> None:
    # interp_engine defaults to "rife" on every job -- it must never fail
    # validation just because RIFE isn't installed, as long as no
    # interpolation was actually requested (fps_multiplier=1, no target_fps).
    settings = make_settings(tmp_path)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))
    source_path = settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    job = await video_jobs.create_job(
        source_path=source_path,
        original_filename="clip.mp4",
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
    )

    assert job.interp_engine == "rife"


async def test_job_manager_accepts_gmfss_when_enabled_and_installed(tmp_path: Path) -> None:
    settings = make_settings_with_gmfss_available(tmp_path)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))
    source_path = settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    job = await video_jobs.create_job(
        source_path=source_path,
        original_filename="clip.mp4",
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
        fps_multiplier=2,
        interp_engine="gmfss",
    )

    assert job.interp_engine == "gmfss"


async def test_job_manager_accepts_gmfss_when_interpolation_flag_explicitly_off(tmp_path: Path) -> None:
    # Regression for the whole-branch review finding: ENABLE_INTERPOLATION
    # (RIFE's capability flag) and ENABLE_GMFSS are independent -- a job
    # requesting GMFSS must succeed purely on ENABLE_GMFSS + installed models,
    # even with ENABLE_INTERPOLATION explicitly false (not just left at its
    # default), so GMFSS-only configs are reachable without also turning on
    # RIFE's flag.
    model_dir = _fake_gmfss_install(tmp_path)
    settings = Settings(
        RUNTIME_DIR=str(tmp_path / "runtime"),
        ENABLE_INTERPOLATION=False,
        ENABLE_GMFSS=True,
        GMFSS_MODEL_DIR=str(model_dir),
    )
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))
    source_path = settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    job = await video_jobs.create_job(
        source_path=source_path,
        original_filename="clip.mp4",
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
        fps_multiplier=2,
        interp_engine="gmfss",
    )

    assert job.interp_engine == "gmfss"


async def test_job_manager_accepts_rife_when_gmfss_explicitly_off(tmp_path: Path) -> None:
    # Inverse of the regression above: a RIFE-only config (ENABLE_GMFSS
    # explicitly false, ENABLE_INTERPOLATION true + RIFE installed) must keep
    # working exactly as before this fix -- the per-engine gate must not have
    # regressed the pre-existing default path.
    binary, models_dir = _fake_rife_install(tmp_path)
    settings = Settings(
        RUNTIME_DIR=str(tmp_path / "runtime"),
        ENABLE_INTERPOLATION=True,
        RIFE_BINARY=str(binary),
        RIFE_MODELS_DIR=str(models_dir),
        ENABLE_GMFSS=False,
    )
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))
    source_path = settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    job = await video_jobs.create_job(
        source_path=source_path,
        original_filename="clip.mp4",
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
        fps_multiplier=2,
        interp_engine="rife",
    )

    assert job.interp_engine == "rife"


async def test_job_manager_rejects_gmfss_when_disabled_by_config(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, ENABLE_GMFSS=False)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))
    source_path = settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    with pytest.raises(ValueError, match="(?i)disabled"):
        await video_jobs.create_job(
            source_path=source_path,
            original_filename="clip.mp4",
            model_name="realesr-animevideov3-x2",
            scale=2,
            output_container="mp4",
            video_codec="libx264",
            video_preset="medium",
            crf=18,
            keep_audio=False,
            fps_multiplier=2,
            interp_engine="gmfss",
        )


async def test_job_manager_rejects_gmfss_when_models_not_installed(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, ENABLE_GMFSS=True, GMFSS_MODEL_DIR=str(tmp_path / "missing"))
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))
    source_path = settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    with pytest.raises(ValueError, match="(?i)not installed"):
        await video_jobs.create_job(
            source_path=source_path,
            original_filename="clip.mp4",
            model_name="realesr-animevideov3-x2",
            scale=2,
            output_container="mp4",
            video_codec="libx264",
            video_preset="medium",
            crf=18,
            keep_audio=False,
            fps_multiplier=2,
            interp_engine="gmfss",
        )


async def test_gmfss_disabled_and_not_installed_messages_are_distinct(tmp_path: Path) -> None:
    disabled_settings = make_settings(tmp_path / "a", ENABLE_GMFSS=False)
    not_installed_settings = make_settings(
        tmp_path / "b", ENABLE_GMFSS=True, GMFSS_MODEL_DIR=str(tmp_path / "missing")
    )
    disabled_jobs = VideoJobManager(
        disabled_settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(disabled_settings)
    )
    not_installed_jobs = VideoJobManager(
        not_installed_settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(not_installed_settings)
    )

    with pytest.raises(ValueError) as disabled_exc:
        disabled_jobs._validate_fps_multiplier(2, "gmfss")
    with pytest.raises(ValueError) as not_installed_exc:
        not_installed_jobs._validate_fps_multiplier(2, "gmfss")

    assert str(disabled_exc.value) != str(not_installed_exc.value)


async def test_job_manager_gmfss_readiness_not_checked_when_interpolation_not_requested(
    tmp_path: Path,
) -> None:
    # interp_engine="gmfss" with fps_multiplier=1 and no target_fps: GMFSS is
    # never actually invoked, so its readiness must not gate job creation.
    settings = make_settings(tmp_path, ENABLE_GMFSS=False)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))
    source_path = settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    job = await video_jobs.create_job(
        source_path=source_path,
        original_filename="clip.mp4",
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
        interp_engine="gmfss",
    )

    assert job.interp_engine == "gmfss"


# ---------------------------------------------------------------------------
# Route-level: create_video_job accepts interp_engine, response exposes it
# ---------------------------------------------------------------------------


async def test_create_video_job_route_accepts_interp_engine(tmp_path: Path) -> None:
    settings = make_settings_with_gmfss_available(tmp_path)
    storage = StorageService(settings)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))

    response = await create_video_job(
        request=None,
        file=make_upload("clip.mp4", b"fake-video-bytes"),
        profile_key="anime-balanced-2x",
        model_name=None,
        scale=None,
        output_container=None,
        video_codec=None,
        video_preset=None,
        crf=None,
        keep_audio=None,
        fps_multiplier=2,
        target_fps=None,
        audio_enhance=None,
        audio_restore=None,
        interp_engine="gmfss",
        model_id=None,
        device=None,
        video_jobs=video_jobs,
        storage=storage,
        settings=settings,
        devices=FakeDevicesService(),
    )

    job = video_jobs.get_job(response.job_id)
    assert job is not None
    assert job.interp_engine == "gmfss"


async def test_create_video_job_route_omitted_interp_engine_defaults_to_rife(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))

    response = await create_video_job(
        request=None,
        file=make_upload("clip.mp4", b"fake-video-bytes"),
        profile_key="anime-balanced-2x",
        model_name=None,
        scale=None,
        output_container=None,
        video_codec=None,
        video_preset=None,
        crf=None,
        keep_audio=None,
        fps_multiplier=None,
        target_fps=None,
        audio_enhance=None,
        audio_restore=None,
        model_id=None,
        device=None,
        video_jobs=video_jobs,
        storage=storage,
        settings=settings,
        devices=FakeDevicesService(),
    )

    job = video_jobs.get_job(response.job_id)
    assert job is not None
    assert job.interp_engine == "rife"


async def test_create_video_job_route_rejects_unknown_interp_engine(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))

    with pytest.raises(HTTPException) as exc_info:
        await create_video_job(
            request=None,
            file=make_upload("clip.mp4", b"fake-video-bytes"),
            profile_key="anime-balanced-2x",
            model_name=None,
            scale=None,
            output_container=None,
            video_codec=None,
            video_preset=None,
            crf=None,
            keep_audio=None,
            fps_multiplier=None,
            target_fps=None,
            audio_enhance=None,
            audio_restore=None,
            interp_engine="bogus",
            model_id=None,
            device=None,
            video_jobs=video_jobs,
            storage=storage,
            settings=settings,
            devices=FakeDevicesService(),
        )

    assert exc_info.value.status_code == 400


def test_video_job_to_response_exposes_interp_engine() -> None:
    from app.api.routes import video_job_to_response

    job = VideoUpscaleJob(
        source_path=Path("clip.mp4"),
        original_filename="clip.mp4",
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
        interp_engine="gmfss",
    )

    response = video_job_to_response(job)

    assert response.interp_engine == "gmfss"
    serialized = response.model_dump(by_alias=True)
    assert serialized["interpEngine"] == "gmfss"


# ---------------------------------------------------------------------------
# capabilities: interpEngines only lists engines that are actually available
# ---------------------------------------------------------------------------


async def test_video_capabilities_lists_only_available_interp_engines(tmp_path: Path) -> None:
    from app.api.routes import video_capabilities

    settings = make_settings_with_rife_available(tmp_path)

    response = await video_capabilities(settings=settings)

    assert response.interp_engines == ["rife"]


async def test_video_capabilities_lists_gmfss_when_available(tmp_path: Path) -> None:
    from app.api.routes import video_capabilities

    settings = make_settings_with_gmfss_available(tmp_path)

    response = await video_capabilities(settings=settings)

    assert "gmfss" in response.interp_engines
