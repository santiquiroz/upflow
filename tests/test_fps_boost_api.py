from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.api.routes import create_video_job, resolve_video_job_fields
from app.config import Settings
from app.models import VideoUpscaleJob
from app.services.device_semaphores import DeviceSemaphores
from app.services.storage import StorageService
from app.services.video_job_manager import VideoJobManager

# ---------------------------------------------------------------------------
# Task 13 (4.5/4.6) - fps_multiplier job field / validation / API / response.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path), **overrides)


def make_upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


def make_profile() -> dict:
    return {
        "key": "anime-balanced-2x",
        "label": "Anime Balanced 2x",
        "category": "anime",
        "description": "test profile",
        "model_key": "realesr-animevideov3-x2",
        "scale": 2,
        "video_codec": "libx264",
        "video_preset": "medium",
        "crf": 17,
        "keep_audio": True,
        "fps_multiplier": 1,
    }


def make_video_job(source_path: Path, **overrides: object) -> VideoUpscaleJob:
    fields = dict(
        source_path=source_path,
        original_filename=source_path.name,
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
    )
    fields.update(overrides)
    return VideoUpscaleJob(**fields)


def _fake_rife_install(tmp_path: Path, model_name: str = "rife-v4.6") -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    binary = tmp_path / "rife-ncnn-vulkan.exe"
    binary.write_bytes(b"fake")
    models_dir = tmp_path / "rife-models"
    (models_dir / model_name).mkdir(parents=True)
    return binary, models_dir


def make_settings_with_rife_available(tmp_path: Path, enable_interpolation: bool = True) -> Settings:
    binary, models_dir = _fake_rife_install(tmp_path)
    return Settings(
        RUNTIME_DIR=str(tmp_path / "runtime"),
        ENABLE_INTERPOLATION=enable_interpolation,
        RIFE_BINARY=str(binary),
        RIFE_MODELS_DIR=str(models_dir),
    )


class FakeUpscaler:
    def __init__(self) -> None:
        self.calls: list[tuple[VideoUpscaleJob, int]] = []

    async def run(self, job: VideoUpscaleJob, fps_multiplier: int = 1) -> Path:
        self.calls.append((job, fps_multiplier))
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
# Model default
# ---------------------------------------------------------------------------


def test_video_upscale_job_fps_multiplier_defaults_to_one(tmp_path: Path) -> None:
    job = make_video_job(tmp_path / "clip.mp4")

    assert job.fps_multiplier == 1


# ---------------------------------------------------------------------------
# resolve_video_job_fields - is-not-None resolution (not `or`)
# ---------------------------------------------------------------------------


def test_resolve_video_job_fields_defaults_fps_multiplier_to_profile_value() -> None:
    resolved = resolve_video_job_fields(
        make_profile(),
        model_name=None,
        scale=None,
        output_container=None,
        video_codec=None,
        video_preset=None,
        crf=None,
        keep_audio=None,
        fps_multiplier=None,
    )

    assert resolved.fps_multiplier == 1


def test_resolve_video_job_fields_keeps_explicit_fps_multiplier() -> None:
    resolved = resolve_video_job_fields(
        make_profile(),
        model_name=None,
        scale=None,
        output_container=None,
        video_codec=None,
        video_preset=None,
        crf=None,
        keep_audio=None,
        fps_multiplier=3,
    )

    assert resolved.fps_multiplier == 3


def test_resolve_video_job_fields_keeps_explicit_fps_multiplier_zero_not_silently_dropped() -> None:
    """Mirrors the crf/scale `is not None` fix: an explicit 0 must reach validation, not be
    silently replaced by the profile default."""
    resolved = resolve_video_job_fields(
        make_profile(),
        model_name=None,
        scale=None,
        output_container=None,
        video_codec=None,
        video_preset=None,
        crf=None,
        keep_audio=None,
        fps_multiplier=0,
    )

    assert resolved.fps_multiplier == 0


# ---------------------------------------------------------------------------
# VideoJobManager._validate_request - fps_multiplier validation
# ---------------------------------------------------------------------------


async def test_video_job_manager_accepts_fps_multiplier_one_without_interpolation(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))
    source_path = tmp_path / "clip.mp4"
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
        fps_multiplier=1,
    )

    assert job.fps_multiplier == 1


@pytest.mark.parametrize("multiplier", [2, 3, 4])
async def test_video_job_manager_accepts_allowed_multipliers_when_interpolation_available(
    tmp_path: Path, multiplier: int
) -> None:
    settings = make_settings_with_rife_available(tmp_path)
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
        fps_multiplier=multiplier,
    )

    assert job.fps_multiplier == multiplier


@pytest.mark.parametrize("multiplier", [5, 0, -1])
async def test_video_job_manager_rejects_invalid_fps_multiplier(tmp_path: Path, multiplier: int) -> None:
    settings = make_settings_with_rife_available(tmp_path)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))
    source_path = settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    with pytest.raises(ValueError):
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
            fps_multiplier=multiplier,
        )


async def test_video_job_manager_rejects_multiplier_when_interpolation_disabled_by_config(
    tmp_path: Path,
) -> None:
    settings = make_settings_with_rife_available(tmp_path, enable_interpolation=False)
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
        )


async def test_video_job_manager_rejects_multiplier_when_rife_not_installed(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path, ENABLE_INTERPOLATION=True, RIFE_BINARY=str(tmp_path / "missing-rife.exe")
    )
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
        )


def test_disabled_and_not_installed_messages_are_distinct(tmp_path: Path) -> None:
    disabled_settings = make_settings_with_rife_available(tmp_path / "a", enable_interpolation=False)
    not_installed_settings = make_settings(
        tmp_path / "b", ENABLE_INTERPOLATION=True, RIFE_BINARY=str(tmp_path / "missing-rife.exe")
    )
    disabled_jobs = VideoJobManager(
        disabled_settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(disabled_settings)
    )
    not_installed_jobs = VideoJobManager(
        not_installed_settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(not_installed_settings)
    )

    with pytest.raises(ValueError) as disabled_exc:
        disabled_jobs._validate_fps_multiplier(2)
    with pytest.raises(ValueError) as not_installed_exc:
        not_installed_jobs._validate_fps_multiplier(2)

    assert str(disabled_exc.value) != str(not_installed_exc.value)


# ---------------------------------------------------------------------------
# Route-level: create_video_job accepts/rejects fps_multiplier, response exposes it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("multiplier", [2, 3, 4])
async def test_create_video_job_route_accepts_allowed_multipliers(tmp_path: Path, multiplier: int) -> None:
    settings = make_settings_with_rife_available(tmp_path)
    storage = StorageService(settings)
    upscaler = FakeUpscaler()
    video_jobs = VideoJobManager(settings, upscaler, FakeMediaTools(), DeviceSemaphores(settings))

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
        fps_multiplier=multiplier,
        target_fps=None,
        audio_enhance=None,
        model_id=None,
        device=None,
        video_jobs=video_jobs,
        storage=storage,
        settings=settings,
        devices=FakeDevicesService(),
    )

    job = video_jobs.get_job(response.job_id)
    assert job is not None
    assert job.fps_multiplier == multiplier


@pytest.mark.parametrize("multiplier", [5, 0, -1])
async def test_create_video_job_route_rejects_invalid_multiplier(tmp_path: Path, multiplier: int) -> None:
    settings = make_settings_with_rife_available(tmp_path)
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
            fps_multiplier=multiplier,
            target_fps=None,
            audio_enhance=None,
            model_id=None,
            device=None,
            video_jobs=video_jobs,
            storage=storage,
            settings=settings,
            devices=FakeDevicesService(),
        )

    assert exc_info.value.status_code == 400
    assert list(settings.uploads_path.iterdir()) == []


async def test_create_video_job_route_rejects_when_interpolation_unavailable(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path, ENABLE_INTERPOLATION=True, RIFE_BINARY=str(tmp_path / "missing-rife.exe")
    )
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
            fps_multiplier=2,
            target_fps=None,
            audio_enhance=None,
            model_id=None,
            device=None,
            video_jobs=video_jobs,
            storage=storage,
            settings=settings,
            devices=FakeDevicesService(),
        )

    assert exc_info.value.status_code == 400
    assert "not installed" in exc_info.value.detail.lower()


async def test_create_video_job_route_omitted_fps_multiplier_defaults_to_off(tmp_path: Path) -> None:
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
        model_id=None,
        device=None,
        video_jobs=video_jobs,
        storage=storage,
        settings=settings,
        devices=FakeDevicesService(),
    )

    job = video_jobs.get_job(response.job_id)
    assert job is not None
    assert job.fps_multiplier == 1


# ---------------------------------------------------------------------------
# video_job_to_response exposes fps_multiplier
# ---------------------------------------------------------------------------


def test_video_job_to_response_exposes_fps_multiplier() -> None:
    from app.api.routes import video_job_to_response

    job = make_video_job(Path("clip.mp4"), fps_multiplier=3)

    response = video_job_to_response(job)

    assert response.fps_multiplier == 3


def test_video_job_response_serializes_output_fps_metadata() -> None:
    from app.api.routes import video_job_to_response

    job = make_video_job(Path("clip.mp4"), fps_multiplier=2)
    job.metadata["outputFps"] = "60/1"

    serialized = video_job_to_response(job).model_dump(by_alias=True)

    assert serialized["fpsMultiplier"] == 2
    assert serialized["metadata"]["outputFps"] == "60/1"


# ---------------------------------------------------------------------------
# Worker passes job.fps_multiplier through to the upscaler
# ---------------------------------------------------------------------------


async def test_worker_passes_job_fps_multiplier_to_upscaler(tmp_path: Path) -> None:
    settings = make_settings_with_rife_available(tmp_path)
    upscaler = FakeUpscaler()
    video_jobs = VideoJobManager(settings, upscaler, FakeMediaTools(), DeviceSemaphores(settings))

    source_path = settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    await video_jobs.start()
    try:
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
            fps_multiplier=3,
        )
        await video_jobs.queue.join()
    finally:
        await video_jobs.stop()

    assert len(upscaler.calls) == 1
    _, multiplier_received = upscaler.calls[0]
    assert multiplier_received == 3
