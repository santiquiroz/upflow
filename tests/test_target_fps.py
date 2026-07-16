from __future__ import annotations

import asyncio
import io
from fractions import Fraction
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.api.routes import create_video_job, resolve_video_job_fields, video_job_to_response
from app.config import Settings
from app.models import VideoUpscaleJob
from app.services.device_semaphores import DeviceSemaphores
from app.services.storage import StorageService
from app.services.video_job_manager import VideoJobManager

# ---------------------------------------------------------------------------
# Task 15 (6.6) - TARGET_FPS mode: an absolute fps target (e.g. anime 23.976 ->
# 60 exact), mutually exclusive with fps_multiplier > 1.
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
    """avg_frame_rate defaults to anime NTSC (23.976) so target_fps=60 is realistic."""

    def __init__(self, avg_frame_rate: str = "24000/1001") -> None:
        self.avg_frame_rate = avg_frame_rate

    async def ffprobe_json(self, source_path: Path) -> dict:
        return {"streams": [{"codec_type": "video", "avg_frame_rate": self.avg_frame_rate}]}


class FakeDevicesService:
    def list_devices(self) -> list[dict]:
        return [{"id": "dml:0", "kind": "gpu", "name": "Fake GPU", "backend": "directml"}]

    def resolve_default(self, devices: list[dict]) -> dict:
        return devices[0]


def make_source(settings: Settings) -> Path:
    source_path = settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")
    return source_path


async def create_job_with_target_fps(
    video_jobs: VideoJobManager, source_path: Path, **overrides: object
) -> VideoUpscaleJob:
    fields = dict(
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
    fields.update(overrides)
    return await video_jobs.create_job(**fields)


# ---------------------------------------------------------------------------
# Model default
# ---------------------------------------------------------------------------


def test_video_upscale_job_target_fps_defaults_to_none(tmp_path: Path) -> None:
    job = make_video_job(tmp_path / "clip.mp4")

    assert job.target_fps is None


# ---------------------------------------------------------------------------
# resolve_video_job_fields - target_fps is never resolved from the profile
# ---------------------------------------------------------------------------


def test_resolve_video_job_fields_passes_through_explicit_target_fps() -> None:
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
        target_fps="60",
    )

    assert resolved.target_fps == "60"


def test_resolve_video_job_fields_defaults_target_fps_to_none_when_omitted() -> None:
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
        target_fps=None,
    )

    assert resolved.target_fps is None


# ---------------------------------------------------------------------------
# VideoJobManager._validate_request - target_fps validation
# ---------------------------------------------------------------------------


async def test_video_job_manager_accepts_target_fps_above_source(tmp_path: Path) -> None:
    settings = make_settings_with_rife_available(tmp_path)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))
    source_path = make_source(settings)

    job = await create_job_with_target_fps(video_jobs, source_path, target_fps="60")

    assert job.target_fps == "60"
    assert job.fps_multiplier == 1


async def test_video_job_manager_accepts_target_fps_at_cap_boundary(tmp_path: Path) -> None:
    settings = make_settings_with_rife_available(tmp_path)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))
    source_path = make_source(settings)

    job = await create_job_with_target_fps(video_jobs, source_path, target_fps="240")

    assert job.target_fps == "240"


@pytest.mark.parametrize("target_fps", ["23", "24000/1001", "10"])
async def test_video_job_manager_rejects_target_fps_at_or_below_source(
    tmp_path: Path, target_fps: str
) -> None:
    settings = make_settings_with_rife_available(tmp_path)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))
    source_path = make_source(settings)

    with pytest.raises(ValueError, match="(?i)greater than"):
        await create_job_with_target_fps(video_jobs, source_path, target_fps=target_fps)


@pytest.mark.parametrize("target_fps", ["abc", "0", "-30", "300"])
async def test_video_job_manager_rejects_invalid_target_fps(tmp_path: Path, target_fps: str) -> None:
    settings = make_settings_with_rife_available(tmp_path)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))
    source_path = make_source(settings)

    with pytest.raises(ValueError):
        await create_job_with_target_fps(video_jobs, source_path, target_fps=target_fps)


async def test_video_job_manager_rejects_target_fps_with_multiplier_simultaneously(tmp_path: Path) -> None:
    settings = make_settings_with_rife_available(tmp_path)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))
    source_path = make_source(settings)

    with pytest.raises(ValueError, match="(?i)mutually exclusive"):
        await create_job_with_target_fps(video_jobs, source_path, target_fps="60", fps_multiplier=2)


async def test_video_job_manager_rejects_target_fps_when_interpolation_disabled_by_config(
    tmp_path: Path,
) -> None:
    settings = make_settings_with_rife_available(tmp_path, enable_interpolation=False)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))
    source_path = make_source(settings)

    with pytest.raises(ValueError, match="(?i)disabled"):
        await create_job_with_target_fps(video_jobs, source_path, target_fps="60")


async def test_video_job_manager_rejects_target_fps_when_rife_not_installed(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path, ENABLE_INTERPOLATION=True, RIFE_BINARY=str(tmp_path / "missing-rife.exe")
    )
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))
    source_path = make_source(settings)

    with pytest.raises(ValueError, match="(?i)not installed"):
        await create_job_with_target_fps(video_jobs, source_path, target_fps="60")


def test_target_fps_disabled_and_not_installed_messages_are_distinct(tmp_path: Path) -> None:
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
        disabled_jobs._validate_target_fps("60", Fraction(24000, 1001))
    with pytest.raises(ValueError) as not_installed_exc:
        not_installed_jobs._validate_target_fps("60", Fraction(24000, 1001))

    assert str(disabled_exc.value) != str(not_installed_exc.value)


# ---------------------------------------------------------------------------
# Route-level: create_video_job accepts/rejects target_fps, response exposes it
# ---------------------------------------------------------------------------


async def test_create_video_job_route_accepts_target_fps(tmp_path: Path) -> None:
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
        fps_multiplier=None,
        target_fps="60",
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
    assert job.target_fps == "60"


async def test_create_video_job_route_rejects_target_fps_and_multiplier_together(tmp_path: Path) -> None:
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
            fps_multiplier=2,
            target_fps="60",
            audio_enhance=None,
            audio_restore=None,
            model_id=None,
            device=None,
            video_jobs=video_jobs,
            storage=storage,
            settings=settings,
            devices=FakeDevicesService(),
        )

    assert exc_info.value.status_code == 400
    assert list(settings.uploads_path.iterdir()) == []


async def test_create_video_job_route_rejects_target_fps_below_source(tmp_path: Path) -> None:
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
            fps_multiplier=None,
            target_fps="10",
            audio_enhance=None,
            audio_restore=None,
            model_id=None,
            device=None,
            video_jobs=video_jobs,
            storage=storage,
            settings=settings,
            devices=FakeDevicesService(),
        )

    assert exc_info.value.status_code == 400
    assert "greater than" in exc_info.value.detail.lower()


async def test_create_video_job_route_omitted_target_fps_defaults_to_none(tmp_path: Path) -> None:
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
    assert job.target_fps is None


# ---------------------------------------------------------------------------
# video_job_to_response / schema exposes target_fps
# ---------------------------------------------------------------------------


def test_video_job_to_response_exposes_target_fps() -> None:
    job = make_video_job(Path("clip.mp4"), target_fps="60")

    response = video_job_to_response(job)

    assert response.target_fps == "60"


def test_video_job_response_serializes_target_fps_alias() -> None:
    job = make_video_job(Path("clip.mp4"), target_fps="60000/1001")

    serialized = video_job_to_response(job).model_dump(by_alias=True)

    assert serialized["targetFps"] == "60000/1001"


def test_video_job_response_target_fps_defaults_to_none_alias() -> None:
    job = make_video_job(Path("clip.mp4"))

    serialized = video_job_to_response(job).model_dump(by_alias=True)

    assert serialized["targetFps"] is None
