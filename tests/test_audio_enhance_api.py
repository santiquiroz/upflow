from __future__ import annotations

import asyncio
import io
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.api.routes import create_video_job, resolve_video_job_fields, video_job_to_response
from app.config import Settings
from app.models import VideoUpscaleJob
from app.services.storage import StorageService
from app.services.video_job_manager import VideoJobManager

# ---------------------------------------------------------------------------
# Task 20 (6.1c) - audio_enhance job field / validation / API / response.
# Mirrors the fps_multiplier + target_fps test structure (test_fps_boost_api.py,
# test_target_fps.py): model default, resolve_video_job_fields pass-through,
# VideoJobManager validation (invalid value / keep_audio required / disabled vs
# not-installed), route-level accept/reject, response field exposure.
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
        keep_audio=True,
    )
    fields.update(overrides)
    return VideoUpscaleJob(**fields)


def _fake_deepfilter_install(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / "deep-filter.exe"
    binary.write_bytes(b"fake")
    return binary


def _fake_rnnoise_install(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    model = directory / "sh.rnnn"
    model.write_bytes(b"fake")
    return model


def make_settings_with_audio_enhance_available(
    tmp_path: Path, enable_audio_enhance: bool = True
) -> Settings:
    deepfilter_binary = _fake_deepfilter_install(tmp_path / "deepfilter-install")
    rnnoise_model = _fake_rnnoise_install(tmp_path / "rnnoise-install")
    return Settings(
        RUNTIME_DIR=str(tmp_path / "runtime"),
        ENABLE_AUDIO_ENHANCE=enable_audio_enhance,
        DEEPFILTER_BINARY=str(deepfilter_binary),
        RNNOISE_MODEL=str(rnnoise_model),
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


# ---------------------------------------------------------------------------
# Model default
# ---------------------------------------------------------------------------


def test_video_upscale_job_audio_enhance_defaults_to_none(tmp_path: Path) -> None:
    job = make_video_job(tmp_path / "clip.mp4")

    assert job.audio_enhance is None


# ---------------------------------------------------------------------------
# resolve_video_job_fields - pass-through, no profile default (mirrors target_fps)
# ---------------------------------------------------------------------------


def test_resolve_video_job_fields_defaults_audio_enhance_to_none() -> None:
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

    assert resolved.audio_enhance is None


def test_resolve_video_job_fields_keeps_explicit_audio_enhance() -> None:
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
        audio_enhance="rnnoise",
    )

    assert resolved.audio_enhance == "rnnoise"


# ---------------------------------------------------------------------------
# VideoJobManager._validate_request - audio_enhance validation
# ---------------------------------------------------------------------------


async def test_video_job_manager_accepts_valid_audio_enhance_mode(tmp_path: Path) -> None:
    settings = make_settings_with_audio_enhance_available(tmp_path)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), asyncio.Semaphore(1))
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
        keep_audio=True,
        audio_enhance="rnnoise",
    )

    assert job.audio_enhance == "rnnoise"


async def test_video_job_manager_accepts_audio_enhance_off(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), asyncio.Semaphore(1))
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
        audio_enhance=None,
    )

    assert job.audio_enhance is None


async def test_video_job_manager_rejects_unknown_audio_enhance_value(tmp_path: Path) -> None:
    settings = make_settings_with_audio_enhance_available(tmp_path)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), asyncio.Semaphore(1))
    source_path = settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    with pytest.raises(ValueError, match="audio_enhance"):
        await video_jobs.create_job(
            source_path=source_path,
            original_filename="clip.mp4",
            model_name="realesr-animevideov3-x2",
            scale=2,
            output_container="mp4",
            video_codec="libx264",
            video_preset="medium",
            crf=18,
            keep_audio=True,
            audio_enhance="not-a-real-mode",
        )


async def test_video_job_manager_rejects_audio_enhance_without_keep_audio(tmp_path: Path) -> None:
    settings = make_settings_with_audio_enhance_available(tmp_path)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), asyncio.Semaphore(1))
    source_path = settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    with pytest.raises(ValueError, match="(?i)keep_audio"):
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
            audio_enhance="rnnoise",
        )


async def test_video_job_manager_rejects_audio_enhance_when_disabled_by_config(tmp_path: Path) -> None:
    settings = make_settings_with_audio_enhance_available(tmp_path, enable_audio_enhance=False)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), asyncio.Semaphore(1))
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
            keep_audio=True,
            audio_enhance="rnnoise",
        )


async def test_video_job_manager_rejects_audio_enhance_when_not_installed(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        ENABLE_AUDIO_ENHANCE=True,
        DEEPFILTER_BINARY=str(tmp_path / "missing-deep-filter.exe"),
        RNNOISE_MODEL=str(tmp_path / "missing.rnnn"),
    )
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), asyncio.Semaphore(1))
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
            keep_audio=True,
            audio_enhance="rnnoise",
        )


def test_disabled_and_not_installed_audio_enhance_messages_are_distinct(tmp_path: Path) -> None:
    disabled_settings = make_settings_with_audio_enhance_available(
        tmp_path / "a", enable_audio_enhance=False
    )
    not_installed_settings = make_settings(
        tmp_path / "b",
        ENABLE_AUDIO_ENHANCE=True,
        DEEPFILTER_BINARY=str(tmp_path / "missing-deep-filter.exe"),
        RNNOISE_MODEL=str(tmp_path / "missing.rnnn"),
    )
    disabled_jobs = VideoJobManager(disabled_settings, FakeUpscaler(), FakeMediaTools(), asyncio.Semaphore(1))
    not_installed_jobs = VideoJobManager(
        not_installed_settings, FakeUpscaler(), FakeMediaTools(), asyncio.Semaphore(1)
    )

    with pytest.raises(ValueError) as disabled_exc:
        disabled_jobs._validate_audio_enhance_mode("rnnoise", keep_audio=True)
    with pytest.raises(ValueError) as not_installed_exc:
        not_installed_jobs._validate_audio_enhance_mode("rnnoise", keep_audio=True)

    assert str(disabled_exc.value) != str(not_installed_exc.value)


# ---------------------------------------------------------------------------
# Route-level: create_video_job accepts/rejects audio_enhance, response exposes it
# ---------------------------------------------------------------------------


async def test_create_video_job_route_accepts_valid_audio_enhance(tmp_path: Path) -> None:
    settings = make_settings_with_audio_enhance_available(tmp_path)
    storage = StorageService(settings)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), asyncio.Semaphore(1))

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
        keep_audio=True,
        fps_multiplier=None,
        target_fps=None,
        audio_enhance="rnnoise",
        video_jobs=video_jobs,
        storage=storage,
        settings=settings,
    )

    job = video_jobs.get_job(response.job_id)
    assert job is not None
    assert job.audio_enhance == "rnnoise"


async def test_create_video_job_route_rejects_invalid_audio_enhance(tmp_path: Path) -> None:
    settings = make_settings_with_audio_enhance_available(tmp_path)
    storage = StorageService(settings)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), asyncio.Semaphore(1))

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
            keep_audio=True,
            fps_multiplier=None,
            target_fps=None,
            audio_enhance="not-a-real-mode",
            video_jobs=video_jobs,
            storage=storage,
            settings=settings,
        )

    assert exc_info.value.status_code == 400
    assert list(settings.uploads_path.iterdir()) == []


async def test_create_video_job_route_rejects_audio_enhance_without_keep_audio(tmp_path: Path) -> None:
    settings = make_settings_with_audio_enhance_available(tmp_path)
    storage = StorageService(settings)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), asyncio.Semaphore(1))

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
            keep_audio=False,
            fps_multiplier=None,
            target_fps=None,
            audio_enhance="rnnoise",
            video_jobs=video_jobs,
            storage=storage,
            settings=settings,
        )

    assert exc_info.value.status_code == 400


async def test_create_video_job_route_rejects_audio_enhance_when_unavailable(tmp_path: Path) -> None:
    settings = make_settings(
        tmp_path,
        ENABLE_AUDIO_ENHANCE=True,
        DEEPFILTER_BINARY=str(tmp_path / "missing-deep-filter.exe"),
        RNNOISE_MODEL=str(tmp_path / "missing.rnnn"),
    )
    storage = StorageService(settings)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), asyncio.Semaphore(1))

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
            keep_audio=True,
            fps_multiplier=None,
            target_fps=None,
            audio_enhance="rnnoise",
            video_jobs=video_jobs,
            storage=storage,
            settings=settings,
        )

    assert exc_info.value.status_code == 400
    assert "not installed" in exc_info.value.detail.lower()


async def test_create_video_job_route_omitted_audio_enhance_defaults_to_none(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    video_jobs = VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), asyncio.Semaphore(1))

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
        video_jobs=video_jobs,
        storage=storage,
        settings=settings,
    )

    job = video_jobs.get_job(response.job_id)
    assert job is not None
    assert job.audio_enhance is None


# ---------------------------------------------------------------------------
# video_job_to_response exposes audio_enhance
# ---------------------------------------------------------------------------


def test_video_job_to_response_exposes_audio_enhance() -> None:
    job = make_video_job(Path("clip.mp4"), audio_enhance="deepfilter")

    response = video_job_to_response(job)

    assert response.audio_enhance == "deepfilter"


def test_video_job_response_serializes_audio_enhance_camel_case() -> None:
    job = make_video_job(Path("clip.mp4"), audio_enhance="deepfilter")

    serialized = video_job_to_response(job).model_dump(by_alias=True)

    assert serialized["audioEnhance"] == "deepfilter"
