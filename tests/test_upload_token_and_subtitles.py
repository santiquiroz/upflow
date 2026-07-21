from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

import asyncio

from app.api.routes import create_video_job, video_job_to_response
from app.config import Settings
from app.models import JobStatus, VideoUpscaleJob
from app.services.device_semaphores import DeviceSemaphores
from app.services.storage import StorageService
from app.services.video_job_manager import VideoJobManager

# ---------------------------------------------------------------------------
# Fase A Task 2 - upload_token resolution + audio_track_indices + keep_subtitles
# on VideoJobManager.create_job, and the create_video_job route wiring that
# lets POST /video/jobs reference a staged /video/analyze upload by token
# instead of re-uploading the file.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path))


def make_settings_with_audio_restore(tmp_path: Path) -> Settings:
    model_path = tmp_path / "models" / "apollo.onnx"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"fake-apollo-model")
    return Settings(
        RUNTIME_DIR=str(tmp_path),
        ENABLE_AUDIO_RESTORE=True,
        APOLLO_RESTORE_MODEL=str(model_path),
    )


def make_upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


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


def make_video_job_manager(settings: Settings) -> VideoJobManager:
    return VideoJobManager(settings, FakeUpscaler(), FakeMediaTools(), DeviceSemaphores(settings))


def stage_upload(settings: Settings, token: str, safe_name: str, content: bytes = b"fake-mp4-bytes") -> Path:
    staged = settings.uploads_path / f"{token}-{safe_name}"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(content)
    return staged


def create_job_kwargs(**overrides: object) -> dict:
    base = dict(
        original_filename="clip.mp4",
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=True,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# VideoJobManager.create_job - upload_token resolution
# ---------------------------------------------------------------------------


async def test_create_job_resolves_source_path_from_upload_token(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_video_job_manager(settings)
    staged = stage_upload(settings, "abc123", "clip.mp4")

    job = await manager.create_job(**create_job_kwargs(source_path=None, upload_token="abc123"))

    assert job.source_path == staged


async def test_create_job_raises_when_upload_token_has_no_staged_file(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_video_job_manager(settings)

    with pytest.raises(ValueError, match="upload_token"):
        await manager.create_job(**create_job_kwargs(source_path=None, upload_token="does-not-exist"))


async def test_create_job_raises_when_neither_source_path_nor_upload_token(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_video_job_manager(settings)

    with pytest.raises(ValueError, match="source_path or upload_token"):
        await manager.create_job(**create_job_kwargs(source_path=None, upload_token=None))


# ---------------------------------------------------------------------------
# VideoJobManager.create_job - audio_track_indices / keep_subtitles defaults
# and container auto-upgrade
# ---------------------------------------------------------------------------


async def test_create_job_defaults_audio_track_indices_and_keep_subtitles(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_video_job_manager(settings)
    source_path = settings.uploads_path / "existing.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    job = await manager.create_job(**create_job_kwargs(source_path=source_path))

    assert job.audio_track_indices is None
    assert job.keep_subtitles is False
    assert job.output_container == "mp4"
    assert "containerUpgradedReason" not in job.metadata


async def test_create_job_passes_through_explicit_audio_track_indices(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_video_job_manager(settings)
    source_path = settings.uploads_path / "existing.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    job = await manager.create_job(
        **create_job_kwargs(source_path=source_path, audio_track_indices=[1, 2])
    )

    assert job.audio_track_indices == [1, 2]


async def test_create_job_upgrades_container_to_mkv_when_keep_subtitles(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_video_job_manager(settings)
    source_path = settings.uploads_path / "existing.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    job = await manager.create_job(
        **create_job_kwargs(source_path=source_path, keep_subtitles=True)
    )

    assert job.output_container == "mkv"
    assert "subtitles" in job.metadata["containerUpgradedReason"]


async def test_create_job_keeps_mkv_without_upgrade_note_when_already_mkv(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_video_job_manager(settings)
    source_path = settings.uploads_path / "existing.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    job = await manager.create_job(
        **create_job_kwargs(source_path=source_path, output_container="mkv", keep_subtitles=True)
    )

    assert job.output_container == "mkv"
    assert "containerUpgradedReason" not in job.metadata


# ---------------------------------------------------------------------------
# VideoJobManager.create_job - audio_output_format ("auto"|"flac"|"aac",
# Fase C Task 8) auto-upgrades the container to mkv for lossless FLAC the same
# way keep_subtitles does above; both reasons can coexist in the same note.
# ---------------------------------------------------------------------------


async def test_create_job_defaults_audio_output_format_to_auto_without_upgrade(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_video_job_manager(settings)
    source_path = settings.uploads_path / "existing.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    job = await manager.create_job(**create_job_kwargs(source_path=source_path))

    assert job.audio_output_format == "auto"
    assert job.output_container == "mp4"
    assert "containerUpgradedReason" not in job.metadata


async def test_create_job_auto_format_upgrades_container_when_restore_active(tmp_path: Path) -> None:
    settings = make_settings_with_audio_restore(tmp_path)
    manager = make_video_job_manager(settings)
    source_path = settings.uploads_path / "existing.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    job = await manager.create_job(
        **create_job_kwargs(
            source_path=source_path, audio_restore="apollo", audio_output_format="auto"
        )
    )

    assert job.output_container == "mkv"
    assert job.audio_output_format == "auto"
    assert "flac" in job.metadata["containerUpgradedReason"].lower()


async def test_create_job_explicit_aac_does_not_upgrade_container_even_with_restore(
    tmp_path: Path,
) -> None:
    settings = make_settings_with_audio_restore(tmp_path)
    manager = make_video_job_manager(settings)
    source_path = settings.uploads_path / "existing.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    job = await manager.create_job(
        **create_job_kwargs(
            source_path=source_path, audio_restore="apollo", audio_output_format="aac"
        )
    )

    assert job.output_container == "mp4"
    assert "containerUpgradedReason" not in job.metadata


async def test_create_job_restore_active_without_flac_format_keeps_container(tmp_path: Path) -> None:
    # "auto" without a restore active must NOT upgrade -- only an ACTIVE
    # restore (or an explicit flac/auto-with-restore) wants lossless audio.
    settings = make_settings_with_audio_restore(tmp_path)
    manager = make_video_job_manager(settings)
    source_path = settings.uploads_path / "existing.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    job = await manager.create_job(
        **create_job_kwargs(source_path=source_path, audio_output_format="auto")
    )

    assert job.output_container == "mp4"
    assert "containerUpgradedReason" not in job.metadata


async def test_create_job_upgrades_container_once_when_subtitles_and_flac_both_apply(
    tmp_path: Path,
) -> None:
    settings = make_settings_with_audio_restore(tmp_path)
    manager = make_video_job_manager(settings)
    source_path = settings.uploads_path / "existing.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")

    job = await manager.create_job(
        **create_job_kwargs(
            source_path=source_path,
            keep_subtitles=True,
            audio_restore="apollo",
            audio_output_format="auto",
        )
    )

    assert job.output_container == "mkv"
    reason = job.metadata["containerUpgradedReason"].lower()
    assert "subtitles" in reason
    assert "flac" in reason


# ---------------------------------------------------------------------------
# create_video_job route - exactly-one-of file/upload_token validation
# ---------------------------------------------------------------------------


async def test_create_video_job_route_rejects_when_neither_file_nor_upload_token(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    video_jobs = make_video_job_manager(settings)

    with pytest.raises(HTTPException) as exc_info:
        await create_video_job(
            request=None,
            file=None,
            upload_token=None,
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

    assert exc_info.value.status_code == 400
    assert "exactly one" in exc_info.value.detail.lower()


async def test_create_video_job_route_rejects_when_both_file_and_upload_token(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    video_jobs = make_video_job_manager(settings)
    stage_upload(settings, "sometoken", "clip.mp4")

    with pytest.raises(HTTPException) as exc_info:
        await create_video_job(
            request=None,
            file=make_upload("clip.mp4", b"fake-video-bytes"),
            upload_token="sometoken",
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

    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# create_video_job route - upload_token path skips re-upload, reuses staged file
# ---------------------------------------------------------------------------


async def test_create_video_job_route_creates_job_from_upload_token_without_saving_new_file(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    video_jobs = make_video_job_manager(settings)
    staged = stage_upload(settings, "abc123", "clip.mp4", content=b"already-staged-bytes")

    response = await create_video_job(
        request=None,
        file=None,
        upload_token="abc123",
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
    assert job.source_path == staged
    assert list(settings.uploads_path.glob("*")) == [staged]


async def test_create_video_job_route_derives_original_filename_from_staged_upload(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    video_jobs = make_video_job_manager(settings)
    stage_upload(settings, "abc123", "my-clip.mp4")

    response = await create_video_job(
        request=None,
        file=None,
        upload_token="abc123",
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
    assert job.original_filename == "my-clip.mp4"


async def test_create_video_job_route_upload_token_job_id_is_independent_of_token(
    tmp_path: Path,
) -> None:
    # Guards against a real bug this task had to avoid: if job_id reused the
    # upload_token, two successful jobs created from the same staged upload
    # would collide in video_jobs.jobs (the second overwriting the first).
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    video_jobs = make_video_job_manager(settings)
    stage_upload(settings, "abc123", "clip.mp4")

    response = await create_video_job(
        request=None,
        file=None,
        upload_token="abc123",
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

    assert response.job_id != "abc123"


async def test_create_video_job_route_upload_token_failure_does_not_delete_staged_file(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    video_jobs = make_video_job_manager(settings)
    staged = stage_upload(settings, "abc123", "clip.mp4")

    with pytest.raises(HTTPException) as exc_info:
        await create_video_job(
            request=None,
            file=None,
            upload_token="abc123",
            profile_key="anime-balanced-2x",
            model_name=None,
            scale=None,
            output_container="bogus-container",
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

    assert exc_info.value.status_code == 400
    assert staged.exists()


# ---------------------------------------------------------------------------
# create_video_job route - audio_track_indices CSV parsing + keep_subtitles wiring
# ---------------------------------------------------------------------------


async def test_create_video_job_route_parses_audio_track_indices_csv(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    video_jobs = make_video_job_manager(settings)

    response = await create_video_job(
        request=None,
        file=make_upload("clip.mp4", b"fake-video-bytes"),
        upload_token=None,
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
        audio_track_indices="1,2",
        model_id=None,
        device=None,
        video_jobs=video_jobs,
        storage=storage,
        settings=settings,
        devices=FakeDevicesService(),
    )

    job = video_jobs.get_job(response.job_id)
    assert job is not None
    assert job.audio_track_indices == [1, 2]


async def test_create_video_job_route_defaults_audio_track_indices_to_none_when_omitted(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    video_jobs = make_video_job_manager(settings)

    response = await create_video_job(
        request=None,
        file=make_upload("clip.mp4", b"fake-video-bytes"),
        upload_token=None,
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
    assert job.audio_track_indices is None
    assert job.keep_subtitles is False


async def test_create_video_job_route_passes_keep_subtitles_and_upgrades_container(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    video_jobs = make_video_job_manager(settings)

    response = await create_video_job(
        request=None,
        file=make_upload("clip.mp4", b"fake-video-bytes"),
        upload_token=None,
        profile_key="anime-balanced-2x",
        model_name=None,
        scale=None,
        output_container="mp4",
        video_codec=None,
        video_preset=None,
        crf=None,
        keep_audio=None,
        fps_multiplier=None,
        target_fps=None,
        audio_enhance=None,
        audio_restore=None,
        keep_subtitles=True,
        model_id=None,
        device=None,
        video_jobs=video_jobs,
        storage=storage,
        settings=settings,
        devices=FakeDevicesService(),
    )

    job = video_jobs.get_job(response.job_id)
    assert job is not None
    assert job.keep_subtitles is True
    assert job.output_container == "mkv"
    assert "subtitles" in job.metadata["containerUpgradedReason"]


async def test_create_video_job_route_passes_audio_output_format_and_upgrades_container(
    tmp_path: Path,
) -> None:
    settings = make_settings_with_audio_restore(tmp_path)
    storage = StorageService(settings)
    video_jobs = make_video_job_manager(settings)

    response = await create_video_job(
        request=None,
        file=make_upload("clip.mp4", b"fake-video-bytes"),
        upload_token=None,
        profile_key="anime-balanced-2x",
        model_name=None,
        scale=None,
        output_container="mp4",
        video_codec=None,
        video_preset=None,
        crf=None,
        keep_audio=None,
        fps_multiplier=None,
        target_fps=None,
        audio_enhance=None,
        audio_restore="apollo",
        audio_output_format="auto",
        model_id=None,
        device=None,
        video_jobs=video_jobs,
        storage=storage,
        settings=settings,
        devices=FakeDevicesService(),
    )

    job = video_jobs.get_job(response.job_id)
    assert job is not None
    assert job.audio_output_format == "auto"
    assert job.output_container == "mkv"
    assert "flac" in job.metadata["containerUpgradedReason"].lower()


async def test_create_video_job_route_defaults_audio_output_format_to_auto_when_omitted(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    video_jobs = make_video_job_manager(settings)

    response = await create_video_job(
        request=None,
        file=make_upload("clip.mp4", b"fake-video-bytes"),
        upload_token=None,
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
    assert job.audio_output_format == "auto"


# ---------------------------------------------------------------------------
# video_job_to_response exposes the new fields
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Reviewer finding fix - two jobs created from the SAME upload_token share one
# staged source file (by design, see
# test_create_video_job_route_upload_token_job_id_is_independent_of_token).
# Cleanup-on-completion must not delete that shared file while a sibling job
# still needs it, but must still delete it once the last referencing job
# finishes (no permanent leak).
# ---------------------------------------------------------------------------


async def test_shared_upload_token_source_survives_until_last_referencing_job_finishes(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    manager = make_video_job_manager(settings)
    staged = stage_upload(settings, "shared-token", "clip.mp4")

    await manager.start()
    try:
        job_a = await manager.create_job(**create_job_kwargs(source_path=None, upload_token="shared-token"))
        job_b = await manager.create_job(**create_job_kwargs(source_path=None, upload_token="shared-token"))

        assert job_a.source_path == staged
        assert job_b.source_path == staged
        assert job_a.id != job_b.id

        terminal_statuses = (JobStatus.completed, JobStatus.failed, JobStatus.cancelled)

        for _ in range(300):
            statuses = (manager.get_job(job_a.id).status, manager.get_job(job_b.id).status)
            if statuses[0] in terminal_statuses or statuses[1] in terminal_statuses:
                break
            await asyncio.sleep(0.01)

        first_status, second_status = (
            manager.get_job(job_a.id).status,
            manager.get_job(job_b.id).status,
        )
        assert (first_status in terminal_statuses) != (second_status in terminal_statuses), (
            "expected exactly one sibling job to reach a terminal status first "
            "(capacity=1 for the shared device slot should serialize them)"
        )
        assert staged.exists(), (
            "shared source must survive while the sibling job still references it "
            "(still queued or running)"
        )

        await manager.queue.join()

        assert manager.get_job(job_a.id).status == JobStatus.completed
        assert manager.get_job(job_b.id).status == JobStatus.completed
        assert not staged.exists(), (
            "shared source must be deleted once the last referencing job finishes, "
            "not leaked permanently"
        )
    finally:
        await manager.stop()


def test_video_job_to_response_exposes_audio_track_indices_and_keep_subtitles() -> None:
    job = VideoUpscaleJob(
        source_path=Path("clip.mp4"),
        original_filename="clip.mp4",
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mkv",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=True,
        audio_track_indices=[1, 2],
        keep_subtitles=True,
    )

    response = video_job_to_response(job)

    assert response.audio_track_indices == [1, 2]
    assert response.keep_subtitles is True
    serialized = response.model_dump(by_alias=True)
    assert serialized["audioTrackIndices"] == [1, 2]
    assert serialized["keepSubtitles"] is True


def test_video_job_to_response_exposes_audio_output_format() -> None:
    job = VideoUpscaleJob(
        source_path=Path("clip.mp4"),
        original_filename="clip.mp4",
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mkv",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=True,
        audio_restore="apollo",
        audio_output_format="flac",
    )

    response = video_job_to_response(job)

    assert response.audio_output_format == "flac"
    serialized = response.model_dump(by_alias=True)
    assert serialized["audioOutputFormat"] == "flac"
