from __future__ import annotations

import asyncio
import io
from fractions import Fraction
from pathlib import Path

import pytest
from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import UploadFile

from app.api.routes import create_video_job, resolve_video_job_fields
from app.config import Settings
from app.models import UpscaleJob, VideoUpscaleJob
from app.services.job_manager import JobManager
from app.services.engines import realesrgan_ncnn as realesrgan_module
from app.services.engines.realesrgan_ncnn import RealEsrganNcnnEngine
from app.services.media_tools import parse_fps_fraction, resolve_video_fps
from app.services.storage import StorageService
from app.services.video_job_manager import VideoJobManager
from app.services.video_upscaler import VideoUpscaler


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path))


def make_upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


def make_png_bytes(color: str = "red") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeUpscaler:
    async def run(self, job: VideoUpscaleJob) -> Path:
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
# 3.1 — crf / scale `or`-default drops explicit 0
# ---------------------------------------------------------------------------


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
    }


def test_resolve_video_job_fields_keeps_explicit_crf_zero() -> None:
    resolved = resolve_video_job_fields(
        make_profile(),
        model_name=None,
        scale=None,
        output_container=None,
        video_codec=None,
        video_preset=None,
        crf=0,
        keep_audio=None,
    )

    assert resolved.crf == 0


def test_resolve_video_job_fields_keeps_explicit_scale_zero() -> None:
    resolved = resolve_video_job_fields(
        make_profile(),
        model_name=None,
        scale=0,
        output_container=None,
        video_codec=None,
        video_preset=None,
        crf=None,
        keep_audio=None,
    )

    assert resolved.scale == 0


def test_resolve_video_job_fields_falls_back_to_profile_when_omitted() -> None:
    resolved = resolve_video_job_fields(
        make_profile(),
        model_name=None,
        scale=None,
        output_container=None,
        video_codec=None,
        video_preset=None,
        crf=None,
        keep_audio=None,
    )

    assert resolved.crf == 17
    assert resolved.scale == 2


async def test_explicit_scale_zero_is_rejected_by_downstream_validation_not_silently_dropped(
    tmp_path: Path,
) -> None:
    """Proves the explicit 0 actually reaches job validation instead of being replaced
    by the profile's valid scale=2 (which would have succeeded silently)."""
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    video_jobs = VideoJobManager(
        settings, FakeUpscaler(), FakeMediaTools(), asyncio.Semaphore(settings.gpu_concurrency)
    )

    with pytest.raises(HTTPException) as exc_info:
        await create_video_job(
            request=None,
            file=make_upload("clip.mp4", b"fake-video-bytes"),
            profile_key="anime-balanced-2x",
            model_name=None,
            scale=0,
            output_container=None,
            video_codec=None,
            video_preset=None,
            crf=None,
            keep_audio=None,
            model_id=None,
            device=None,
            video_jobs=video_jobs,
            storage=storage,
            settings=settings,
            devices=FakeDevicesService(),
        )

    assert exc_info.value.status_code == 400
    assert "supports only scales" in exc_info.value.detail


# ---------------------------------------------------------------------------
# 3.2 — fps fallback accepts ffprobe "0/1"
# ---------------------------------------------------------------------------


def test_parse_fps_fraction_rejects_zero_numerator() -> None:
    assert parse_fps_fraction("0/1") is None


def test_parse_fps_fraction_rejects_zero_over_zero() -> None:
    assert parse_fps_fraction("0/0") is None


def test_parse_fps_fraction_rejects_empty_string() -> None:
    assert parse_fps_fraction("") is None


def test_parse_fps_fraction_rejects_none() -> None:
    assert parse_fps_fraction(None) is None


def test_parse_fps_fraction_accepts_valid_rate() -> None:
    assert parse_fps_fraction("24000/1001") == Fraction(24000, 1001)


def test_resolve_video_fps_falls_through_avg_to_r_frame_rate() -> None:
    assert resolve_video_fps("0/1", "25/1") == Fraction(25, 1)


def test_resolve_video_fps_falls_through_both_invalid_to_default() -> None:
    assert resolve_video_fps("0/0", "") == Fraction(30, 1)


def test_resolve_video_fps_prefers_valid_avg_frame_rate() -> None:
    assert resolve_video_fps("24/1", "30/1") == Fraction(24, 1)


# ---------------------------------------------------------------------------
# 3.3 — output "exists" but 0 bytes marked completed
# ---------------------------------------------------------------------------


class ZeroByteOutputVideoUpscaler(VideoUpscaler):
    """Fakes the pipeline but writes a 0-byte encoded output file."""

    async def _run_process(self, command: list[str]) -> None:
        if "-vsync" in command:
            frames_in_dir = Path(command[-1]).parent
            frames_in_dir.mkdir(parents=True, exist_ok=True)
            (frames_in_dir / "00000001.png").write_bytes(b"fake-frame-in")
        elif command[0] == str(self.settings.engine_binary_path):
            frames_out_dir = Path(command[command.index("-o") + 1])
            frames_out_dir.mkdir(parents=True, exist_ok=True)
            (frames_out_dir / "00000001.png").write_bytes(b"fake-frame-out")
        elif "-framerate" in command:
            output_path = Path(command[-1])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"")


class ZeroByteMediaTools:
    def available(self) -> bool:
        return True

    async def ffprobe_json(self, source_path: Path) -> dict:
        return {
            "streams": [{"codec_type": "video", "width": 4, "height": 4, "avg_frame_rate": "30/1"}],
            "format": {"duration": "1.0"},
        }


class ZeroByteEngine:
    def available(self) -> bool:
        return True


def make_video_job(source_path: Path) -> VideoUpscaleJob:
    return VideoUpscaleJob(
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


async def test_zero_byte_video_output_fails_the_job(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    StorageService(settings)
    upscaler = ZeroByteOutputVideoUpscaler(settings, ZeroByteEngine(), ZeroByteMediaTools())

    source_path = settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")
    job = make_video_job(source_path)

    with pytest.raises(RuntimeError, match="no output file was produced"):
        await upscaler.run(job)


async def test_zero_byte_image_engine_output_fails_the_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path)

    async def fake_run_guarded_process(command: list[str], timeout: float):
        output_path = Path(command[command.index("-o") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"")
        return b"", b"", 0

    monkeypatch.setattr(realesrgan_module, "run_guarded_process", fake_run_guarded_process)

    engine = RealEsrganNcnnEngine(settings)
    source_path = settings.uploads_path / "photo.png"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(make_png_bytes())

    upscale_job = UpscaleJob(
        source_path=source_path,
        original_filename="photo.png",
        model_name="realesrgan-x4plus",
        scale=4,
        output_format="png",
    )

    with pytest.raises(RuntimeError, match="no output file was produced"):
        await engine.run(upscale_job)


# ---------------------------------------------------------------------------
# 3.4 — no image format allow-list
# ---------------------------------------------------------------------------


def make_gif_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color="red").save(buffer, format="GIF")
    return buffer.getvalue()


def test_validate_input_image_rejects_non_whitelisted_format(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    jobs = JobManager(settings, object(), asyncio.Semaphore(settings.gpu_concurrency))
    source_path = tmp_path / "sneaky.gif"
    source_path.write_bytes(make_gif_bytes())

    with pytest.raises(ValueError, match="(?i)format"):
        jobs._validate_input_image(source_path)


def test_validate_input_image_accepts_png(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    jobs = JobManager(settings, object(), asyncio.Semaphore(settings.gpu_concurrency))
    source_path = tmp_path / "ok.png"
    source_path.write_bytes(make_png_bytes())

    jobs._validate_input_image(source_path)


# ---------------------------------------------------------------------------
# 3.10 — mislabeled profile general-balanced-2x actually uses scale=4
# ---------------------------------------------------------------------------


def test_general_profile_label_matches_its_configured_scale() -> None:
    settings = make_settings(Path("."))
    for profile in settings.video_profile_catalog:
        if profile["category"] != "general":
            continue
        assert f"{profile['scale']}x" in profile["key"]
        assert f"{profile['scale']}x" in profile["label"]


def test_general_balanced_2x_key_no_longer_exists() -> None:
    settings = make_settings(Path("."))
    assert "general-balanced-2x" not in settings.video_profile_keys
