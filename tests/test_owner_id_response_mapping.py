from __future__ import annotations

from pathlib import Path

from app.api.routes import (
    audio_job_to_response,
    generation_job_to_response,
    job_to_response,
    video_job_to_response,
)
from app.models import AudioJob, GenerationJob, UpscaleJob, VideoUpscaleJob


def test_job_to_response_includes_owner_id(tmp_path: Path) -> None:
    job = UpscaleJob(
        source_path=tmp_path / "source.png",
        original_filename="source.png",
        model_name="realesrgan-x4plus",
        scale=4,
        output_format="png",
        owner_id="u1",
    )

    response = job_to_response(job)

    assert response.owner_id == "u1"


def test_video_job_to_response_includes_owner_id(tmp_path: Path) -> None:
    job = VideoUpscaleJob(
        source_path=tmp_path / "source.mp4",
        original_filename="source.mp4",
        model_name="realesr-animevideov3-x2",
        scale=2,
        output_container="mp4",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=False,
        owner_id="u1",
    )

    response = video_job_to_response(job)

    assert response.owner_id == "u1"


def test_audio_job_to_response_includes_owner_id(tmp_path: Path) -> None:
    job = AudioJob(
        source_path=tmp_path / "source.wav",
        original_filename="source.wav",
        owner_id="u1",
    )

    response = audio_job_to_response(job)

    assert response.owner_id == "u1"


def test_generation_job_to_response_includes_owner_id() -> None:
    job = GenerationJob(
        prompt="a cat",
        model_id="sd15",
        owner_id="u1",
    )

    response = generation_job_to_response(job)

    assert response.owner_id == "u1"
