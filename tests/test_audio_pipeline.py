from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import pytest
from starlette.datastructures import UploadFile

from app.api.routes import audio_job_to_response, create_audio_job
from app.config import Settings
from app.models import AudioJob
from app.services.audio_job_manager import AudioJobManager
from app.services.audio_pipeline import AudioPipeline
from app.services.device_semaphores import DeviceSemaphores
from app.services.storage import StorageService

# ---------------------------------------------------------------------------
# Fase C Task 9 - output_format ("wav"|"flac"|"mp3", default "flac") on the
# standalone AudioPipeline. Mirrors Task 8's video-side test shape
# (test_audio_output_format.py) but for the standalone chain: the final
# "finalizing" stage picks the output extension + codec by job.output_format
# instead of always producing a .wav via shutil.move. "wav" keeps the
# pre-existing no-re-encode move (current is already PCM WAV from decode/
# denoise/restore); "flac"/"mp3" re-encode current -> output_path via ffmpeg,
# same _run_process wrapper _decode_to_wav already uses.
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path))


def make_audio_job(source_path: Path, **overrides: object) -> AudioJob:
    fields = dict(
        source_path=source_path,
        original_filename=source_path.name,
        denoise="rnnoise",
    )
    fields.update(overrides)
    return AudioJob(**fields)


class StageTrackingAudioPipeline(AudioPipeline):
    """Fakes _run_process so no real ffmpeg/engine binary runs; records commands."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.commands: list[list[str]] = []

    async def _run_process(self, command: list[str], failure_message: str) -> None:
        self.commands.append(command)
        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-audio-bytes")


class FakeAudioEnhancer:
    async def run(self, input_wav: Path, output_wav: Path) -> None:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        output_wav.write_bytes(b"fake-denoised-audio")


def make_pipeline(tmp_path: Path) -> StageTrackingAudioPipeline:
    settings = make_settings(tmp_path)
    return StageTrackingAudioPipeline(settings, {"rnnoise": FakeAudioEnhancer()}, {})


def write_source(pipeline: StageTrackingAudioPipeline) -> Path:
    source_path = pipeline.settings.uploads_path / "clip.wav"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-source-bytes")
    return source_path


# ---------------------------------------------------------------------------
# Model default
# ---------------------------------------------------------------------------


def test_audio_job_output_format_defaults_to_flac(tmp_path: Path) -> None:
    job = make_audio_job(tmp_path / "clip.wav")

    assert job.output_format == "flac"


# ---------------------------------------------------------------------------
# Pipeline: extension + codec per format
# ---------------------------------------------------------------------------


async def test_output_format_flac_produces_flac_extension_and_codec(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path)
    job = make_audio_job(write_source(pipeline), output_format="flac")

    output_path = await pipeline.run(job)

    assert output_path.suffix == ".flac"
    encode_command = pipeline.commands[-1]
    assert encode_command[-3:-1] == ["-c:a", "flac"]
    assert encode_command[-1] == str(output_path)


async def test_output_format_wav_produces_wav_extension_without_reencode(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path)
    job = make_audio_job(write_source(pipeline), output_format="wav")

    output_path = await pipeline.run(job)

    assert output_path.suffix == ".wav"
    # wav is a move of the already-PCM current file, not a re-encode: denoise
    # runs through the injected FakeAudioEnhancer (not _run_process), so only
    # the decode step ever calls _run_process -- no extra ffmpeg call for output.
    assert len(pipeline.commands) == 1
    assert output_path.exists()


async def test_output_format_mp3_produces_mp3_extension_and_codec(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path)
    job = make_audio_job(write_source(pipeline), output_format="mp3")

    output_path = await pipeline.run(job)

    assert output_path.suffix == ".mp3"
    encode_command = pipeline.commands[-1]
    assert encode_command[-5:-1] == ["-c:a", "libmp3lame", "-b:a", "192k"]


async def test_default_output_format_is_flac_end_to_end(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path)
    job = make_audio_job(write_source(pipeline))  # no output_format override

    output_path = await pipeline.run(job)

    assert job.output_format == "flac"
    assert output_path.suffix == ".flac"


async def test_output_encode_reads_from_the_final_processed_wav(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path)
    job = make_audio_job(write_source(pipeline), output_format="flac", denoise="rnnoise", restore=None)

    await pipeline.run(job)

    encode_command = pipeline.commands[-1]
    input_index = encode_command.index("-i")
    assert Path(encode_command[input_index + 1]).name == "denoised.wav"


async def test_work_dir_removed_after_flac_output(tmp_path: Path) -> None:
    pipeline = make_pipeline(tmp_path)
    job = make_audio_job(write_source(pipeline), output_format="flac")

    output_path = await pipeline.run(job)

    work_dir = pipeline.settings.temp_path / f"audio-{job.id}"
    assert not work_dir.exists()
    assert output_path.exists()


# ---------------------------------------------------------------------------
# AudioJobManager.create_job - output_format validation + pass-through
# ---------------------------------------------------------------------------


def make_manager(settings: Settings) -> AudioJobManager:
    pipeline = AudioPipeline(settings, {}, {})
    return AudioJobManager(settings, pipeline, DeviceSemaphores(settings))


def write_upload_source(settings: Settings, name: str = "clip.wav") -> Path:
    source_path = settings.uploads_path / name
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-source-bytes")
    return source_path


async def test_audio_job_manager_defaults_output_format_to_flac(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)

    job = await manager.create_job(
        source_path=write_upload_source(settings),
        original_filename="clip.wav",
        denoise="rnnoise",
    )

    assert job.output_format == "flac"


async def test_audio_job_manager_accepts_requested_output_format(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)

    job = await manager.create_job(
        source_path=write_upload_source(settings),
        original_filename="clip.wav",
        denoise="rnnoise",
        output_format="mp3",
    )

    assert job.output_format == "mp3"


async def test_audio_job_manager_rejects_unknown_output_format(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)

    with pytest.raises(ValueError, match="output_format"):
        await manager.create_job(
            source_path=write_upload_source(settings),
            original_filename="clip.wav",
            denoise="rnnoise",
            output_format="ogg",
        )


# ---------------------------------------------------------------------------
# Route: create_audio_job forwards output_format, response exposes it
# ---------------------------------------------------------------------------


def make_upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=io.BytesIO(content), filename=filename)


async def test_create_audio_job_route_forwards_output_format(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    storage = StorageService(settings)
    manager = make_manager(settings)

    response = await create_audio_job(
        request=None,
        file=make_upload("clip.wav", b"fake-audio-bytes"),
        denoise="rnnoise",
        restore=None,
        device=None,
        output_format="mp3",
        audio_jobs=manager,
        storage=storage,
        settings=settings,
    )

    job = manager.get_job(response.job_id)
    assert job is not None
    assert job.output_format == "mp3"


def test_audio_job_to_response_exposes_output_format(tmp_path: Path) -> None:
    job = make_audio_job(tmp_path / "clip.wav", output_format="mp3")

    response = audio_job_to_response(job)

    assert response.output_format == "mp3"


def test_audio_job_response_serializes_output_format_camel_case(tmp_path: Path) -> None:
    job = make_audio_job(tmp_path / "clip.wav", output_format="mp3")

    serialized = audio_job_to_response(job).model_dump(by_alias=True)

    assert serialized["outputFormat"] == "mp3"


def test_audio_job_to_response_exposes_timestamps(tmp_path: Path) -> None:
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    finished = datetime(2026, 1, 1, 0, 3, 12, tzinfo=timezone.utc)
    job = make_audio_job(tmp_path / "clip.wav", started_at=started, finished_at=finished)

    response = audio_job_to_response(job)

    assert response.started_at == started
    assert response.finished_at == finished


def test_audio_job_response_serializes_timestamps_camel_case(tmp_path: Path) -> None:
    job = make_audio_job(tmp_path / "clip.wav")

    serialized = audio_job_to_response(job).model_dump(by_alias=True)

    assert "createdAt" in serialized
    assert "startedAt" in serialized
    assert "finishedAt" in serialized
