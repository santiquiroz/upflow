from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.models import VideoUpscaleJob
from app.services.storage import StorageService
from app.services.video_upscaler import VideoUpscaler

# ---------------------------------------------------------------------------
# Task 12 (4.4) - Wire RIFE interpolation into the video pipeline: stage
# ordering (extract -> upscale -> interpolate -> encode) and encode source
# selection (frames-interp when enabled, frames-out when off).
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path))


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


class FakeVideoEngine:
    def available(self) -> bool:
        return True


class FakeMediaTools:
    def available(self) -> bool:
        return True

    async def ffprobe_json(self, source_path: Path) -> dict:
        return {
            "streams": [{"codec_type": "video", "width": 4, "height": 4, "avg_frame_rate": "30/1"}],
            "format": {"duration": "1.0"},
        }


class FakeRifeEngine:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[tuple[Path, Path, int, int]] = []

    async def run(self, frames_in: Path, frames_out: Path, source_frame_count: int, multiplier: int) -> Path:
        self.events.append("interpolate")
        self.calls.append((frames_in, frames_out, source_frame_count, multiplier))
        frames_out.mkdir(parents=True, exist_ok=True)
        for index in range(source_frame_count * multiplier):
            (frames_out / f"{index:08d}.png").write_bytes(b"fake-interp-frame")
        return frames_out


class StageTrackingVideoUpscaler(VideoUpscaler):
    """Fakes _run_process so no real ffmpeg/engine binary runs; records stage order + encode command."""

    def __init__(self, *args: object, events: list[str], **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.events = events
        self.encode_commands: list[list[str]] = []

    async def _run_process(self, command: list[str]) -> None:
        if "-vsync" in command:
            self.events.append("extract")
            self._write_dummy_frame(command)
        elif "-vn" in command:
            self.events.append("extract_audio")
            self._write_dummy_audio(command)
        elif command[0] == self.settings.engine_binary:
            self.events.append("upscale")
            self._write_dummy_upscaled_frame(command)
        elif "-framerate" in command:
            self.events.append("encode")
            self.encode_commands.append(command)
            self._write_dummy_output(command)

    @staticmethod
    def _write_dummy_frame(command: list[str]) -> None:
        frames_in_dir = Path(command[-1]).parent
        frames_in_dir.mkdir(parents=True, exist_ok=True)
        (frames_in_dir / "00000001.png").write_bytes(b"fake-frame-in")

    @staticmethod
    def _write_dummy_audio(command: list[str]) -> None:
        audio_path = Path(command[-1])
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"fake-audio")

    @staticmethod
    def _write_dummy_upscaled_frame(command: list[str]) -> None:
        frames_out_dir = Path(command[command.index("-o") + 1])
        frames_out_dir.mkdir(parents=True, exist_ok=True)
        (frames_out_dir / "00000001.png").write_bytes(b"fake-frame-out")

    @staticmethod
    def _write_dummy_output(command: list[str]) -> None:
        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-output-video")


def make_upscaler(
    tmp_path: Path, events: list[str], rife_engine: FakeRifeEngine | None
) -> StageTrackingVideoUpscaler:
    settings = make_settings(tmp_path)
    StorageService(settings)
    return StageTrackingVideoUpscaler(settings, FakeVideoEngine(), FakeMediaTools(), rife_engine, events=events)


def write_source(upscaler: StageTrackingVideoUpscaler) -> Path:
    source_path = upscaler.settings.uploads_path / "clip.mp4"
    source_path.write_bytes(b"fake-video-bytes")
    return source_path


async def test_interpolation_runs_after_upscale_and_before_encode(tmp_path: Path) -> None:
    events: list[str] = []
    upscaler = make_upscaler(tmp_path, events, FakeRifeEngine(events))
    job = make_video_job(write_source(upscaler))

    await upscaler.run(job, fps_multiplier=2)

    assert events == ["extract", "upscale", "interpolate", "encode"]


async def test_no_interpolation_when_multiplier_is_one(tmp_path: Path) -> None:
    events: list[str] = []
    upscaler = make_upscaler(tmp_path, events, FakeRifeEngine(events))
    job = make_video_job(write_source(upscaler))

    await upscaler.run(job)

    assert events == ["extract", "upscale", "encode"]
    assert "interpolate" not in events


async def test_encode_reads_from_frames_interp_when_multiplier_enabled(tmp_path: Path) -> None:
    events: list[str] = []
    upscaler = make_upscaler(tmp_path, events, FakeRifeEngine(events))
    job = make_video_job(write_source(upscaler))

    await upscaler.run(job, fps_multiplier=2)

    encode_command = upscaler.encode_commands[0]
    frames_arg = encode_command[encode_command.index("-i") + 1]
    assert "frames-interp" in frames_arg


async def test_encode_reads_from_frames_out_when_multiplier_is_one(tmp_path: Path) -> None:
    events: list[str] = []
    upscaler = make_upscaler(tmp_path, events, FakeRifeEngine(events))
    job = make_video_job(write_source(upscaler))

    await upscaler.run(job)

    encode_command = upscaler.encode_commands[0]
    frames_arg = encode_command[encode_command.index("-i") + 1]
    assert "frames-out" in frames_arg
    assert "frames-interp" not in frames_arg


async def test_encode_framerate_reflects_multiplier(tmp_path: Path) -> None:
    events: list[str] = []
    upscaler = make_upscaler(tmp_path, events, FakeRifeEngine(events))
    job = make_video_job(write_source(upscaler))

    await upscaler.run(job, fps_multiplier=2)

    encode_command = upscaler.encode_commands[0]
    framerate = encode_command[encode_command.index("-framerate") + 1]
    assert framerate == "60/1"


async def test_encode_framerate_unchanged_when_multiplier_is_one(tmp_path: Path) -> None:
    events: list[str] = []
    upscaler = make_upscaler(tmp_path, events, FakeRifeEngine(events))
    job = make_video_job(write_source(upscaler))

    await upscaler.run(job)

    encode_command = upscaler.encode_commands[0]
    framerate = encode_command[encode_command.index("-framerate") + 1]
    assert framerate == "30"


async def test_rife_engine_receives_upscaled_frame_count_and_multiplier(tmp_path: Path) -> None:
    events: list[str] = []
    rife_engine = FakeRifeEngine(events)
    upscaler = make_upscaler(tmp_path, events, rife_engine)
    job = make_video_job(write_source(upscaler))

    await upscaler.run(job, fps_multiplier=3)

    assert len(rife_engine.calls) == 1
    frames_in_arg, frames_out_arg, source_frame_count, multiplier = rife_engine.calls[0]
    assert frames_in_arg.name == "frames-out"
    assert frames_out_arg.name == "frames-interp"
    assert source_frame_count == 1
    assert multiplier == 3


async def test_work_dir_removed_after_interpolated_run(tmp_path: Path) -> None:
    events: list[str] = []
    upscaler = make_upscaler(tmp_path, events, FakeRifeEngine(events))
    job = make_video_job(write_source(upscaler))

    output_path = await upscaler.run(job, fps_multiplier=2)

    work_dir = upscaler.settings.video_work_path / job.id
    assert not work_dir.exists()
    assert output_path.exists()


async def test_run_raises_clear_error_when_multiplier_enabled_without_rife_engine(tmp_path: Path) -> None:
    events: list[str] = []
    upscaler = make_upscaler(tmp_path, events, None)
    job = make_video_job(write_source(upscaler))

    with pytest.raises(RuntimeError, match="RIFE"):
        await upscaler.run(job, fps_multiplier=2)
