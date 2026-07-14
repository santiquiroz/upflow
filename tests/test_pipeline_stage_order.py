from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.models import VideoUpscaleJob
from app.services.media_tools import compute_target_frame_count
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
        self.calls: list[tuple[Path, Path, int, int, int | None]] = []

    async def run(
        self,
        frames_in: Path,
        frames_out: Path,
        source_frame_count: int,
        multiplier: int = 1,
        *,
        target_frame_count: int | None = None,
    ) -> Path:
        self.events.append("interpolate")
        self.calls.append((frames_in, frames_out, source_frame_count, multiplier, target_frame_count))
        resolved_count = target_frame_count if target_frame_count is not None else source_frame_count * multiplier
        frames_out.mkdir(parents=True, exist_ok=True)
        for index in range(resolved_count):
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
        elif command[0] == str(self.settings.engine_binary_path):
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
    frames_in_arg, frames_out_arg, source_frame_count, multiplier, target_frame_count = rife_engine.calls[0]
    assert frames_in_arg.name == "frames-out"
    assert frames_out_arg.name == "frames-interp"
    assert source_frame_count == 1
    assert multiplier == 3
    assert target_frame_count is None


async def test_work_dir_removed_after_interpolated_run(tmp_path: Path) -> None:
    events: list[str] = []
    upscaler = make_upscaler(tmp_path, events, FakeRifeEngine(events))
    job = make_video_job(write_source(upscaler))

    output_path = await upscaler.run(job, fps_multiplier=2)

    work_dir = upscaler.settings.video_work_path / job.id
    assert not work_dir.exists()
    assert output_path.exists()


async def test_extract_and_encode_commands_use_resolved_absolute_ffmpeg_path(tmp_path: Path) -> None:
    """Task 16 review fix: raw settings.ffmpeg_binary (a CWD-relative forward-slash
    string) crashes asyncio.create_subprocess_exec on Windows with WinError 2.
    Every ffmpeg invocation must use the resolved absolute ffmpeg_binary_path.
    """
    events: list[str] = []
    upscaler = make_upscaler(tmp_path, events, FakeRifeEngine(events))
    job = make_video_job(write_source(upscaler))

    await upscaler.run(job, fps_multiplier=2)

    expected_ffmpeg = str(upscaler.settings.ffmpeg_binary_path)
    assert expected_ffmpeg != upscaler.settings.ffmpeg_binary
    encode_command = upscaler.encode_commands[0]
    assert encode_command[0] == expected_ffmpeg


async def test_upscale_command_uses_resolved_absolute_engine_path(tmp_path: Path) -> None:
    events: list[str] = []
    rife_engine = FakeRifeEngine(events)
    upscaler = make_upscaler(tmp_path, events, rife_engine)
    job = make_video_job(write_source(upscaler))

    captured: list[list[str]] = []
    original_run_process = upscaler._run_process

    async def capturing_run_process(command: list[str]) -> None:
        captured.append(command)
        await original_run_process(command)

    upscaler._run_process = capturing_run_process  # type: ignore[method-assign]

    await upscaler.run(job, fps_multiplier=2)

    expected_engine = str(upscaler.settings.engine_binary_path)
    assert expected_engine != upscaler.settings.engine_binary
    upscale_commands = [command for command in captured if command[0] == expected_engine]
    assert len(upscale_commands) == 1

    upscale_command = upscale_commands[0]
    models_arg = upscale_command[upscale_command.index("-m") + 1]
    assert models_arg == str(upscaler.settings.engine_models_path)
    assert models_arg != upscaler.settings.engine_models_dir


async def test_run_raises_clear_error_when_multiplier_enabled_without_rife_engine(tmp_path: Path) -> None:
    events: list[str] = []
    upscaler = make_upscaler(tmp_path, events, None)
    job = make_video_job(write_source(upscaler))

    with pytest.raises(RuntimeError, match="RIFE"):
        await upscaler.run(job, fps_multiplier=2)


# ---------------------------------------------------------------------------
# Task 13 - keep_audio=True combined with fps_multiplier=2: audio mux args
# must stay unchanged while the encode framerate doubles (closes a Task 12
# review gap; the earlier stage-order tests never exercised keep_audio=True
# together with interpolation in the same run).
# ---------------------------------------------------------------------------


class FakeMediaToolsWithAudio:
    def available(self) -> bool:
        return True

    async def ffprobe_json(self, source_path: Path) -> dict:
        return {
            "streams": [
                {"codec_type": "video", "width": 4, "height": 4, "avg_frame_rate": "30/1"},
                {"codec_type": "audio"},
            ],
            "format": {"duration": "1.0"},
        }


def make_video_job_with_audio(source_path: Path) -> VideoUpscaleJob:
    return VideoUpscaleJob(
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


def make_upscaler_with_audio(
    tmp_path: Path, events: list[str], rife_engine: FakeRifeEngine | None
) -> StageTrackingVideoUpscaler:
    settings = make_settings(tmp_path)
    StorageService(settings)
    return StageTrackingVideoUpscaler(
        settings, FakeVideoEngine(), FakeMediaToolsWithAudio(), rife_engine, events=events
    )


async def test_keep_audio_and_fps_multiplier_combo_preserves_audio_mux_and_doubles_framerate(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    upscaler = make_upscaler_with_audio(tmp_path, events, FakeRifeEngine(events))
    job = make_video_job_with_audio(write_source(upscaler))

    await upscaler.run(job, fps_multiplier=2)

    assert events == ["extract", "extract_audio", "upscale", "interpolate", "encode"]

    encode_command = upscaler.encode_commands[0]

    framerate = encode_command[encode_command.index("-framerate") + 1]
    assert framerate == "60/1"

    map_indices = [index for index, arg in enumerate(encode_command) if arg == "-map"]
    mapped_values = [encode_command[index + 1] for index in map_indices]
    assert mapped_values == ["0:v:0", "1:a:0"], "audio mux mapping must stay unchanged by interpolation"
    assert encode_command[-3:-1] == ["-c:a", "copy"], "audio codec copy flag must stay unchanged"


# ---------------------------------------------------------------------------
# Task 13 review fix - job.metadata["outputFps"] coverage: populated with the
# multiplied fps for interpolated runs and the original fps otherwise.
# ---------------------------------------------------------------------------


async def test_output_fps_metadata_reflects_multiplier_on_interpolated_run(tmp_path: Path) -> None:
    events: list[str] = []
    upscaler = make_upscaler(tmp_path, events, FakeRifeEngine(events))
    job = make_video_job(write_source(upscaler))

    await upscaler.run(job, fps_multiplier=2)

    assert job.metadata["outputFps"] == "60/1"


async def test_output_fps_metadata_keeps_original_fps_when_multiplier_is_one(tmp_path: Path) -> None:
    events: list[str] = []
    upscaler = make_upscaler(tmp_path, events, FakeRifeEngine(events))
    job = make_video_job(write_source(upscaler))

    await upscaler.run(job)

    assert job.metadata["outputFps"] == "30"


# ---------------------------------------------------------------------------
# Task 15 (6.6) - TARGET_FPS mode: job.target_fps drives interpolation to an
# absolute frame count instead of a multiplier. FakeMediaTools in this file
# reports avg_frame_rate "30/1" as the source.
# ---------------------------------------------------------------------------


def make_video_job_with_target_fps(source_path: Path, target_fps: str) -> VideoUpscaleJob:
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
        target_fps=target_fps,
    )


async def test_target_fps_interpolation_runs_after_upscale_and_before_encode(tmp_path: Path) -> None:
    events: list[str] = []
    upscaler = make_upscaler(tmp_path, events, FakeRifeEngine(events))
    job = make_video_job_with_target_fps(write_source(upscaler), "60")

    await upscaler.run(job)

    assert events == ["extract", "upscale", "interpolate", "encode"]


async def test_target_fps_encode_reads_from_frames_interp(tmp_path: Path) -> None:
    events: list[str] = []
    upscaler = make_upscaler(tmp_path, events, FakeRifeEngine(events))
    job = make_video_job_with_target_fps(write_source(upscaler), "60")

    await upscaler.run(job)

    encode_command = upscaler.encode_commands[0]
    frames_arg = encode_command[encode_command.index("-i") + 1]
    assert "frames-interp" in frames_arg


async def test_target_fps_encode_framerate_is_normalized_target(tmp_path: Path) -> None:
    events: list[str] = []
    upscaler = make_upscaler(tmp_path, events, FakeRifeEngine(events))
    job = make_video_job_with_target_fps(write_source(upscaler), "60")

    await upscaler.run(job)

    encode_command = upscaler.encode_commands[0]
    framerate = encode_command[encode_command.index("-framerate") + 1]
    assert framerate == "60/1"


async def test_target_fps_output_fps_metadata_is_normalized_target(tmp_path: Path) -> None:
    events: list[str] = []
    upscaler = make_upscaler(tmp_path, events, FakeRifeEngine(events))
    job = make_video_job_with_target_fps(write_source(upscaler), "60")

    await upscaler.run(job)

    assert job.metadata["outputFps"] == "60/1"


async def test_rife_engine_receives_absolute_target_frame_count(tmp_path: Path) -> None:
    events: list[str] = []
    rife_engine = FakeRifeEngine(events)
    upscaler = make_upscaler(tmp_path, events, rife_engine)
    job = make_video_job_with_target_fps(write_source(upscaler), "60")

    await upscaler.run(job)

    assert len(rife_engine.calls) == 1
    _, _, source_frame_count, _, target_frame_count = rife_engine.calls[0]
    assert target_frame_count == compute_target_frame_count(source_frame_count, "30/1", "60")


async def test_run_raises_clear_error_when_target_fps_set_without_rife_engine(tmp_path: Path) -> None:
    events: list[str] = []
    upscaler = make_upscaler(tmp_path, events, None)
    job = make_video_job_with_target_fps(write_source(upscaler), "60")

    with pytest.raises(RuntimeError, match="RIFE"):
        await upscaler.run(job)
