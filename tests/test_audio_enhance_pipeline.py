from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings
from app.models import VideoUpscaleJob
from app.services.storage import StorageService
from app.services.video_upscaler import VideoUpscaler

# ---------------------------------------------------------------------------
# Task 20 (6.1c) - Wire audio enhancement into the video pipeline: extract
# WAV(48k) -> AudioEnhancer.run -> mux the enhanced WAV re-encoded to AAC at
# encode time. The off / no-audio-stream paths must stay byte-identical to
# the pre-Task-20 pipeline (test_pipeline_stage_order.py already covers the
# base keep_audio mux; this file focuses on the new audio_enhance branch).
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path) -> Settings:
    return Settings(RUNTIME_DIR=str(tmp_path))


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


def make_video_job_with_audio(source_path: Path, **overrides: object) -> VideoUpscaleJob:
    fields = dict(keep_audio=True)
    fields.update(overrides)
    return make_video_job(source_path, **fields)


class FakeVideoEngine:
    def available(self) -> bool:
        return True


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


class FakeMediaToolsNoAudio:
    def available(self) -> bool:
        return True

    async def ffprobe_json(self, source_path: Path) -> dict:
        return {
            "streams": [{"codec_type": "video", "width": 4, "height": 4, "avg_frame_rate": "30/1"}],
            "format": {"duration": "1.0"},
        }


class FakeAudioEnhancer:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[tuple[Path, Path]] = []

    async def run(self, input_wav: Path, output_wav: Path) -> None:
        self.events.append("enhance_audio")
        self.calls.append((input_wav, output_wav))
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        output_wav.write_bytes(b"fake-enhanced-audio")


class StageTrackingVideoUpscaler(VideoUpscaler):
    """Fakes _run_process so no real ffmpeg/engine binary runs; records stage order + commands."""

    def __init__(self, *args: object, events: list[str], **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.events = events
        self.encode_commands: list[list[str]] = []
        self.extract_audio_commands: list[list[str]] = []

    async def _run_process(self, command: list[str]) -> None:
        if "-fps_mode" in command:
            self.events.append("extract")
            self._write_dummy_frame(command)
        elif "-vn" in command:
            self.events.append("extract_audio")
            self.extract_audio_commands.append(command)
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
    tmp_path: Path,
    events: list[str],
    media_tools: object,
    audio_enhancers: dict[str, object] | None = None,
) -> StageTrackingVideoUpscaler:
    settings = make_settings(tmp_path)
    StorageService(settings)
    return StageTrackingVideoUpscaler(
        settings, FakeVideoEngine(), media_tools, None, audio_enhancers=audio_enhancers, events=events
    )


def write_source(upscaler: StageTrackingVideoUpscaler) -> Path:
    source_path = upscaler.settings.uploads_path / "clip.mp4"
    source_path.write_bytes(b"fake-video-bytes")
    return source_path


# ---------------------------------------------------------------------------
# Stage order + wav extraction args
# ---------------------------------------------------------------------------


async def test_audio_enhance_stage_order_extract_enhance_before_upscale(tmp_path: Path) -> None:
    events: list[str] = []
    audio_enhancer = FakeAudioEnhancer(events)
    upscaler = make_upscaler(tmp_path, events, FakeMediaToolsWithAudio(), {"rnnoise": audio_enhancer})
    job = make_video_job_with_audio(write_source(upscaler), audio_enhance="rnnoise")

    await upscaler.run(job)

    assert events == ["extract", "extract_audio", "enhance_audio", "upscale", "encode"]


async def test_audio_enhancer_receives_wav_extracted_at_48khz(tmp_path: Path) -> None:
    events: list[str] = []
    audio_enhancer = FakeAudioEnhancer(events)
    upscaler = make_upscaler(tmp_path, events, FakeMediaToolsWithAudio(), {"rnnoise": audio_enhancer})
    job = make_video_job_with_audio(write_source(upscaler), audio_enhance="rnnoise")

    await upscaler.run(job)

    extract_command = upscaler.extract_audio_commands[0]
    assert extract_command[extract_command.index("-acodec") + 1] == "pcm_s16le"
    assert extract_command[extract_command.index("-ar") + 1] == "48000"
    assert Path(extract_command[-1]).name == "audio.wav"

    assert len(audio_enhancer.calls) == 1
    input_wav, output_wav = audio_enhancer.calls[0]
    assert input_wav.name == "audio.wav"
    assert output_wav.name == "audio-enhanced.wav"


# ---------------------------------------------------------------------------
# Encode-time mux: enhanced wav re-encoded to AAC
# ---------------------------------------------------------------------------


async def test_audio_enhance_final_mux_uses_enhanced_wav_reencoded_to_aac(tmp_path: Path) -> None:
    events: list[str] = []
    audio_enhancer = FakeAudioEnhancer(events)
    upscaler = make_upscaler(tmp_path, events, FakeMediaToolsWithAudio(), {"rnnoise": audio_enhancer})
    job = make_video_job_with_audio(write_source(upscaler), audio_enhance="rnnoise")

    await upscaler.run(job)

    encode_command = upscaler.encode_commands[0]
    input_indices = [index for index, arg in enumerate(encode_command) if arg == "-i"]
    assert len(input_indices) == 2
    audio_input_path = Path(encode_command[input_indices[1] + 1])
    assert audio_input_path.name == "audio-enhanced.wav"
    assert encode_command[-5:-1] == ["-c:a", "aac", "-b:a", "192k"]

    map_indices = [index for index, arg in enumerate(encode_command) if arg == "-map"]
    mapped_values = [encode_command[index + 1] for index in map_indices]
    assert mapped_values == ["0:v:0", "1:a:0"]


async def test_audio_enhance_metadata_marks_enhanced_true(tmp_path: Path) -> None:
    events: list[str] = []
    audio_enhancer = FakeAudioEnhancer(events)
    upscaler = make_upscaler(tmp_path, events, FakeMediaToolsWithAudio(), {"rnnoise": audio_enhancer})
    job = make_video_job_with_audio(write_source(upscaler), audio_enhance="rnnoise")

    await upscaler.run(job)

    assert job.metadata["audioEnhanced"] is True


# ---------------------------------------------------------------------------
# Off path: byte-identical to the pre-Task-20 pipeline
# ---------------------------------------------------------------------------


async def test_audio_enhance_off_keeps_extraction_and_mux_unchanged(tmp_path: Path) -> None:
    events: list[str] = []
    upscaler = make_upscaler(tmp_path, events, FakeMediaToolsWithAudio())
    job = make_video_job_with_audio(write_source(upscaler))

    await upscaler.run(job)

    assert events == ["extract", "extract_audio", "upscale", "encode"]
    extract_command = upscaler.extract_audio_commands[0]
    assert Path(extract_command[-1]).name == "audio.m4a"
    assert "-acodec" not in extract_command
    assert extract_command[-5:-1] == ["-c:a", "aac", "-b:a", "192k"]

    encode_command = upscaler.encode_commands[0]
    input_indices = [index for index, arg in enumerate(encode_command) if arg == "-i"]
    audio_input_path = Path(encode_command[input_indices[1] + 1])
    assert audio_input_path.name == "audio.m4a"
    assert encode_command[-3:-1] == ["-c:a", "copy"]
    assert "audioEnhanced" not in job.metadata


async def test_audio_enhance_ignored_when_keep_audio_false(tmp_path: Path) -> None:
    events: list[str] = []
    audio_enhancer = FakeAudioEnhancer(events)
    upscaler = make_upscaler(tmp_path, events, FakeMediaToolsWithAudio(), {"rnnoise": audio_enhancer})
    job = make_video_job(write_source(upscaler), keep_audio=False, audio_enhance="rnnoise")

    await upscaler.run(job)

    assert "extract_audio" not in events
    assert "enhance_audio" not in events
    assert "audioEnhanced" not in job.metadata


# ---------------------------------------------------------------------------
# No audio stream: skip enhance cleanly, note in metadata, don't fail
# ---------------------------------------------------------------------------


async def test_audio_enhance_skips_cleanly_without_audio_stream(tmp_path: Path) -> None:
    events: list[str] = []
    audio_enhancer = FakeAudioEnhancer(events)
    upscaler = make_upscaler(tmp_path, events, FakeMediaToolsNoAudio(), {"rnnoise": audio_enhancer})
    job = make_video_job_with_audio(write_source(upscaler), audio_enhance="rnnoise")

    output_path = await upscaler.run(job)

    assert "extract_audio" not in events
    assert "enhance_audio" not in events
    assert job.metadata["audioEnhanced"] == "skipped_no_audio"
    encode_command = upscaler.encode_commands[0]
    assert "-map" not in encode_command
    assert output_path.exists()


# ---------------------------------------------------------------------------
# Mux gate: extraction exiting 0 without a usable file must not feed a
# nonexistent path into the encode command — pre-Task-20 the pipeline
# silently encoded a muted video (audio_path.exists() gate), so failing the
# whole job here would be a behavior regression.
# ---------------------------------------------------------------------------


class SilentAudioExtractionUpscaler(StageTrackingVideoUpscaler):
    """Audio extraction exits 0 but writes no file (exotic-codec edge case)."""

    async def _run_process(self, command: list[str]) -> None:
        if "-vn" in command:
            self.events.append("extract_audio")
            self.extract_audio_commands.append(command)
            return
        await super()._run_process(command)


class EmptyAudioFileUpscaler(StageTrackingVideoUpscaler):
    """Audio extraction exits 0 but leaves a 0-byte file behind."""

    @staticmethod
    def _write_dummy_audio(command: list[str]) -> None:
        audio_path = Path(command[-1])
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"")


async def test_encode_drops_audio_when_extraction_exits_zero_without_file(tmp_path: Path) -> None:
    events: list[str] = []
    settings = make_settings(tmp_path)
    StorageService(settings)
    upscaler = SilentAudioExtractionUpscaler(
        settings, FakeVideoEngine(), FakeMediaToolsWithAudio(), None, None, events=events
    )
    job = make_video_job_with_audio(write_source(upscaler))

    output_path = await upscaler.run(job)

    assert "extract_audio" in events
    encode_command = upscaler.encode_commands[0]
    assert "-map" not in encode_command
    assert "-c:a" not in encode_command
    assert output_path.exists()


async def test_encode_drops_audio_when_extraction_produces_empty_file(tmp_path: Path) -> None:
    events: list[str] = []
    settings = make_settings(tmp_path)
    StorageService(settings)
    upscaler = EmptyAudioFileUpscaler(
        settings, FakeVideoEngine(), FakeMediaToolsWithAudio(), None, None, events=events
    )
    job = make_video_job_with_audio(write_source(upscaler))

    output_path = await upscaler.run(job)

    encode_command = upscaler.encode_commands[0]
    assert "-map" not in encode_command
    assert "-c:a" not in encode_command
    assert output_path.exists()


# ---------------------------------------------------------------------------
# Misconfiguration guard
# ---------------------------------------------------------------------------


async def test_audio_enhance_raises_clear_error_when_engine_not_configured(tmp_path: Path) -> None:
    events: list[str] = []
    upscaler = make_upscaler(tmp_path, events, FakeMediaToolsWithAudio(), {})
    job = make_video_job_with_audio(write_source(upscaler), audio_enhance="deepfilter")

    with pytest.raises(RuntimeError, match="deepfilter"):
        await upscaler.run(job)


# ---------------------------------------------------------------------------
# Work dir cleanup (wav files live in work_dir, already removed via shutil.rmtree)
# ---------------------------------------------------------------------------


async def test_work_dir_removed_after_enhanced_audio_run(tmp_path: Path) -> None:
    events: list[str] = []
    audio_enhancer = FakeAudioEnhancer(events)
    upscaler = make_upscaler(tmp_path, events, FakeMediaToolsWithAudio(), {"rnnoise": audio_enhancer})
    job = make_video_job_with_audio(write_source(upscaler), audio_enhance="rnnoise")

    output_path = await upscaler.run(job)

    work_dir = upscaler.settings.video_work_path / job.id
    assert not work_dir.exists()
    assert output_path.exists()
