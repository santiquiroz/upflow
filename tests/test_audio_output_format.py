from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.models import VideoUpscaleJob
from app.services.storage import StorageService
from app.services.video_upscaler import VideoUpscaler

# ---------------------------------------------------------------------------
# Fase C Task 8 - audio_output_format ("auto"|"flac"|"aac") on VideoUpscaleJob.
# _prepare_processed_audio (the enhance/restore chain from Fase B) returns
# FLAC codec args instead of the always-AAC tail when the resolved format
# wants lossless audio: "auto" defaults to FLAC only while a restore actually
# ran (mirrors VideoJobManager._resolve_output_container's mkv-upgrade rule),
# an explicit "flac" always wants it, an explicit "aac" always forces the
# pre-existing lossy path regardless of restore. This file focuses on the
# VideoUpscaler side; the container-upgrade side lives in
# test_upload_token_and_subtitles.py next to the sibling keep_subtitles tests.
# ---------------------------------------------------------------------------


class _FakeEngine:
    def available(self) -> bool:
        return True


class _FakeMediaTools:
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


class FakeRestorer:
    def __init__(self) -> None:
        self.calls: list[tuple[Path, Path, str]] = []

    async def run(self, input_wav: Path, output_wav: Path, device: str) -> None:
        self.calls.append((input_wav, output_wav, device))
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        output_wav.write_bytes(b"fake-restored-audio")


class _CapturingUpscaler(VideoUpscaler):
    """Fakes _run_process so no real ffmpeg binary runs; records commands and
    writes a dummy file at the command's output path so downstream steps
    (restore) have something to consume. Mirrors test_audio_subtitle_mux.py's
    _CapturingUpscaler, extended to leave a usable file behind."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.commands: list[list[str]] = []

    async def _run_process(self, command: list[str]) -> None:
        self.commands.append(list(command))
        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-audio")


def make_upscaler(tmp_path: Path, restorer: FakeRestorer | None = None) -> _CapturingUpscaler:
    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))
    restorers = {"apollo": restorer} if restorer is not None else None
    return _CapturingUpscaler(settings, _FakeEngine(), _FakeMediaTools(), restorers=restorers)


def make_video_job(source_path: Path, **overrides: object) -> VideoUpscaleJob:
    fields: dict[str, object] = dict(
        original_filename=source_path.name,
        model_name="realesr-animevideov3-x4",
        scale=4,
        output_container="mkv",
        video_codec="libx264",
        video_preset="medium",
        crf=18,
        keep_audio=True,
    )
    fields.update(overrides)
    return VideoUpscaleJob(source_path=source_path, **fields)


# ---------------------------------------------------------------------------
# _prepare_processed_audio - codec args by audio_output_format
# ---------------------------------------------------------------------------


async def test_auto_format_returns_flac_codec_when_restore_active(tmp_path: Path) -> None:
    restorer = FakeRestorer()
    upscaler = make_upscaler(tmp_path, restorer)
    source_path = tmp_path / "source.mkv"
    source_path.write_bytes(b"fake")
    job = make_video_job(source_path, audio_restore="apollo", audio_output_format="auto")

    _, codec_args = await upscaler._prepare_processed_audio(job, tmp_path / "audio.m4a")

    assert codec_args == ["-c:a", "flac"]
    assert len(restorer.calls) == 1


async def test_explicit_flac_format_returns_flac_codec_when_restore_active(tmp_path: Path) -> None:
    restorer = FakeRestorer()
    upscaler = make_upscaler(tmp_path, restorer)
    source_path = tmp_path / "source.mkv"
    source_path.write_bytes(b"fake")
    job = make_video_job(source_path, audio_restore="apollo", audio_output_format="flac")

    _, codec_args = await upscaler._prepare_processed_audio(job, tmp_path / "audio.m4a")

    assert codec_args == ["-c:a", "flac"]


async def test_explicit_aac_format_forces_aac_codec_even_when_restore_active(tmp_path: Path) -> None:
    restorer = FakeRestorer()
    upscaler = make_upscaler(tmp_path, restorer)
    source_path = tmp_path / "source.mkv"
    source_path.write_bytes(b"fake")
    job = make_video_job(source_path, audio_restore="apollo", audio_output_format="aac")

    _, codec_args = await upscaler._prepare_processed_audio(job, tmp_path / "audio.m4a")

    assert codec_args == ["-c:a", "aac", "-b:a", "192k"]


async def test_auto_format_returns_aac_codec_when_no_restore_active(tmp_path: Path) -> None:
    # No audio_restore (and no audio_enhance) -- "auto" keeps the pre-existing
    # AAC behavior; FLAC is reserved for when a restore actually ran on this
    # track, matching VideoJobManager._resolve_output_container's rule.
    upscaler = make_upscaler(tmp_path)
    source_path = tmp_path / "source.mkv"
    source_path.write_bytes(b"fake")
    job = make_video_job(source_path, audio_output_format="auto")

    _, codec_args = await upscaler._prepare_processed_audio(job, tmp_path / "audio.m4a")

    assert codec_args == ["-c:a", "aac", "-b:a", "192k"]


def test_default_audio_output_format_is_auto() -> None:
    job = make_video_job(Path("clip.mp4"))

    assert job.audio_output_format == "auto"


# ---------------------------------------------------------------------------
# Extra-track copy precedence (Fase A Task 3, see test_audio_subtitle_mux.py)
# still holds when the primary codec is FLAC instead of AAC. _extra_audio_copy_args
# is codec-agnostic (it always emits "-c:a:<pos> copy" after the general "-c:a"),
# so this locks that in explicitly for the new FLAC path.
# ---------------------------------------------------------------------------


def _positions_of(cmd: list[str], needle: str) -> list[int]:
    return [i for i, token in enumerate(cmd) if token == needle]


def test_extra_audio_tracks_stay_copy_when_primary_is_flac(tmp_path: Path) -> None:
    upscaler = make_upscaler(tmp_path)
    source_path = tmp_path / "source.mkv"
    job = make_video_job(source_path, audio_track_indices=[1, 2], keep_subtitles=False)
    audio_mux_path = tmp_path / "audio-restored.wav"
    audio_mux_path.write_bytes(b"fake")

    cmd = upscaler._build_encode_command(
        job, tmp_path / "frames-out", "24/1", audio_mux_path, ["-c:a", "flac"], tmp_path / "out.mkv", "libx264"
    )

    # The extra (output audio position 1) is still overridden to copy...
    assert "-c:a:1" in cmd
    assert cmd[cmd.index("-c:a:1") + 1] == "copy"
    # ...and the general flac (primary, output audio 0) is emitted BEFORE the
    # per-position override so ffmpeg's last-match-wins keeps copy for the
    # extra while output audio 0 stays flac.
    general_ca = _positions_of(cmd, "-c:a")
    assert general_ca, "primary must still receive the general -c:a flac"
    assert min(general_ca) < cmd.index("-c:a:1")
    assert cmd[min(general_ca) + 1] == "flac"
    assert "-c:a:2" not in cmd


# ---------------------------------------------------------------------------
# Full pipeline (run()) - confirms nothing downstream of _prepare_processed_audio
# silently overrides the FLAC codec args before the final mux.
# ---------------------------------------------------------------------------


class FakeVideoEngine:
    def available(self) -> bool:
        return True


class StageTrackingVideoUpscaler(VideoUpscaler):
    """Fakes _run_process for the full run() pipeline; records encode commands."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.encode_commands: list[list[str]] = []

    async def _run_process(self, command: list[str]) -> None:
        if "-fps_mode" in command:
            self._write_dummy_frame(command)
        elif "-vn" in command:
            self._write_dummy_audio(command)
        elif command[0] == str(self.settings.engine_binary_path):
            self._write_dummy_upscaled_frame(command)
        elif "-framerate" in command:
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


def make_full_upscaler(tmp_path: Path, restorer: FakeRestorer) -> StageTrackingVideoUpscaler:
    settings = Settings(RUNTIME_DIR=str(tmp_path))
    StorageService(settings)
    return StageTrackingVideoUpscaler(
        settings, FakeVideoEngine(), _FakeMediaTools(), None, restorers={"apollo": restorer}
    )


async def test_full_pipeline_mux_uses_flac_when_restore_active_and_format_auto(tmp_path: Path) -> None:
    restorer = FakeRestorer()
    upscaler = make_full_upscaler(tmp_path, restorer)
    source_path = upscaler.settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")
    job = make_video_job(
        source_path,
        model_name="realesr-animevideov3-x2",
        scale=2,
        audio_restore="apollo",
        audio_output_format="auto",
        output_container="mkv",
    )

    await upscaler.run(job)

    encode_command = upscaler.encode_commands[0]
    assert encode_command[-3:-1] == ["-c:a", "flac"]
    assert job.metadata["audioRestored"] is True


async def test_full_pipeline_mux_uses_aac_when_explicit_format_forces_it(tmp_path: Path) -> None:
    restorer = FakeRestorer()
    upscaler = make_full_upscaler(tmp_path, restorer)
    source_path = upscaler.settings.uploads_path / "clip.mp4"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-video-bytes")
    job = make_video_job(
        source_path,
        model_name="realesr-animevideov3-x2",
        scale=2,
        audio_restore="apollo",
        audio_output_format="aac",
        output_container="mkv",
    )

    await upscaler.run(job)

    encode_command = upscaler.encode_commands[0]
    assert encode_command[-5:-1] == ["-c:a", "aac", "-b:a", "192k"]
