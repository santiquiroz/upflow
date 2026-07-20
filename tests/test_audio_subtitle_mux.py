from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.models import VideoUpscaleJob
from app.services.video_upscaler import VideoUpscaler

# ---------------------------------------------------------------------------
# Fase A Task 3 - _build_encode_command maps EXTRA audio tracks and subtitles
# straight from the ORIGINAL source file into the final mux. The PRIMARY
# audio track (enhanced/restored, or a plain copy) keeps coming from
# audio_mux_path unchanged; this only adds streams the pipeline would
# otherwise silently drop (job.audio_track_indices / job.keep_subtitles from
# Fase A Task 2).
# ---------------------------------------------------------------------------


class _FakeEngine:
    def available(self) -> bool:
        return True


class _FakeMediaTools:
    def available(self) -> bool:
        return True


def make_upscaler(tmp_path: Path) -> VideoUpscaler:
    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))
    return VideoUpscaler(settings, _FakeEngine(), _FakeMediaTools())


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


def make_audio_mux(tmp_path: Path) -> Path:
    audio_mux_path = tmp_path / "audio.m4a"
    audio_mux_path.write_bytes(b"fake")
    return audio_mux_path


# ---------------------------------------------------------------------------
# Extra audio tracks
# ---------------------------------------------------------------------------


def test_maps_extra_audio_track_by_absolute_stream_index(tmp_path: Path) -> None:
    upscaler = make_upscaler(tmp_path)
    source_path = tmp_path / "source.mkv"
    # Multi-stream original: 0=video, 1=primary audio (jpn), 3=extra audio
    # (eng) -- index 2 is deliberately skipped (e.g. a subtitle stream in the
    # real file) to prove the mapping uses the REAL absolute ffprobe stream
    # index, not a position within the audio-only sub-list.
    job = make_video_job(source_path, audio_track_indices=[1, 3], keep_subtitles=False)
    audio_mux_path = make_audio_mux(tmp_path)

    cmd = upscaler._build_encode_command(
        job, tmp_path / "frames-out", "24/1", audio_mux_path, ["-c:a", "copy"], tmp_path / "out.mkv", "libx264"
    )

    # inputs in order: 0=frames, 1=audio_mux_path, 2=source
    assert cmd[cmd.index(str(audio_mux_path)) - 1] == "-i"
    source_idx = cmd.index(str(source_path))
    assert cmd[source_idx - 1] == "-i"
    assert "2:3" in cmd
    assert cmd[cmd.index("2:3") - 1] == "-map"


def test_maps_multiple_extra_audio_tracks_in_one_source_input(tmp_path: Path) -> None:
    upscaler = make_upscaler(tmp_path)
    source_path = tmp_path / "source.mkv"
    job = make_video_job(source_path, audio_track_indices=[1, 3, 4], keep_subtitles=False)
    audio_mux_path = make_audio_mux(tmp_path)

    cmd = upscaler._build_encode_command(
        job, tmp_path / "frames-out", "24/1", audio_mux_path, ["-c:a", "copy"], tmp_path / "out.mkv", "libx264"
    )

    assert "2:3" in cmd
    assert "2:4" in cmd
    assert cmd.count(str(source_path)) == 1


def test_single_audio_track_index_does_not_add_source_input(tmp_path: Path) -> None:
    # A single entry in audio_track_indices means "no EXTRA tracks" -- the
    # primary already covers it via audio_mux_path, so nothing else to map.
    upscaler = make_upscaler(tmp_path)
    source_path = tmp_path / "source.mkv"
    job = make_video_job(source_path, audio_track_indices=[1], keep_subtitles=False)
    audio_mux_path = make_audio_mux(tmp_path)

    cmd = upscaler._build_encode_command(
        job, tmp_path / "frames-out", "24/1", audio_mux_path, ["-c:a", "copy"], tmp_path / "out.mkv", "libx264"
    )

    assert str(source_path) not in cmd


# ---------------------------------------------------------------------------
# Subtitles
# ---------------------------------------------------------------------------


def test_maps_subtitles_when_keep_subtitles(tmp_path: Path) -> None:
    upscaler = make_upscaler(tmp_path)
    source_path = tmp_path / "source.mkv"
    job = make_video_job(source_path, audio_track_indices=None, keep_subtitles=True)

    cmd = upscaler._build_encode_command(
        job, tmp_path / "frames-out", "24/1", None, [], tmp_path / "out.mkv", "libx264"
    )

    # no audio_mux_path this time -> inputs: 0=frames, 1=source
    source_idx = cmd.index(str(source_path))
    assert cmd[source_idx - 1] == "-i"
    assert "1:s?" in cmd
    assert cmd[cmd.index("1:s?") - 1] == "-map"
    assert cmd[cmd.index("-c:s") + 1] == "copy"


def test_source_input_added_only_once_for_extra_audio_and_subtitles_together(tmp_path: Path) -> None:
    upscaler = make_upscaler(tmp_path)
    source_path = tmp_path / "source.mkv"
    job = make_video_job(source_path, audio_track_indices=[1, 3], keep_subtitles=True)
    audio_mux_path = make_audio_mux(tmp_path)

    cmd = upscaler._build_encode_command(
        job, tmp_path / "frames-out", "24/1", audio_mux_path, ["-c:a", "copy"], tmp_path / "out.mkv", "libx264"
    )

    assert cmd.count(str(source_path)) == 1
    assert cmd.count("-i") == 3  # frames dir, audio_mux_path, source_path -- each exactly once
    assert "2:3" in cmd
    assert "2:s?" in cmd


# ---------------------------------------------------------------------------
# Regression: unchanged when there's nothing extra to preserve
# ---------------------------------------------------------------------------


def test_no_extra_tracks_or_subtitles_is_byte_identical_to_before(tmp_path: Path) -> None:
    upscaler = make_upscaler(tmp_path)
    source_path = tmp_path / "source.mp4"
    job = make_video_job(source_path, audio_track_indices=None, keep_subtitles=False, output_container="mp4")
    audio_mux_path = make_audio_mux(tmp_path)

    cmd = upscaler._build_encode_command(
        job, tmp_path / "frames-out", "24/1", audio_mux_path, ["-c:a", "copy"], tmp_path / "out.mp4", "libx264"
    )

    assert str(source_path) not in cmd
    assert cmd == [
        str(upscaler.settings.ffmpeg_binary_path),
        "-y",
        "-framerate",
        "24/1",
        "-i",
        str(tmp_path / "frames-out" / "%08d.png"),
        "-i",
        str(audio_mux_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        *upscaler._build_video_encode_options(job, "libx264"),
        "-c:a",
        "copy",
        str(tmp_path / "out.mp4"),
    ]


def test_no_audio_no_subtitles_no_extra_source_input(tmp_path: Path) -> None:
    upscaler = make_upscaler(tmp_path)
    source_path = tmp_path / "source.mp4"
    job = make_video_job(source_path, audio_track_indices=None, keep_subtitles=False, output_container="mp4")

    cmd = upscaler._build_encode_command(
        job, tmp_path / "frames-out", "24/1", None, [], tmp_path / "out.mp4", "libx264"
    )

    assert str(source_path) not in cmd
    assert "-c:s" not in cmd
