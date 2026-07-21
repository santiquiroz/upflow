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
    # CRITICAL: the subtitle -map switches ffmpeg to explicit-map mode; without
    # an explicit video map the frames stream is dropped and the output has no
    # video. The video map must be emitted even when audio_mux_path is None.
    assert "0:v:0" in cmd
    assert cmd[cmd.index("0:v:0") - 1] == "-map"


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
    # The video map must appear exactly once here -- emitted with the audio_mux
    # input, never double-emitted by the source-input branch.
    assert cmd.count("0:v:0") == 1


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


# ---------------------------------------------------------------------------
# Reviewer finding 1 (CRITICAL): the general "-c:a <codec>" applies to EVERY
# output audio stream. When enhance/restore re-encodes the primary, the extra
# tracks must NOT be dragged along -- they were mapped raw from the source and
# must be copied verbatim. A per-output-position "-c:a:<pos> copy" (emitted
# AFTER the general codec so ffmpeg's last-match-wins resolution keeps it) must
# override the general codec for each extra, while the primary keeps the codec.
# ---------------------------------------------------------------------------


def _positions_of(cmd: list[str], needle: str) -> list[int]:
    return [i for i, token in enumerate(cmd) if token == needle]


def test_extra_audio_tracks_are_copied_while_primary_is_reencoded(tmp_path: Path) -> None:
    upscaler = make_upscaler(tmp_path)
    source_path = tmp_path / "source.mkv"
    # audio_restore/enhance active -> primary re-encoded to AAC 192k. Extra
    # track (absolute stream 2) must stay copy, not be transcoded.
    job = make_video_job(source_path, audio_track_indices=[1, 2], keep_subtitles=False)
    audio_mux_path = make_audio_mux(tmp_path)

    cmd = upscaler._build_encode_command(
        job,
        tmp_path / "frames-out",
        "24/1",
        audio_mux_path,
        ["-c:a", "aac", "-b:a", "192k"],
        tmp_path / "out.mkv",
        "libx264",
    )

    # The extra (output audio position 1) is overridden to copy...
    assert "-c:a:1" in cmd
    assert cmd[cmd.index("-c:a:1") + 1] == "copy"
    # ...and the general aac (primary, output audio 0) is emitted BEFORE the
    # per-position override so ffmpeg's last-match-wins keeps copy for the extra
    # while output audio 0 stays aac.
    general_ca = _positions_of(cmd, "-c:a")
    assert general_ca, "primary must still receive the general -c:a aac"
    assert min(general_ca) < cmd.index("-c:a:1")
    assert cmd[min(general_ca) + 1] == "aac"
    # Only one extra -> exactly one per-position override, no "-c:a:2".
    assert "-c:a:2" not in cmd


def test_two_extra_audio_tracks_each_get_a_copy_override(tmp_path: Path) -> None:
    upscaler = make_upscaler(tmp_path)
    source_path = tmp_path / "source.mkv"
    job = make_video_job(source_path, audio_track_indices=[1, 2, 3], keep_subtitles=False)
    audio_mux_path = make_audio_mux(tmp_path)

    cmd = upscaler._build_encode_command(
        job,
        tmp_path / "frames-out",
        "24/1",
        audio_mux_path,
        ["-c:a", "aac", "-b:a", "192k"],
        tmp_path / "out.mkv",
        "libx264",
    )

    assert cmd[cmd.index("-c:a:1") + 1] == "copy"
    assert cmd[cmd.index("-c:a:2") + 1] == "copy"
    assert "-c:a:3" not in cmd


def test_no_extra_tracks_emits_no_per_position_copy(tmp_path: Path) -> None:
    upscaler = make_upscaler(tmp_path)
    source_path = tmp_path / "source.mkv"
    job = make_video_job(source_path, audio_track_indices=[1], keep_subtitles=False)
    audio_mux_path = make_audio_mux(tmp_path)

    cmd = upscaler._build_encode_command(
        job,
        tmp_path / "frames-out",
        "24/1",
        audio_mux_path,
        ["-c:a", "aac", "-b:a", "192k"],
        tmp_path / "out.mkv",
        "libx264",
    )

    assert "-c:a:1" not in cmd


# ---------------------------------------------------------------------------
# Reviewer finding 3 (IMPORTANT): a duplicate index (CSV-parsed with no dedup
# at the route) must not map the primary stream twice -- once as primary, once
# as extra.
# ---------------------------------------------------------------------------


def test_duplicate_primary_index_is_not_mapped_twice(tmp_path: Path) -> None:
    upscaler = make_upscaler(tmp_path)
    source_path = tmp_path / "source.mkv"
    # [1, 2, 1]: 1 is primary (via audio_mux_path), 2 is the only real extra;
    # the trailing 1 must be dropped, not re-mapped as an extra.
    job = make_video_job(source_path, audio_track_indices=[1, 2, 1], keep_subtitles=False)
    audio_mux_path = make_audio_mux(tmp_path)

    cmd = upscaler._build_encode_command(
        job,
        tmp_path / "frames-out",
        "24/1",
        audio_mux_path,
        ["-c:a", "aac", "-b:a", "192k"],
        tmp_path / "out.mkv",
        "libx264",
    )

    assert upscaler._extra_audio_track_indices(job) == [2]
    # stream 1 is only ever the primary (never mapped as an extra "2:1")
    assert "2:1" not in cmd
    assert "2:2" in cmd
    # exactly one extra -> exactly one per-position copy override
    assert "-c:a:1" in cmd
    assert "-c:a:2" not in cmd


# ---------------------------------------------------------------------------
# Reviewer finding 2 (IMPORTANT): the user's chosen PRIMARY track must be
# selected explicitly in the extraction step. Without -map, ffmpeg picks its
# own default stream and ignores the user's choice.
# ---------------------------------------------------------------------------


class _CapturingUpscaler(VideoUpscaler):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.commands: list[list[str]] = []

    async def _run_process(self, command: list[str]) -> None:
        self.commands.append(list(command))


def make_capturing_upscaler(tmp_path: Path) -> _CapturingUpscaler:
    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))
    return _CapturingUpscaler(settings, _FakeEngine(), _FakeMediaTools())


async def test_single_index_maps_primary_stream_explicitly_in_extraction(tmp_path: Path) -> None:
    upscaler = make_capturing_upscaler(tmp_path)
    source_path = tmp_path / "source.mkv"
    source_path.write_bytes(b"fake")
    job = make_video_job(source_path, audio_track_indices=[3], keep_subtitles=False)

    await upscaler._prepare_original_audio(job, tmp_path / "audio.m4a")

    cmd = upscaler.commands[-1]
    assert "-map" in cmd
    # ABSOLUTE index, mapped right after the source input, never "0:a:3".
    map_idx = cmd.index("-map")
    assert cmd[map_idx + 1] == "0:3"
    assert "0:a:3" not in cmd
    assert cmd[cmd.index(str(source_path)) - 1] == "-i"
    assert cmd.index(str(source_path)) < map_idx


async def test_single_index_maps_primary_stream_explicitly_in_wav_extraction(tmp_path: Path) -> None:
    upscaler = make_capturing_upscaler(tmp_path)
    source_path = tmp_path / "source.mkv"
    source_path.write_bytes(b"fake")
    job = make_video_job(source_path, audio_track_indices=[3], keep_subtitles=False)

    await upscaler._extract_audio_wav(job, tmp_path / "audio.wav")

    cmd = upscaler.commands[-1]
    assert cmd[cmd.index("-map") + 1] == "0:3"
    assert "pcm_s16le" in cmd


async def test_extraction_without_track_indices_is_byte_identical(tmp_path: Path) -> None:
    upscaler = make_capturing_upscaler(tmp_path)
    source_path = tmp_path / "source.mkv"
    source_path.write_bytes(b"fake")
    job = make_video_job(source_path, audio_track_indices=None, keep_subtitles=False)
    audio_path = tmp_path / "audio.m4a"

    await upscaler._prepare_original_audio(job, audio_path)

    assert upscaler.commands[-1] == [
        str(upscaler.settings.ffmpeg_binary_path),
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(audio_path),
    ]


async def test_wav_extraction_without_track_indices_is_byte_identical(tmp_path: Path) -> None:
    upscaler = make_capturing_upscaler(tmp_path)
    source_path = tmp_path / "source.mkv"
    source_path.write_bytes(b"fake")
    job = make_video_job(source_path, audio_track_indices=None, keep_subtitles=False)
    audio_wav_path = tmp_path / "audio.wav"

    await upscaler._extract_audio_wav(job, audio_wav_path)

    assert upscaler.commands[-1] == [
        str(upscaler.settings.ffmpeg_binary_path),
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "48000",
        str(audio_wav_path),
    ]


# ---------------------------------------------------------------------------
# CRITICAL (fix round 2): video-stream loss. Task 3 added extra-audio/subtitle
# maps gated by _needs_source_input, which fires INDEPENDENTLY of audio_mux_path
# (e.g. keep_audio=False + keep_subtitles=True, or a source whose primary audio
# is an unusable codec). Any -map puts ffmpeg in explicit-map mode and drops
# every unmapped stream -- so without an explicit video map the frames stream
# is silently dropped and the output has no video track. The video map must be
# present in EVERY explicit-map branch, and emitted exactly once.
# ---------------------------------------------------------------------------


def test_subtitles_only_without_audio_mux_still_maps_video(tmp_path: Path) -> None:
    # keep_audio=False-equivalent: no audio_mux_path, subtitles requested.
    upscaler = make_upscaler(tmp_path)
    source_path = tmp_path / "source.mkv"
    job = make_video_job(source_path, audio_track_indices=None, keep_subtitles=True, keep_audio=False)

    cmd = upscaler._build_encode_command(
        job, tmp_path / "frames-out", "24/1", None, [], tmp_path / "out.mkv", "libx264"
    )

    assert cmd.count("0:v:0") == 1
    assert cmd[cmd.index("0:v:0") - 1] == "-map"
    # Video must be mapped BEFORE the subtitle stream so it stays output 0.
    assert cmd.index("0:v:0") < cmd.index("1:s?")


def test_extra_audio_only_without_audio_mux_still_maps_video(tmp_path: Path) -> None:
    # audio_mux_path=None while extra tracks are requested (e.g. the primary's
    # codec was unusable so no mux file was produced) -- the source-input branch
    # still fires and must carry the video map.
    upscaler = make_upscaler(tmp_path)
    source_path = tmp_path / "source.mkv"
    job = make_video_job(source_path, audio_track_indices=[1, 3], keep_subtitles=False)

    cmd = upscaler._build_encode_command(
        job, tmp_path / "frames-out", "24/1", None, [], tmp_path / "out.mkv", "libx264"
    )

    assert cmd.count("0:v:0") == 1
    assert cmd[cmd.index("0:v:0") - 1] == "-map"
    # source is input 1 (no audio_mux consumed index 1), extra audio mapped 1:3.
    assert "1:3" in cmd
    assert cmd.index("0:v:0") < cmd.index("1:3")


# ---------------------------------------------------------------------------
# Final-review finding (IMPORTANT): extra-audio mapping must be gated by
# keep_audio. Before this fix, a job with keep_audio=False (so audio_mux_path
# is None) but audio_track_indices with 2+ entries still added the source
# input and mapped the extras -- since audio_mux_path was None,
# _extra_audio_copy_args never ran, so ffmpeg silently RE-ENCODED the extra
# track to the container default codec instead of honoring "keep audio off".
# ---------------------------------------------------------------------------


def test_keep_audio_false_maps_no_extra_audio_even_with_multiple_indices(tmp_path: Path) -> None:
    upscaler = make_upscaler(tmp_path)
    source_path = tmp_path / "source.mkv"
    job = make_video_job(
        source_path, audio_track_indices=[0, 1], keep_subtitles=False, keep_audio=False
    )

    assert upscaler._extra_audio_track_indices(job) == []
    assert upscaler._needs_source_input(job) is False

    cmd = upscaler._build_encode_command(
        job, tmp_path / "frames-out", "24/1", None, [], tmp_path / "out.mkv", "libx264"
    )

    # No source input at all -- nothing extra to preserve, so no re-encoded
    # stray audio track can sneak into the output.
    assert str(source_path) not in cmd
    assert "-c:a" not in cmd


def test_keep_audio_false_with_keep_subtitles_still_maps_subs_and_video(tmp_path: Path) -> None:
    # Must-not-break regression: keep_audio=False + keep_subtitles=True is a
    # VALID, independent combination -- subtitles have nothing to do with audio.
    upscaler = make_upscaler(tmp_path)
    source_path = tmp_path / "source.mkv"
    job = make_video_job(
        source_path, audio_track_indices=[0, 1], keep_subtitles=True, keep_audio=False
    )

    assert upscaler._extra_audio_track_indices(job) == []
    assert upscaler._needs_source_input(job) is True

    cmd = upscaler._build_encode_command(
        job, tmp_path / "frames-out", "24/1", None, [], tmp_path / "out.mkv", "libx264"
    )

    source_idx = cmd.index(str(source_path))
    assert cmd[source_idx - 1] == "-i"
    assert cmd.count("0:v:0") == 1
    assert cmd[cmd.index("0:v:0") - 1] == "-map"
    assert "1:s?" in cmd
    assert cmd[cmd.index("-c:s") + 1] == "copy"
    # No extra audio stream index (e.g. "1:1") was mapped from the source.
    assert "1:1" not in cmd
    assert "-c:a" not in cmd


def test_audio_present_with_source_maps_video_exactly_once_byte_identical(tmp_path: Path) -> None:
    # Regression guard: audio_mux_path present AND source input needed. The
    # video map is emitted exactly once (with the audio input), never doubled by
    # the source-input branch. Full byte-for-byte assertion.
    upscaler = make_upscaler(tmp_path)
    source_path = tmp_path / "source.mkv"
    job = make_video_job(source_path, audio_track_indices=[1, 3], keep_subtitles=True)
    audio_mux_path = make_audio_mux(tmp_path)

    cmd = upscaler._build_encode_command(
        job, tmp_path / "frames-out", "24/1", audio_mux_path, ["-c:a", "copy"], tmp_path / "out.mkv", "libx264"
    )

    assert cmd.count("0:v:0") == 1
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
        "-i",
        str(source_path),
        "-map",
        "2:3",
        "-map",
        "2:s?",
        *upscaler._build_video_encode_options(job, "libx264"),
        "-c:a",
        "copy",
        "-c:a:1",
        "copy",
        "-c:s",
        "copy",
        str(tmp_path / "out.mkv"),
    ]
