from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.services.video_analysis import (
    BLOCK_KEY,
    BLUR_KEY,
    STATS_MAX_PIXELS,
    CctvDiagnosis,
    FrameStats,
    FreezeInterval,
    InterlaceReport,
    MetricSummary,
    SourceFacts,
    VideoAnalysisError,
    analyze_video,
    build_blockdetect_command,
    build_blurdetect_command,
    build_diagnosis,
    build_frame_stats_command,
    build_freezedetect_command,
    build_idet_command,
    diagnosis_warnings,
    frame_stats,
    interlace_report,
    parse_freezes,
    parse_interlace,
    parse_metadata_values,
    split_yuv420p_frames,
    stats_dimensions,
    stats_step,
    suggest_preset,
    summarize,
    yuv420p_frame_size,
)
from ffmpeg_support import needs_ffmpeg

FIXTURES = Path(__file__).parent / "fixtures" / "ffmpeg"
CLIP_NAME = "clip.mkv"
TESTSRC_25 = ["-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25"]
FFV1 = ["-c:v", "ffv1"]
# testsrc2 confunde a idet (un clip progresivo sale TFF); testsrc suavizado no.
IDET_SOURCE_25 = ["-f", "lavfi", "-i", "testsrc=size=320x240:rate=25"]
IDET_SOURCE_50 = ["-f", "lavfi", "-i", "testsrc=size=320x240:rate=50"]
SOFTEN = "gblur=sigma=1.5"

CLIPS: dict[str, list[str]] = {
    "interlaced_tff": [*IDET_SOURCE_50, "-vf", f"{SOFTEN},interlace=scan=tff", "-t", "2", *FFV1],
    "interlaced_bff": [*IDET_SOURCE_50, "-vf", f"{SOFTEN},interlace=scan=bff", "-t", "2", *FFV1],
    "progressive": [*IDET_SOURCE_25, "-vf", SOFTEN, "-t", "2", *FFV1],
    "blocky": [*TESTSRC_25, "-t", "8", "-c:v", "mpeg2video", "-qscale:v", "31", "-g", "12"],
    "clean": [*TESTSRC_25, "-t", "8", *FFV1],
    "blurry": [*TESTSRC_25, "-t", "8", "-vf", "gblur=sigma=4", *FFV1],
    "freeze_middle": [
        *TESTSRC_25, "-t", "3", "-filter_complex", "split[a][b];[a][b]freezeframes=first=25:last=49:replace=24", *FFV1,
    ],
    "freeze_to_end": [
        *TESTSRC_25, "-t", "2", "-filter_complex", "split[a][b];[a][b]freezeframes=first=25:last=100:replace=24", *FFV1,
    ],
}

FIXTURE_RECIPES: dict[str, tuple[str, object]] = {
    "idet_tff.txt": ("interlaced_tff", build_idet_command),
    "idet_bff.txt": ("interlaced_bff", build_idet_command),
    "idet_progressive.txt": ("progressive", build_idet_command),
    "blockdetect_mpeg2_q31.txt": ("blocky", build_blockdetect_command),
    "blockdetect_ffv1.txt": ("clean", build_blockdetect_command),
    "blurdetect_gblur4.txt": ("blurry", build_blurdetect_command),
    "blurdetect_ffv1.txt": ("clean", build_blurdetect_command),
    "freezedetect_middle.txt": ("freeze_middle", build_freezedetect_command),
    "freezedetect_to_end.txt": ("freeze_to_end", build_freezedetect_command),
    "freezedetect_none.txt": ("progressive", build_freezedetect_command),
}


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def make_clip(ffmpeg: Path, workdir: Path, args: list[str], name: str = CLIP_NAME) -> Path:
    subprocess.run(
        [str(ffmpeg), "-hide_banner", "-v", "error", "-y", *args, name],
        cwd=workdir,
        check=True,
        capture_output=True,
    )
    return workdir / name


def record_stderr(ffmpeg: Path, workdir: Path, recipe: str) -> str:
    clip, build = FIXTURE_RECIPES[recipe]
    make_clip(ffmpeg, workdir, CLIPS[clip])
    # Ruta relativa y cwd = workdir: el fixture no guarda rutas de la maquina.
    completed = subprocess.run(build(ffmpeg, Path(CLIP_NAME)), cwd=workdir, capture_output=True, check=True)
    return completed.stderr.decode("utf-8", errors="replace").replace("\r\n", "\n")


def record_all_fixtures(workdir: Path) -> None:
    ffmpeg = Settings().ffmpeg_binary_path
    for recipe in FIXTURE_RECIPES:
        (FIXTURES / recipe).write_text(record_stderr(ffmpeg, workdir, recipe), encoding="utf-8", newline="\n")


def parsed(recipe: str, stderr: str) -> object:
    if recipe.startswith("idet"):
        return parse_interlace(stderr)
    if recipe.startswith("blockdetect"):
        return summarize(parse_metadata_values(stderr, BLOCK_KEY))
    if recipe.startswith("blurdetect"):
        return summarize(parse_metadata_values(stderr, BLUR_KEY))
    return parse_freezes(stderr)


def facts(**overrides) -> SourceFacts:
    return SourceFacts(**{"width": 1920, "height": 1080, "frame_count": 2500, "measured_fps": 25.0, **overrides})


def stats(**overrides) -> FrameStats:
    base = {
        "samples": 50,
        "luma_mean": 120.0,
        "chroma_deviation": 12.0,
        "clipped_high_pct": 0.0,
        "clipped_low_pct": 0.0,
        "monochrome": False,
    }
    return FrameStats(**{**base, **overrides})


def yuv_frames(luma: np.ndarray, u: int, v: int, count: int = 1) -> bytes:
    height, width = luma.shape
    chroma = np.full(width * height // 4, u, np.uint8).tobytes() + np.full(width * height // 4, v, np.uint8).tobytes()
    return (luma.astype(np.uint8).tobytes() + chroma) * count


# --- idet ---------------------------------------------------------------------------------------


def test_idet_parser_reads_the_multi_frame_counts_of_a_tff_clip() -> None:
    report = parse_interlace(fixture("idet_tff.txt"))

    assert report == InterlaceReport(interlaced=True, field_order="tff", tff=50, bff=0, progressive=0, undetermined=0)


def test_idet_parser_reports_bottom_field_first() -> None:
    report = parse_interlace(fixture("idet_bff.txt"))

    assert report is not None and report.interlaced and report.field_order == "bff"


def test_idet_parser_reports_a_progressive_clip_as_not_interlaced() -> None:
    report = parse_interlace(fixture("idet_progressive.txt"))

    assert report is not None
    assert report.interlaced is False
    assert report.field_order is None
    assert report.progressive > 0


def test_idet_parser_returns_none_when_the_filter_printed_nothing() -> None:
    assert parse_interlace("frame=    0 fps=0.0\n") is None


def test_idet_parser_ignores_the_single_frame_line() -> None:
    stderr = (
        "[Parsed_idet_0 @ 0000] Single frame detection: TFF:    40 BFF:     0 Progressive:    10 Undetermined:     0\n"
        "[Parsed_idet_0 @ 0000] Multi frame detection: TFF:     0 BFF:     0 Progressive:    50 Undetermined:     0\n"
    )

    report = parse_interlace(stderr)

    assert report is not None and report.tff == 0 and report.progressive == 50


def test_interlace_needs_enough_decided_frames() -> None:
    assert interlace_report(tff=5, bff=0, progressive=0, undetermined=495).interlaced is False


def test_interlace_needs_most_decided_frames_to_be_interlaced() -> None:
    assert interlace_report(tff=40, bff=0, progressive=60, undetermined=0).interlaced is False
    assert interlace_report(tff=60, bff=0, progressive=40, undetermined=0).interlaced is True


def test_idet_command_limits_the_pass_to_the_first_500_frames() -> None:
    command = build_idet_command(Path("ffmpeg.exe"), Path("work.mkv"))

    assert command[command.index("-vf") + 1] == "idet"
    assert command[command.index("-frames:v") + 1] == "500"
    assert command[-3:] == ["-f", "null", "-"]


# --- blockdetect / blurdetect -------------------------------------------------------------------


def test_blockdetect_parser_reads_one_value_per_sampled_frame() -> None:
    values = parse_metadata_values(fixture("blockdetect_mpeg2_q31.txt"), BLOCK_KEY)

    # 8 s a 25 fps = 200 cuadros; 1 de cada 25 = 8 muestras.
    assert len(values) == 8
    assert all(value > 0 for value in values)


def test_blockdetect_scores_the_mpeg2_q31_clip_far_above_the_lossless_one() -> None:
    blocky = summarize(parse_metadata_values(fixture("blockdetect_mpeg2_q31.txt"), BLOCK_KEY))
    clean = summarize(parse_metadata_values(fixture("blockdetect_ffv1.txt"), BLOCK_KEY))

    assert blocky is not None and clean is not None
    assert blocky.median > 3 * clean.median


def test_blurdetect_scores_the_blurred_clip_above_the_sharp_one() -> None:
    blurry = summarize(parse_metadata_values(fixture("blurdetect_gblur4.txt"), BLUR_KEY))
    sharp = summarize(parse_metadata_values(fixture("blurdetect_ffv1.txt"), BLUR_KEY))

    assert blurry is not None and sharp is not None
    assert blurry.samples == 8
    assert blurry.median > 2 * sharp.median


def test_metadata_parser_only_reads_its_key_and_skips_non_finite_values() -> None:
    stderr = (
        "[Parsed_metadata_2 @ 01] lavfi.block=12.5\n"
        "[Parsed_metadata_2 @ 01] lavfi.blur=99.0\n"
        "[Parsed_metadata_2 @ 01] lavfi.block=nan\n"
        "[Parsed_metadata_2 @ 01] lavfi.block=inf\n"
        "[Parsed_blockdetect_1 @ 02] block mean: 12.5\n"
    )

    assert parse_metadata_values(stderr, BLOCK_KEY) == [12.5]


def test_sampled_commands_take_one_frame_in_25_up_to_200() -> None:
    for build, detector, key in (
        (build_blockdetect_command, "blockdetect", BLOCK_KEY),
        (build_blurdetect_command, "blurdetect", BLUR_KEY),
    ):
        command = build(Path("ffmpeg.exe"), Path("work.mkv"))

        assert command[command.index("-vf") + 1] == (
            f"select=not(mod(n\\,25)),{detector},metadata=mode=print:key={key}"
        )
        assert command[command.index("-frames:v") + 1] == "200"
        assert command[command.index("-fps_mode") + 1] == "passthrough"


def test_summarize_uses_numpy_mean_median_and_p90() -> None:
    summary = summarize([1.0, 2.0, 3.0, 4.0, 10.0])

    assert summary is not None
    assert (summary.samples, summary.mean, summary.median) == (5, 4.0, 3.0)
    assert summary.p90 == pytest.approx(7.6)


def test_summarize_of_nothing_is_none() -> None:
    assert summarize([]) is None


# --- freezedetect -------------------------------------------------------------------------------


def test_freeze_parser_reads_a_closed_interval() -> None:
    assert parse_freezes(fixture("freezedetect_middle.txt")) == [FreezeInterval(start=0.96, end=2.0)]


def test_freeze_parser_keeps_a_freeze_that_runs_to_the_end_open() -> None:
    assert parse_freezes(fixture("freezedetect_to_end.txt")) == [FreezeInterval(start=0.96, end=None)]


def test_freeze_parser_finds_nothing_in_a_moving_clip() -> None:
    assert parse_freezes(fixture("freezedetect_none.txt")) == []


def test_freeze_parser_ignores_an_end_without_start() -> None:
    stderr = (
        "[fd @ 1] lavfi.freezedetect.freeze_end: 1.5\n"
        "[fd @ 1] lavfi.freezedetect.freeze_start: 3\n"
        "[fd @ 1] lavfi.freezedetect.freeze_duration: 1\n"
        "[fd @ 1] lavfi.freezedetect.freeze_end: 4\n"
    )

    assert parse_freezes(stderr) == [FreezeInterval(start=3.0, end=4.0)]


def test_freezedetect_command_uses_the_spec_thresholds_over_the_whole_video() -> None:
    command = build_freezedetect_command(Path("ffmpeg.exe"), Path("work.mkv"))

    assert command[command.index("-vf") + 1] == "freezedetect=n=-60dB:d=0.2"
    assert "-frames:v" not in command


# --- numpy frame stats --------------------------------------------------------------------------


def test_frame_stats_of_a_gray_frame_is_monochrome_with_zero_chroma_deviation() -> None:
    raw = yuv_frames(np.full((4, 8), 60), u=128, v=128, count=3)

    result = frame_stats(split_yuv420p_frames(raw, 8, 4), 8, 4)

    assert result == FrameStats(
        samples=3, luma_mean=60.0, chroma_deviation=0.0, clipped_high_pct=0.0, clipped_low_pct=0.0, monochrome=True
    )


def test_frame_stats_measures_chroma_deviation_and_clipping() -> None:
    luma = np.array([[255, 250, 249, 128], [5, 6, 0, 128]])

    result = frame_stats(split_yuv420p_frames(yuv_frames(luma, u=148, v=108), 4, 2), 4, 2)

    assert result is not None
    assert result.chroma_deviation == 20.0
    assert result.clipped_high_pct == pytest.approx(25.0)
    assert result.clipped_low_pct == pytest.approx(25.0)
    assert result.monochrome is False


def test_split_frames_drops_a_truncated_last_frame() -> None:
    raw = yuv_frames(np.zeros((2, 2)), u=128, v=128, count=2) + b"\x00\x00"

    assert split_yuv420p_frames(raw, 2, 2).shape == (2, yuv420p_frame_size(2, 2))


def test_frame_stats_of_no_frames_is_none() -> None:
    assert frame_stats(split_yuv420p_frames(b"", 4, 4), 4, 4) is None


def test_stats_dimensions_cap_the_pixels_and_stay_even() -> None:
    width, height = stats_dimensions(1920, 1080)

    assert width * height <= STATS_MAX_PIXELS
    assert (width % 2, height % 2) == (0, 0)
    assert stats_dimensions(321, 241) == (320, 240)
    assert stats_dimensions(704, 576) == (704, 576)


@pytest.mark.parametrize("bad", [{"width": 0}, {"height": -1}, {"frame_count": -5}, {"measured_fps": 0.0}])
def test_source_facts_reject_impossible_values(bad) -> None:
    with pytest.raises(ValueError):
        facts(**bad)


def test_stats_step_spreads_50_frames_over_the_index() -> None:
    assert stats_step(2500) == 50
    assert stats_step(30) == 1
    assert stats_step(None) == 25


def test_frame_stats_command_samples_50_spaced_frames_at_full_range() -> None:
    command = build_frame_stats_command(Path("ffmpeg.exe"), Path("work.mkv"), facts(width=640, height=360))

    assert command[command.index("-vf") + 1] == (
        "select=not(mod(n\\,50)),scale=640:360:flags=neighbor:out_range=full,format=yuv420p"
    )
    assert command[command.index("-frames:v") + 1] == "50"
    assert command[-3:] == ["-f", "rawvideo", "-"]


# --- suggested preset and warnings --------------------------------------------------------------

INTERLACED = InterlaceReport(interlaced=True, field_order="tff", tff=400, bff=0, progressive=0, undetermined=100)
PROGRESSIVE = InterlaceReport(interlaced=False, tff=0, bff=0, progressive=500, undetermined=0)
NIGHT = stats(luma_mean=70.0, chroma_deviation=1.0, monochrome=True)


@pytest.mark.parametrize(
    ("source", "interlace", "frame", "expected"),
    [
        (facts(), PROGRESSIVE, stats(), "day"),
        (facts(), None, None, "day"),
        (facts(), PROGRESSIVE, NIGHT, "night_ir"),
        (facts(), INTERLACED, NIGHT, "night_ir"),
        (facts(), PROGRESSIVE, stats(luma_mean=180.0, chroma_deviation=1.0, monochrome=True), "day"),
        (facts(), INTERLACED, stats(), "analog"),
        (facts(width=704, height=576), INTERLACED, stats(), "analog"),
        (facts(width=704, height=576), PROGRESSIVE, stats(), "low_res"),
        (facts(width=352, height=288), None, None, "low_res"),
        (facts(width=960, height=576), PROGRESSIVE, stats(), "day"),
    ],
)
def test_suggested_preset(source, interlace, frame, expected) -> None:
    assert suggest_preset(source, interlace, frame) == expected


def test_a_clean_source_has_no_warnings() -> None:
    blocking = MetricSummary(samples=8, mean=10.0, median=10.0, p90=11.0)
    blur = MetricSummary(samples=8, mean=4.0, median=4.0, p90=5.0)

    assert diagnosis_warnings(facts(), PROGRESSIVE, blocking, blur, [], stats()) == []


def test_every_detected_problem_becomes_a_warning() -> None:
    blocking = MetricSummary(samples=8, mean=120.0, median=116.0, p90=140.0)
    blur = MetricSummary(samples=8, mean=13.0, median=13.0, p90=13.5)
    source = facts(measured_fps=12.5, is_vfr=True, is_lite=True)
    frame = stats(clipped_high_pct=4.0, chroma_deviation=0.5, monochrome=True, luma_mean=60.0)

    warnings = diagnosis_warnings(source, INTERLACED, blocking, blur, [FreezeInterval(start=1.0)], frame)

    assert warnings == [
        "cctv.warning.interlaced",
        "cctv.warning.heavyBlocking",
        "cctv.warning.blurry",
        "cctv.warning.frozenFrames",
        "cctv.warning.clippedHighlights",
        "cctv.warning.monochrome",
        "cctv.warning.lowFrameRate",
        "cctv.warning.variableFrameRate",
        "cctv.warning.liteAspect",
    ]


def test_duplicates_from_the_index_also_warn_about_frozen_frames() -> None:
    assert diagnosis_warnings(facts(probable_duplicates=3), None, None, None, [], None) == [
        "cctv.warning.frozenFrames"
    ]


def test_diagnosis_serializes_with_camel_case_aliases() -> None:
    diagnosis = build_diagnosis(facts(), PROGRESSIVE, None, None, [], NIGHT)

    payload = diagnosis.model_dump(by_alias=True)

    assert payload["suggestedPreset"] == "night_ir"
    assert payload["frameStats"]["lumaMean"] == 70.0
    assert payload["interlace"]["fieldOrder"] is None
    assert payload["warnings"] == ["cctv.warning.monochrome"]


# --- orchestration with a fake runner -----------------------------------------------------------


class FakeRunner:
    def __init__(self, outputs: dict[str, tuple[bytes, bytes, int]]) -> None:
        self.outputs = outputs
        self.commands: list[list[str]] = []

    async def __call__(self, command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        self.commands.append(command)
        filters = command[command.index("-vf") + 1]
        return next(output for marker, output in self.outputs.items() if marker in filters)


def _recorded(name: str) -> tuple[bytes, bytes, int]:
    return b"", fixture(name).encode("utf-8"), 0


async def test_analyze_video_runs_every_pass_and_builds_the_diagnosis() -> None:
    dark_gray = yuv_frames(np.full((360, 640), 40), u=128, v=128, count=50)
    runner = FakeRunner(
        {
            "idet": _recorded("idet_tff.txt"),
            "blockdetect": _recorded("blockdetect_mpeg2_q31.txt"),
            "blurdetect": _recorded("blurdetect_ffv1.txt"),
            "freezedetect": _recorded("freezedetect_middle.txt"),
            "format=yuv420p": (dark_gray, b"", 0),
        }
    )

    diagnosis = await analyze_video(Path("ffmpeg.exe"), Path("work.mkv"), facts(width=640, height=360), run=runner)

    assert [command[command.index("-vf") + 1].split(",")[-1] for command in runner.commands] == [
        "idet",
        f"metadata=mode=print:key={BLOCK_KEY}",
        f"metadata=mode=print:key={BLUR_KEY}",
        "freezedetect=n=-60dB:d=0.2",
        "format=yuv420p",
    ]
    assert isinstance(diagnosis, CctvDiagnosis)
    assert diagnosis.interlace is not None and diagnosis.interlace.field_order == "tff"
    assert diagnosis.blocking is not None and diagnosis.blocking.samples == 8
    assert diagnosis.freezes == [FreezeInterval(start=0.96, end=2.0)]
    assert diagnosis.frame_stats is not None and diagnosis.frame_stats.samples == 50
    assert diagnosis.suggested_preset == "night_ir"


async def test_a_failing_pass_raises_with_its_name_and_stderr() -> None:
    runner = FakeRunner({"idet": (b"", b"work.mkv: Invalid data found when processing input", 1)})

    with pytest.raises(VideoAnalysisError, match="idet.*Invalid data") as caught:
        await analyze_video(Path("ffmpeg.exe"), Path("work.mkv"), facts(), run=runner)

    assert caught.value.pass_name == "idet"


# --- real vendored ffmpeg -----------------------------------------------------------------------


@needs_ffmpeg
@pytest.mark.parametrize("recipe", sorted(FIXTURE_RECIPES))
def test_recorded_fixture_still_matches_what_the_binary_prints(recipe: str, tmp_path: Path) -> None:
    live = record_stderr(Settings().ffmpeg_binary_path, tmp_path, recipe)

    assert parsed(recipe, live) == parsed(recipe, fixture(recipe))


@needs_ffmpeg
async def test_real_analysis_of_an_interlaced_clip_suggests_analog(tmp_path: Path) -> None:
    ffmpeg = Settings().ffmpeg_binary_path
    video = make_clip(ffmpeg, tmp_path, CLIPS["interlaced_tff"])

    diagnosis = await analyze_video(ffmpeg, video, SourceFacts(width=320, height=240, frame_count=50))

    assert diagnosis.interlace is not None and diagnosis.interlace.field_order == "tff"
    assert diagnosis.frame_stats is not None and diagnosis.frame_stats.samples == 50
    assert diagnosis.frame_stats.monochrome is False
    assert diagnosis.suggested_preset == "analog"


@needs_ffmpeg
async def test_real_frame_stats_see_a_dark_gray_clip_as_night_ir(tmp_path: Path) -> None:
    ffmpeg = Settings().ffmpeg_binary_path
    video = make_clip(ffmpeg, tmp_path, ["-f", "lavfi", "-i", "color=c=0x303030:size=1920x1080:rate=25", "-t", "4", *FFV1])

    diagnosis = await analyze_video(ffmpeg, video, SourceFacts(width=1920, height=1080, frame_count=100))

    assert diagnosis.frame_stats is not None
    assert diagnosis.frame_stats.samples == 50
    assert diagnosis.frame_stats.chroma_deviation < 1.0
    assert diagnosis.suggested_preset == "night_ir"


@needs_ffmpeg
async def test_real_frame_stats_read_limited_range_white_as_clipped(tmp_path: Path) -> None:
    ffmpeg = Settings().ffmpeg_binary_path
    white_tv_range = ["-f", "lavfi", "-i", "color=white:size=64x48:rate=25", "-t", "1", "-pix_fmt", "yuv420p", *FFV1]
    video = make_clip(ffmpeg, tmp_path, white_tv_range)

    diagnosis = await analyze_video(ffmpeg, video, SourceFacts(width=64, height=48, frame_count=25))

    assert diagnosis.frame_stats is not None
    assert diagnosis.frame_stats.clipped_high_pct == 100.0
    assert "cctv.warning.clippedHighlights" in diagnosis.warnings
