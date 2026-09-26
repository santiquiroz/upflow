from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from app.config import Settings
from app.services import cctv_clarify_runner as runner
from app.services.cctv_chain import ResolvedStep, steps_from_request
from app.services.cctv_frame_index import FrameEntry
from app.services.cctv_ingest import MediaTools, ingest_working_copy
from app.services.ffmpeg_capabilities import FfmpegCapabilities
from app.services.ffmpeg_filters import FrameGeometry
from app.services.ffmpeg_progress_runner import spawn_process
from app.services.media_signature import sniff_file
from app.services.video_analysis import FrameStats, SourceFacts
from ffmpeg_support import needs_ffmpeg

FFMPEG = Path("C:/tools/ffmpeg.exe")
FFPROBE = Path("C:/tools/ffprobe.exe")
WORK = Path("C:/session/work.mkv")
THREADS = runner.ClarifyThreads(filter_threads=12, ffv1_slices=4, x264_threads=4)
GEOMETRY = FrameGeometry(352, 288)


def frames(count: int, fps: float = 25.0) -> tuple[FrameEntry, ...]:
    return tuple(FrameEntry(n, n / fps, n % 25 == 0, "I" if n % 25 == 0 else "P", 1000) for n in range(count))


def steps(*raw: dict) -> tuple[ResolvedStep, ...]:
    return steps_from_request(list(raw), "classic")


TRIM = {"id": "trim", "params": {"start_frame": 5, "end_frame": 54}}
DENOISE = {"id": "denoise", "params": {"filter": "hqdn3d"}}
GRAY = {"id": "gray", "params": {}}


def plan(*raw: dict, has_audio: bool = True, geometry: FrameGeometry = GEOMETRY, boxes=()) -> runner.ClarifyPlan:
    return runner.ClarifyPlan(
        work=WORK,
        steps=steps(*raw),
        geometry=geometry,
        source_pix_fmt="yuv420p",
        frames=frames(75),
        has_audio=has_audio,
        app_version="9.9.9",
        osd_boxes=boxes,
    )


def analysis_argv(p: runner.ClarifyPlan, threads: runner.ClarifyThreads = THREADS) -> list[str]:
    return runner.build_analysis_command(FFMPEG, p, Path("C:/out/analysis.mkv"), threads)


def viewing_argv(geometry: FrameGeometry = GEOMETRY, threads: runner.ClarifyThreads = THREADS) -> list[str]:
    return runner.build_viewing_command(FFMPEG, Path("C:/out/analysis.mkv"), Path("C:/out/viewing.mp4"), geometry, threads)


def value_after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def values_after(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, arg in enumerate(argv) if arg == flag]


# --- argv de la pasada de analisis ---


@pytest.mark.parametrize("build", [lambda: analysis_argv(plan(DENOISE)), lambda: viewing_argv()])
def test_bitexact_fflags_go_after_the_input_and_right_before_the_output(build) -> None:
    argv = build()

    position = argv.index("-fflags")
    assert argv[position : position + 2] == ["-fflags", "+bitexact"]
    assert position > argv.index("-i")
    assert argv.count("-fflags") == 1
    assert argv[-1].endswith((runner.ANALYSIS_NAME, runner.VIEWING_NAME))
    assert position >= len(argv) - 5


def test_analysis_uses_one_process_with_passthrough_ffv1_fixed_slices_and_all_cores() -> None:
    argv = analysis_argv(plan(DENOISE))

    assert argv[:3] == [str(FFMPEG), "-hide_banner", "-nostdin"]
    assert "-progress" in argv and value_after(argv, "-progress") == "pipe:1"
    assert value_after(argv, "-fps_mode") == "passthrough"
    assert value_after(argv, "-c:v") == "ffv1" and value_after(argv, "-level") == "3"
    assert value_after(argv, "-g") == "1" and value_after(argv, "-slicecrc") == "1"
    assert value_after(argv, "-slices") == "4"
    assert value_after(argv, "-flags:v") == "+bitexact" and value_after(argv, "-flags:a") == "+bitexact"
    assert value_after(argv, "-c:a") == "flac"
    assert values_after(argv, "-threads") == ["12", "12"]
    assert value_after(argv, "-filter_threads") == "12" and value_after(argv, "-filter_complex_threads") == "12"
    assert argv.index("-filter_complex") > argv.index("-i")
    assert value_after(argv, "-metadata") == "comment=Processed with Upflow classic filters 9.9.9; see report.json"


def test_ffv1_slices_stay_fixed_when_the_core_count_changes() -> None:
    one = analysis_argv(plan(DENOISE), runner.ClarifyThreads(1, 4, 4))
    many = analysis_argv(plan(DENOISE), runner.ClarifyThreads(24, 4, 4))

    assert value_after(one, "-slices") == value_after(many, "-slices") == "4"
    assert [a for a in one if a not in ("1", "24")] == [a for a in many if a not in ("1", "24")]


def test_the_graph_maps_the_video_label_and_ends_in_the_source_pix_fmt() -> None:
    argv = analysis_argv(plan(DENOISE, has_audio=False))

    graph = value_after(argv, "-filter_complex")
    assert graph.startswith("[0:v]") and graph.endswith(",format=pix_fmts=yuv420p[v]")
    assert values_after(argv, "-map") == ["[v]", "0:a?"]


def test_the_gray_step_makes_the_graph_end_in_gray() -> None:
    graph = value_after(analysis_argv(plan(GRAY)), "-filter_complex")

    assert graph.endswith("format=pix_fmts=gray[v]")


def test_trim_with_audio_cuts_the_audio_at_the_frame_timestamps_and_maps_it() -> None:
    argv = analysis_argv(plan(TRIM, DENOISE))

    graph = value_after(argv, "-filter_complex")
    assert "trim=start_frame=5:end_frame=55" in graph
    assert graph.endswith(";[0:a]atrim=start=0.2:end=2.2[a]")
    assert values_after(argv, "-map") == ["[v]", "[a]"]


def test_trim_to_the_last_frame_leaves_the_audio_end_open() -> None:
    last = {"id": "trim", "params": {"start_frame": 10, "end_frame": 74}}

    graph = value_after(analysis_argv(plan(last)), "-filter_complex")

    assert graph.endswith("[0:a]atrim=start=0.4[a]")


def test_trim_without_audio_keeps_the_optional_audio_map() -> None:
    argv = analysis_argv(plan(TRIM, has_audio=False))

    assert "atrim" not in value_after(argv, "-filter_complex")
    assert values_after(argv, "-map") == ["[v]", "0:a?"]


@pytest.mark.parametrize(("value", "text"), [(0.0, "0"), (2.2, "2.2"), (0.00001, "0.00001"), (-0.04, "-0.04")])
def test_seconds_are_written_without_exponents(value: float, text: str) -> None:
    assert runner.seconds_text(value) == text


def test_atrim_needs_a_timestamp_for_the_first_kept_frame() -> None:
    untimed = (FrameEntry(0, None, True, "I", 10),)

    with pytest.raises(ValueError, match="timestamp"):
        runner.atrim_filter(steps(TRIM)[0], untimed)


def test_osd_boxes_go_through_the_restoring_graph() -> None:
    graph = value_after(analysis_argv(plan(DENOISE, boxes=((8, 8, 64, 16),))), "-filter_complex")

    assert "split=2[m][o]" in graph and "overlay=x=8:y=8" in graph and graph.endswith("[v]")


@pytest.mark.parametrize(
    ("source", "steps_", "expected"),
    [("yuv420p", (), "yuv420p"), ("yuvj420p", (), "yuvj420p"), ("yuv420p", (GRAY,), "gray")],
)
def test_output_pix_fmt_is_the_source_one_unless_gray(source: str, steps_: tuple, expected: str) -> None:
    assert runner.output_pix_fmt(source, steps(*steps_)) == expected


# --- argv de la copia de visualizacion ---


def test_viewing_copy_uses_fixed_x264_threads_independent_of_the_core_count() -> None:
    few = viewing_argv(threads=runner.ClarifyThreads(1, 4, 4))
    many = viewing_argv(threads=runner.ClarifyThreads(24, 4, 4))

    assert few == many
    assert value_after(many, "-c:v") == "libx264" and value_after(many, "-preset") == "medium"
    assert value_after(many, "-crf") == "16" and value_after(many, "-pix_fmt") == "yuv420p"
    assert value_after(many, "-threads") == "4"
    assert value_after(many, "-x264-params") == "threads=4:lookahead_threads=1"
    assert value_after(many, "-c:a") == "aac" and value_after(many, "-b:a") == "128k"
    assert value_after(many, "-flags") == "+bitexact" and value_after(many, "-movflags") == "+faststart"
    assert value_after(many, "-i") == str(Path("C:/out/analysis.mkv"))


def test_viewing_copy_of_lite_footage_doubles_columns_with_nearest_neighbor() -> None:
    argv = viewing_argv(FrameGeometry(960, 1080, Fraction(2)))

    assert value_after(argv, "-vf") == (
        "scale=w=1920:h=1080:flags=neighbor+accurate_rnd+full_chroma_int+bitexact,setsar=1"
    )


@pytest.mark.parametrize("sar", [Fraction(1), Fraction(4, 3), Fraction(1, 2)])
def test_viewing_copy_keeps_non_integer_or_square_aspect_untouched(sar: Fraction) -> None:
    assert "-vf" not in viewing_argv(FrameGeometry(704, 576, sar))


# --- Conteo ---


@pytest.mark.parametrize(
    ("start", "end", "total", "expected"),
    [(0, 74, 75, 75), (5, 54, 75, 50), (70, 200, 75, 5), (80, 90, 75, 0), (3, 3, 75, 1)],
)
def test_trimmed_count_counts_the_kept_frames(start: int, end: int, total: int, expected: int) -> None:
    trim = ResolvedStep("trim", "trim", {"start_frame": start, "end_frame": end})

    assert runner.trimmed_count(total, trim) == expected


def test_trimmed_count_without_trim_is_the_total() -> None:
    assert runner.trimmed_count(75, None) == 75


def test_frame_count_command_counts_decoded_frames_of_the_first_video_stream() -> None:
    argv = runner.build_frame_count_command(FFPROBE, WORK)

    assert argv[0] == str(FFPROBE) and argv[-1] == str(WORK)
    assert "-count_frames" in argv and value_after(argv, "-select_streams") == "v:0"
    assert value_after(argv, "-show_entries") == "stream=nb_read_frames"


def test_parse_frame_count_reads_the_number_and_rejects_anything_else() -> None:
    assert runner.parse_frame_count(b"75\r\n") == 75
    with pytest.raises(runner.ClarifyStepError, match="frame count"):
        runner.parse_frame_count(b"N/A\n")


def test_a_frame_count_mismatch_fails_with_its_key() -> None:
    runner.check_frame_counts(50, 50)
    with pytest.raises(runner.FrameCountMismatch) as caught:
        runner.check_frame_counts(50, 49)

    assert caught.value.key == "cctv.error.frameCountMismatch"
    assert (caught.value.expected, caught.value.actual) == (50, 49)


# --- Clipping ---


def stats(high: float, low: float) -> FrameStats:
    return FrameStats(
        samples=50, luma_mean=100, chroma_deviation=0, clipped_high_pct=high, clipped_low_pct=low, monochrome=True
    )


def test_clipped_pct_adds_both_ends() -> None:
    assert runner.clipped_pct(stats(1.5, 0.25)) == pytest.approx(1.75)
    assert runner.clipped_pct(None) is None


@pytest.mark.parametrize(
    ("before", "after", "increased"),
    [(1.0, 1.5, False), (1.0, 1.51, True), (2.0, 0.5, False), (None, 3.0, False)],
)
def test_clipping_increase_is_flagged_only_above_half_a_point(before, after, increased: bool) -> None:
    report = runner.ClippingReport(before, after)

    assert report.increased is increased
    limitation = runner.clipping_limitation(report)
    assert (limitation is not None) is increased


def test_clipping_limitation_says_by_how_much() -> None:
    limitation = runner.clipping_limitation(runner.ClippingReport(0.25, 3.5))

    assert limitation.key == "cctv.limitation.clippingIncreased"
    assert limitation.text == "Processing increased clipped pixels by 3.25%"
    assert runner.ClippingReport(0.25, 3.5).to_json() == {
        "beforePct": 0.25,
        "afterPct": 3.5,
        "increasePct": 3.25,
        "increased": True,
    }


def test_clipping_command_samples_fifty_frames_after_the_same_trim() -> None:
    facts = SourceFacts(width=352, height=288, frame_count=500)

    argv = runner.build_clipping_command(FFMPEG, WORK, facts, ["trim=start_frame=5:end_frame=55"])

    assert value_after(argv, "-vf").startswith("trim=start_frame=5:end_frame=55,select=not(mod(n\\,10)),scale=352:288")
    assert value_after(argv, "-frames:v") == "50" and value_after(argv, "-fps_mode") == "passthrough"
    assert argv[-3:] == ["-f", "rawvideo", "-"]


def test_threads_use_every_core_for_filters_and_the_settings_for_x264_and_slices() -> None:
    assert runner.clarify_threads(x264_threads=4, ffv1_slices=6, cpu_count=24) == runner.ClarifyThreads(24, 6, 4)
    assert runner.clarify_threads(4, 4).filter_threads == (os.cpu_count() or 1)


# --- reproduce.cmd ---


CAPS = FfmpegCapabilities(
    binary_sha256="ab" * 32,
    version="ffmpeg version N-123588-g9c63742425-20260323",
    configuration=("--enable-gpl",),
    filters=frozenset(),
    encoders=frozenset(),
    cpu_extensions=("sse4_2", "avx2"),
)


def reproduce_text(argv: list[str], path_map: dict[str, str]) -> str:
    return runner.build_reproduce_script(
        [runner.ReproduceStep("analysis copy", tuple(argv))],
        [runner.ReproduceCheck("reproduced/analysis.mkv", "02_processed/clip__upflow-clarify__1.mkv")],
        path_map,
        CAPS,
    )


def test_reproduce_cmd_holds_the_exact_argv_with_package_relative_paths() -> None:
    argv = analysis_argv(plan(TRIM, DENOISE))
    path_map = {str(WORK): "work.mkv", str(Path("C:/out/analysis.mkv")): "reproduced/analysis.mkv"}

    text = reproduce_text(argv, path_map)

    command = next(line for line in text.splitlines() if line.startswith('"%FFMPEG%"'))
    expected = [path_map.get(arg, arg).replace("/", "\\") for arg in argv[1:]]
    assert command == '"%FFMPEG%" ' + " ".join(f'"{arg}"' for arg in expected)
    assert "C:\\" not in command and "C:/" not in command


def test_reproduce_cmd_names_the_build_and_compares_against_sha256sums() -> None:
    text = reproduce_text(["ffmpeg", "-i", "x"], {})

    assert text.startswith("@echo off\r\n") and text.endswith("\r\n")
    assert "rem Expected ffmpeg build: ffmpeg version N-123588-g9c63742425-20260323" in text
    assert f"rem Expected ffmpeg.exe sha256: {'ab' * 32}" in text
    assert "sse4_2 avx2" in text
    assert 'set "PRODUCED=reproduced\\analysis.mkv"' in text
    assert 'set "LISTED=02_processed/clip__upflow-clarify__1.mkv"' in text
    assert "SHA256SUMS.txt" in text and "certutil -hashfile" in text


def test_reproduce_cmd_switches_to_utf8_and_restores_the_code_page_on_both_exits() -> None:
    lines = reproduce_text(["ffmpeg", "-i", "x"], {}).splitlines()

    assert lines.index("chcp 65001 >nul") < next(i for i, line in enumerate(lines) if line.startswith('"%FFMPEG%"'))
    assert lines.count(runner.RESTORE_CODEPAGE) == 2


def test_reproduce_cmd_escapes_percent_signs() -> None:
    text = reproduce_text(["ffmpeg", "-metadata", "comment=100%"], {})

    assert '"comment=100%%"' in text


def test_reproduce_cmd_quotes_listed_names_everywhere_they_are_echoed() -> None:
    script = runner.build_reproduce_script([], [runner.ReproduceCheck("reproduced/a.mkv", "x & y 5%.mkv")], {}, CAPS)

    assert 'set "PRODUCED=reproduced\\a.mkv"\r\nset "LISTED=x & y 5%%.mkv"\r\ncall :check\r\n' in script
    assert '(echo DIFFERENT "%LISTED%" & set "FAILED=1") else (echo MATCH "%LISTED%")' in script


@pytest.mark.parametrize("bad", ['say "hi"', "two\nlines"])
def test_reproduce_cmd_refuses_arguments_it_cannot_quote(bad: str) -> None:
    with pytest.raises(ValueError, match="cmd"):
        reproduce_text(["ffmpeg", bad], {})


# --- Orquestacion con procesos falsos ---


def fake_tools(counts: list[bytes], spawned: list[list[str]]) -> runner.ClarifyTools:
    async def run(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        return counts.pop(0), b"", 0

    async def spawn(command: list[str]):
        spawned.append(command)
        return await spawn_process([sys.executable, "-c", "print('frame=1'); print('progress=end')"])

    return runner.ClarifyTools(FFMPEG, FFPROBE, run=run, spawn=spawn)


async def test_a_frame_count_mismatch_stops_before_the_viewing_copy(tmp_path: Path) -> None:
    spawned: list[list[str]] = []

    with pytest.raises(runner.FrameCountMismatch):
        await runner.run_clarify(fake_tools([b"75\n", b"74\n"], spawned), plan(DENOISE), tmp_path, THREADS)

    assert len(spawned) == 1 and spawned[0][-1].endswith(runner.ANALYSIS_NAME)


async def test_a_failed_count_step_names_the_step(tmp_path: Path) -> None:
    async def run(command: list[str], timeout: float) -> tuple[bytes, bytes, int]:
        return b"", b"moov atom not found", 1

    async def spawn(command: list[str]):
        return await spawn_process([sys.executable, "-c", "print('progress=end')"])

    tools = runner.ClarifyTools(FFMPEG, FFPROBE, run=run, spawn=spawn)
    with pytest.raises(runner.ClarifyStepError, match="count_frames.*moov atom"):
        await runner.run_clarify(tools, plan(DENOISE), tmp_path, THREADS)


# --- Con ffmpeg real ---


def ffmpeg_path() -> Path:
    return Settings().ffmpeg_binary_path


def media_tools() -> MediaTools:
    settings = Settings()
    return MediaTools(ffmpeg=settings.ffmpeg_binary_path, ffprobe=settings.ffprobe_binary_path)


def real_tools() -> runner.ClarifyTools:
    settings = Settings()
    return runner.ClarifyTools(settings.ffmpeg_binary_path, settings.ffprobe_binary_path)


def make_clip(tmp_path: Path) -> Path:
    clip = tmp_path / "camera.mkv"
    video = "testsrc2=size=352x288:rate=25,noise=alls=12:allf=t"
    command = [
        str(ffmpeg_path()), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", video, "-f", "lavfi", "-i", "sine=sample_rate=8000",
        "-t", "3", "-threads", "1", "-c:v", "libx264", "-x264-params", "threads=1", "-b:v", "200k", "-g", "25",
        "-c:a", "pcm_alaw", "-shortest", str(clip),
    ]  # fmt: skip
    subprocess.run(command, check=True, capture_output=True)
    return clip


async def ingest(tmp_path: Path, clip: Path):
    session = tmp_path / "cctv-session"
    session.mkdir(exist_ok=True)
    return await ingest_working_copy(media_tools(), clip, session, sniff_file(clip))


def real_plan(ingested, *raw: dict, boxes=()) -> runner.ClarifyPlan:
    return runner.ClarifyPlan(
        work=ingested.working_copy.path,
        steps=steps(*raw),
        geometry=FrameGeometry(ingested.video.width, ingested.video.height),
        source_pix_fmt="yuv420p",
        frames=ingested.index.frames,
        has_audio=bool(ingested.audio),
        app_version="9.9.9",
        osd_boxes=boxes,
    )


def probe_json(path: Path, entries: str) -> str:
    settings = Settings()
    command = [str(settings.ffprobe_binary_path), "-v", "error", "-show_entries", entries, "-of", "compact", str(path)]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def decoded_audio_seconds(path: Path) -> float:
    command = [str(ffmpeg_path()), "-v", "error", "-i", str(path), "-map", "0:a", "-ac", "1", "-f", "s16le", "-"]
    samples = len(subprocess.run(command, check=True, capture_output=True).stdout) // 2
    return samples / 8000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@needs_ffmpeg
async def test_real_clarify_keeps_every_trimmed_frame_and_both_outputs(tmp_path: Path) -> None:
    ingested = await ingest(tmp_path, make_clip(tmp_path))
    out = tmp_path / "out"
    out.mkdir()
    stages: list[tuple[str, float]] = []
    p = real_plan(ingested, TRIM, DENOISE, boxes=((8, 8, 96, 24),))

    result = await runner.run_clarify(real_tools(), p, out, THREADS, lambda s, f: stages.append((s, f)))

    assert result.expected_frames == result.output_frames == 50
    analysis = probe_json(result.analysis, "stream=codec_name,pix_fmt:format_tags=comment")
    assert "codec_name=ffv1|pix_fmt=yuv420p" in analysis and "codec_name=flac" in analysis
    assert "Processed with Upflow classic filters 9.9.9" in analysis
    viewing = probe_json(result.viewing, "stream=codec_name,nb_frames")
    assert "codec_name=h264|nb_frames=50" in viewing and "codec_name=aac" in viewing
    assert decoded_audio_seconds(result.analysis) == pytest.approx(2.0, abs=0.01)
    assert stages[-1] == ("verifying", 1.0)
    clarifying = [f for s, f in stages if s == "clarifying"]
    assert clarifying == sorted(clarifying) and clarifying[-1] == pytest.approx(1.0)
    assert result.clipping.before_pct is not None and not result.clipping.increased


@needs_ffmpeg
async def test_real_contrast_boost_reports_the_clipping_increase(tmp_path: Path) -> None:
    ingested = await ingest(tmp_path, make_clip(tmp_path))
    levels = {"id": "levels", "params": {"filter": "eq", "contrast": 3.0}}

    result = await runner.run_clarify(real_tools(), real_plan(ingested, levels), tmp_path, THREADS)

    assert result.clipping.increased
    assert result.limitations[0].key == "cctv.limitation.clippingIncreased"
    assert result.clipping.after_pct > result.clipping.before_pct + 0.5


@needs_ffmpeg
@pytest.mark.skipif(sys.platform != "win32", reason="reproduce.cmd is a Windows batch file")
async def test_real_reproduce_cmd_rebuilds_the_outputs_and_matches_sha256sums(tmp_path: Path) -> None:
    ingested = await ingest(tmp_path, make_clip(tmp_path))
    out = tmp_path / "out"
    out.mkdir()
    result = await runner.run_clarify(real_tools(), real_plan(ingested, TRIM, DENOISE), out, THREADS)
    package = tmp_path / "package folder"
    package.mkdir()
    shutil.copyfile(ingested.working_copy.path, package / "work.mkv")
    stem = "02_processed/cam 1 & (50%)__upflow-clarify__1"
    listed = {"analysis": f"{stem}.mkv", "viewing": f"{stem}.mp4"}
    sums = f"{sha256(result.analysis)} *{listed['analysis']}\n{sha256(result.viewing)} *{listed['viewing']}\n"
    (package / "SHA256SUMS.txt").write_text(sums, encoding="utf-8")
    path_map = {
        str(ingested.working_copy.path): "work.mkv",
        str(result.analysis): "reproduced/analysis.mkv",
        str(result.viewing): "reproduced/viewing.mp4",
    }
    checks = [
        runner.ReproduceCheck("reproduced/analysis.mkv", listed["analysis"]),
        runner.ReproduceCheck("reproduced/viewing.mp4", listed["viewing"]),
    ]
    caps = FfmpegCapabilities(sha256(ffmpeg_path()), "ffmpeg version test", (), frozenset(), frozenset(), ())
    script = runner.build_reproduce_script(result.reproduce_steps(), checks, path_map, caps)
    (package / "reproduce.cmd").write_bytes(script.encode("utf-8"))
    env = {**os.environ, "FFMPEG": str(ffmpeg_path())}

    ok = subprocess.run(["cmd", "/c", str(package / "reproduce.cmd")], capture_output=True, text=True, env=env)

    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert f'MATCH "{listed["analysis"]}"' in ok.stdout and f'MATCH "{listed["viewing"]}"' in ok.stdout
    assert "WARNING" not in ok.stdout
    (package / "SHA256SUMS.txt").write_text(sums.replace(sha256(result.viewing), "0" * 64), encoding="utf-8")

    changed = subprocess.run(["cmd", "/c", str(package / "reproduce.cmd")], capture_output=True, text=True, env=env)

    assert changed.returncode == 1
    assert f'DIFFERENT "{listed["viewing"]}"' in changed.stdout and f'MATCH "{listed["analysis"]}"' in changed.stdout
