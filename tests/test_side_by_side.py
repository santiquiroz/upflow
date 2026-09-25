from __future__ import annotations

import hashlib
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.services import cctv_clarify_runner as runner
from app.services import side_by_side as sbs
from app.services.cctv_chain import steps_from_request
from app.services.cctv_ingest import MediaTools, ingest_working_copy
from app.services.ffmpeg_filters import FrameGeometry
from app.services.ffmpeg_progress_runner import spawn_process
from app.services.label_band import FONT_PATH, FontIntegrityError
from app.services.media_signature import sniff_file
from ffmpeg_support import needs_ffmpeg

FFMPEG = Path("C:/tools/ffmpeg.exe")
THREADS = runner.ClarifyThreads(filter_threads=12, ffv1_slices=4, x264_threads=4)
TRIM = {"id": "trim", "params": {"start_frame": 5, "end_frame": 54}}
CROP = {"id": "crop", "params": {"x": 16, "y": 8, "w": 320, "h": 272}}
DENOISE = {"id": "denoise", "params": {"filter": "hqdn3d"}}


def steps(*raw: dict):
    return steps_from_request(list(raw), "classic")


def plan(*raw: dict, geometry=FrameGeometry(352, 288), lane="classic", counters=True) -> sbs.ComparisonPlan:
    return sbs.ComparisonPlan(
        original=Path("C:/session/work.mkv"),
        processed=Path("C:/out/analysis.mkv"),
        steps=steps(*raw),
        processed_geometry=geometry,
        lane=lane,
        processed_frames=50,
        with_counters=counters,
    )


def argv_of(p: sbs.ComparisonPlan, counters: str | None = "drawtext=x") -> list[str]:
    return sbs.build_comparison_command(
        FFMPEG, p, Path("C:/out/comparison_title.png"), 40, Path("C:/out/comparison.mp4"), THREADS, counters
    )


def value_after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


# --- Geometria y grafo ---


def test_lite_mode_compares_at_square_pixels() -> None:
    assert sbs.display_size(FrameGeometry(960, 1080, Fraction(2))) == (1920, 1080)
    assert sbs.display_size(FrameGeometry(704, 576, Fraction(12, 11))) == (704, 576)


def test_the_original_gets_the_same_cut_and_no_processing_filter() -> None:
    chain = sbs.original_chain(steps(TRIM, CROP, DENOISE), (640, 544), None)

    assert chain.startswith("[0:v]trim=start_frame=5:end_frame=55,crop=")
    assert "hqdn3d" not in chain and "bwdif" not in chain
    assert chain.index("crop=") < chain.index("scale=")
    assert chain.endswith("setsar=1,format=yuv420p,setpts=PTS-STARTPTS[sbs_original]")


def test_the_original_is_enlarged_with_nearest_neighbor_only() -> None:
    chain = sbs.original_chain(steps(TRIM), (704, 576), None)

    scale = next(part for part in chain.split(",") if part.startswith("scale="))
    assert scale == "scale=w=704:h=576:flags=neighbor+accurate_rnd+full_chroma_int+bitexact"


def test_the_processed_side_is_only_scaled_when_its_pixels_are_not_square() -> None:
    square = sbs.processed_chain(FrameGeometry(352, 288), (352, 288))
    lite = sbs.processed_chain(FrameGeometry(960, 1080, Fraction(2)), (1920, 1080))

    assert square == "[1:v]setsar=1,format=yuv420p,setpts=PTS-STARTPTS[sbs_processed]"
    assert "scale=w=1920:h=1080:flags=neighbor+" in lite


def test_counters_use_the_bundled_ofl_font_and_start_at_the_trim() -> None:
    counters = sbs.counters_filter(FONT_PATH, 5, 288)

    assert counters.startswith("drawtext=fontfile=")
    assert "SourceCodePro-Regular.ttf" in counters and "consola" not in counters.lower()
    assert r"text=#%{frame_num}  %{pts\\:hms}" in counters
    assert "start_number=5" in counters


def test_the_planned_counters_follow_the_trim_start_and_can_be_turned_off() -> None:
    assert "start_number=5" in sbs.planned_counters(plan(TRIM), FONT_PATH)
    assert "start_number=0" in sbs.planned_counters(plan(DENOISE), FONT_PATH)
    assert sbs.planned_counters(plan(TRIM, counters=False), FONT_PATH) is None


def test_the_graph_stacks_both_sides_under_a_title_band() -> None:
    graph = value_after(argv_of(plan(TRIM)), "-filter_complex")

    assert "[sbs_original][sbs_processed]hstack=inputs=2,pad=w=iw:h=ih+40:x=0:y=40:color=black" in graph
    assert "movie=C\\\\:/out/comparison_title.png[sbs_title]" in graph
    assert graph.endswith("[sbs_stacked][sbs_title]overlay=x=0:y=0:eval=init[sbs_out]")
    assert "drawtext=x" in graph


def test_the_comparison_encode_is_deterministic_x264_without_audio() -> None:
    argv = argv_of(plan(TRIM))

    assert value_after(argv, "-map") == "[sbs_out]" and value_after(argv, "-fps_mode") == "passthrough"
    assert "-an" in argv
    assert value_after(argv, "-c:v") == "libx264" and value_after(argv, "-threads") == "4"
    assert value_after(argv, "-x264-params") == "threads=4:lookahead_threads=1"
    position = argv.index("-fflags")
    assert argv[position : position + 2] == ["-fflags", "+bitexact"] and position > argv.index("-filter_complex")
    assert argv[-1] == str(Path("C:/out/comparison.mp4"))
    assert value_after(argv, "-progress") == "pipe:1"


# --- Banda de titulo ---


@pytest.mark.parametrize(
    ("lane", "right"),
    [("classic", ("PROCESSED (classic filters)", "PROCESADO (filtros clásicos)")),
     ("ai", ("AI VISUALIZATION", "VISUALIZACIÓN CON IA"))],
)  # fmt: skip
def test_the_title_captions_are_bilingual_per_lane(lane: str, right: tuple[str, str]) -> None:
    assert sbs.ORIGINAL_CAPTION == ("ORIGINAL", "ORIGINAL")
    assert sbs.processed_caption(lane) == right


def test_the_title_png_spans_both_halves_with_an_even_height(tmp_path: Path) -> None:
    height = sbs.write_title(tmp_path / "t.png", FONT_PATH, (352, 288), "classic")
    image = Image.open(tmp_path / "t.png")

    assert image.size == (704, height) and height % 2 == 0
    left, right = np.asarray(image)[:, :352], np.asarray(image)[:, 352:]
    assert left.max() > 200 and right.max() > 200


# --- Con ffmpeg real ---


def settings_tools() -> runner.ClarifyTools:
    settings = Settings()
    return runner.ClarifyTools(settings.ffmpeg_binary_path, settings.ffprobe_binary_path)


def make_clip(tmp_path: Path) -> Path:
    clip = tmp_path / "camera.mkv"
    command = [
        str(Settings().ffmpeg_binary_path), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=352x288:rate=25,noise=alls=12:allf=t", "-t", "3",
        "-c:v", "libx264", "-threads", "1", "-x264-params", "threads=1", "-b:v", "300k", "-g", "25", str(clip),
    ]  # fmt: skip
    subprocess.run(command, check=True, capture_output=True)
    return clip


async def clarified(tmp_path: Path, *raw: dict):
    clip = make_clip(tmp_path)
    session = tmp_path / "session"
    session.mkdir()
    settings = Settings()
    media = MediaTools(ffmpeg=settings.ffmpeg_binary_path, ffprobe=settings.ffprobe_binary_path)
    ingested = await ingest_working_copy(media, clip, session, sniff_file(clip))
    chain = steps(*raw)
    clarify_plan = runner.ClarifyPlan(
        ingested.working_copy.path, chain, FrameGeometry(352, 288), "yuv420p", ingested.index.frames, False, "9.9.9"
    )
    out = tmp_path / "out"
    out.mkdir()
    result = await runner.run_clarify(settings_tools(), clarify_plan, out, THREADS)
    return sbs.ComparisonPlan(
        ingested.working_copy.path, result.analysis, chain, result.output_geometry, "classic", result.output_frames
    )


def probe(path: Path) -> str:
    command = [
        str(Settings().ffprobe_binary_path), "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,nb_read_frames", "-of", "compact", str(path),
    ]  # fmt: skip
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()


def first_frame(path: Path, width: int, height: int) -> np.ndarray:
    command = [str(Settings().ffmpeg_binary_path), "-v", "error", "-i", str(path), "-frames:v", "1",
               "-f", "rawvideo", "-pix_fmt", "gray", "-"]  # fmt: skip
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(height, width)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@needs_ffmpeg
async def test_real_comparison_keeps_every_processed_frame_and_is_reproducible(tmp_path: Path) -> None:
    comparison_plan = await clarified(tmp_path, TRIM, CROP, DENOISE)
    first, second = tmp_path / "run1", tmp_path / "run2"
    first.mkdir(), second.mkdir()
    progress: list[float] = []

    a = await sbs.run_side_by_side(settings_tools(), comparison_plan, first, first, THREADS, progress.append)
    b = await sbs.run_side_by_side(settings_tools(), comparison_plan, second, second, THREADS)

    title_height = Image.open(a.title).height
    assert probe(a.path) == f"stream|codec_name=h264|width=640|height={272 + title_height}|nb_read_frames=50"
    assert a.warnings == ()
    assert sha256(a.path) == sha256(b.path)
    assert progress == sorted(progress) and progress[-1] == pytest.approx(1.0)


@needs_ffmpeg
async def test_real_drawtext_failure_falls_back_without_counters_and_warns(tmp_path: Path) -> None:
    comparison_plan = await clarified(tmp_path, TRIM)
    with_counters, without = tmp_path / "with", tmp_path / "without"
    with_counters.mkdir(), without.mkdir()

    async def broken_drawtext(command: list[str]):
        return await spawn_process([arg.replace("drawtext=", "no_such_drawtext=") for arg in command])

    good = await sbs.run_side_by_side(settings_tools(), comparison_plan, with_counters, with_counters, THREADS)
    broken_tools = runner.ClarifyTools(settings_tools().ffmpeg, settings_tools().ffprobe, spawn=broken_drawtext)
    fallback = await sbs.run_side_by_side(broken_tools, comparison_plan, without, without, THREADS)

    assert fallback.warnings == ("cctv.warning.comparisonWithoutCounters",)
    assert not any("drawtext=" in arg for arg in fallback.command)
    height = 288 + Image.open(good.title).height
    title = Image.open(good.title).height
    counted, plain = first_frame(good.path, 704, height), first_frame(fallback.path, 704, height)
    counter_area = (slice(title, title + 40), slice(0, 200))
    assert np.abs(counted[counter_area].astype(int) - plain[counter_area]).mean() > 10
    right_half = (slice(title + 60, height), slice(352, 704))
    assert np.abs(counted[right_half].astype(int) - plain[right_half]).mean() < 1


async def test_a_tampered_font_stops_the_comparison_before_ffmpeg(tmp_path: Path) -> None:
    font = tmp_path / "SourceCodePro-Regular.ttf"
    font.write_bytes(b"tampered")

    async def no_spawn(command: list[str]):
        raise AssertionError("ffmpeg must not run")

    tools = runner.ClarifyTools(FFMPEG, FFMPEG, spawn=no_spawn)

    with pytest.raises(FontIntegrityError):
        await sbs.run_side_by_side(tools, plan(TRIM), tmp_path, tmp_path, THREADS, font_path=font)
