from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.services import cctv_clarify_runner as runner
from app.services import frame_export as fe
from app.services.cctv_chain import steps_from_request
from app.services.cctv_ingest import MediaTools, ingest_working_copy
from app.services.ffmpeg_filters import FrameGeometry
from app.services.label_band import LabelAssets
from app.services.media_signature import sniff_file
from ffmpeg_support import needs_ffmpeg

FFMPEG = Path("C:/tools/ffmpeg.exe")
WORK = Path("C:/session/work.mkv")
THREADS = runner.ClarifyThreads(filter_threads=4, ffv1_slices=4, x264_threads=4)
FRAMEHASH_OUTPUT = b"""#format: frame checksums
#version: 2
#hash: SHA256
#tb 0: 1/25
#stream#, dts,        pts, duration,     size, hash
0,          7,          7,        1,   152064, AAB0E340f1dea8df615c8f345030862552970274e294bf78f92a916a4d860c86
"""


def value_after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


# --- Validacion ---


def test_still_frames_are_sorted_and_deduplicated() -> None:
    assert fe.checked_still_frames([30, 5, 30, 12], first=0, last=74) == (5, 12, 30)


def test_more_than_twenty_still_frames_are_refused() -> None:
    with pytest.raises(fe.StillFrameError) as error:
        fe.checked_still_frames(list(range(21)), first=0, last=74)

    assert error.value.key == "cctv.error.tooManyStillFrames"


@pytest.mark.parametrize("frame", [4, 55])
def test_still_frames_outside_the_trimmed_range_are_refused(frame: int) -> None:
    with pytest.raises(fe.StillFrameError) as error:
        fe.checked_still_frames([frame], first=5, last=54)

    assert error.value.key == "cctv.error.stillFrameOutOfRange"


def test_twenty_distinct_frames_at_the_range_edges_are_fine() -> None:
    frames = [5, 54, *range(10, 28)]

    assert fe.checked_still_frames(frames, first=5, last=54) == tuple(sorted(frames))


# --- Comandos ---


def test_the_still_command_selects_frame_n_exactly_without_deinterlacing() -> None:
    argv = fe.build_still_command(FFMPEG, WORK, 42, Path("C:/out/original_f42.png"))

    assert value_after(argv, "-vf") == "select=eq(n\\,42)"
    assert value_after(argv, "-frames:v") == "1" and value_after(argv, "-fps_mode") == "passthrough"
    assert value_after(argv, "-map") == "0:v:0" and value_after(argv, "-i") == str(WORK)
    assert argv[-1] == str(Path("C:/out/original_f42.png"))
    assert not any(name in arg for arg in argv for name in ("bwdif", "yadif", "scale=", "-ss"))


def test_negative_frame_numbers_are_rejected() -> None:
    with pytest.raises(ValueError):
        fe.select_frame_filter(-1)


def test_a_labeled_still_puts_the_select_before_the_band(tmp_path: Path) -> None:
    label = LabelAssets(tmp_path / "band.png", tmp_path / "mark.png", 48, 4)

    vf = value_after(fe.build_still_command(FFMPEG, WORK, 3, tmp_path / "p.png", label), "-vf")

    assert vf.startswith("select=eq(n\\,3),null[lb_image];") and "pad=w=iw:h=ih+48" in vf


def test_the_framehash_command_hashes_the_decoded_frame_with_sha256() -> None:
    single = fe.build_framehash_command(FFMPEG, WORK, 7)
    whole = fe.build_framehash_command(FFMPEG, WORK)

    assert value_after(single, "-vf") == "select=eq(n\\,7)" and value_after(single, "-frames:v") == "1"
    for argv in (single, whole):
        assert value_after(argv, "-f") == "framehash" and value_after(argv, "-hash") == "sha256"
        assert argv[-1] == "-" and value_after(argv, "-fps_mode") == "passthrough"
    assert "-vf" not in whole


def test_framehash_lines_are_parsed_and_lowercased() -> None:
    assert fe.parse_framehashes(FRAMEHASH_OUTPUT) == (
        fe.FrameHash(7, "aab0e340f1dea8df615c8f345030862552970274e294bf78f92a916a4d860c86"),
    )


def test_a_malformed_framehash_line_is_an_error() -> None:
    with pytest.raises(ValueError):
        fe.parse_framehashes(b"0, 1, 2\n")


def test_a_single_frame_hash_needs_exactly_one_line() -> None:
    with pytest.raises(runner.ClarifyStepError):
        fe.single_hash((), 7)


# --- Tiempos ---


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.0, "00:00:00.000"), (1.28, "00:00:01.280"), (3725.0406, "01:02:05.041"), (-0.04, "-00:00:00.040")],
)
def test_timecode_is_hours_minutes_seconds_and_millis(seconds: float, expected: str) -> None:
    assert fe.timecode(seconds) == expected


@pytest.mark.parametrize(("target", "expected"), [(0.0, 0), (0.05, 1), (0.059, 1), (0.061, 2), (0.9, 3), (-1.0, 0)])
def test_nearest_frame_by_pts_picks_the_closest_timestamp(target: float, expected: int) -> None:
    assert fe.nearest_frame((0.0, 0.04, 0.08, 0.12), target) == expected


def test_nearest_frame_needs_frames() -> None:
    with pytest.raises(ValueError):
        fe.nearest_frame((), 0.0)


def request(**overrides) -> fe.StillRequest:
    fields = {
        "original": fe.StillSource(WORK, tuple(n * 0.1 for n in range(20))),
        "processed": fe.StillSource(Path("C:/out/analysis.mkv"), tuple(n * 0.04 for n in range(40))),
        "frames": (8,),
        "output_dir": Path("C:/out"),
        "trim_start": 5,
    }
    return fe.StillRequest(**{**fields, **overrides})


def test_exact_lane_maps_frame_n_through_the_trim_start() -> None:
    assert fe.processed_frame(request(), 8) == 3


def test_retimed_lane_pairs_by_the_nearest_pts_from_the_trim_start() -> None:
    assert fe.processed_frame(request(retimed=True), 8) == 8


def test_the_band_and_the_xmp_go_together(tmp_path: Path) -> None:
    label = LabelAssets(tmp_path / "band.png", tmp_path / "mark.png", 48, 4)

    with pytest.raises(ValueError):
        request(label=label)
    with pytest.raises(ValueError):
        request(xmp_packet="<x/>")


def test_an_approximate_pair_says_so_in_the_report(tmp_path: Path) -> None:
    original = fe.StillFrame("original", 8, tmp_path / "original_f8.png", 0.8, "a" * 64)
    processed = fe.StillFrame("processed", 20, tmp_path / "processed_f8.png", 0.8, "b" * 64)

    exact = fe.StillPair(8, original, processed, approximate=False).to_json(tmp_path)
    approximate = fe.StillPair(8, original, processed, approximate=True).to_json(tmp_path)

    assert "note" not in exact and exact["approximate"] is False
    assert approximate["note"] == "approximate (re-timed frames)"
    assert exact["original"] == {
        "role": "original",
        "frame": 8,
        "file": "original_f8.png",
        "ptsTime": 0.8,
        "timecode": "00:00:00.800",
        "framehashSha256": "a" * 64,
    }


# --- Con ffmpeg real ---


def ffmpeg_path() -> Path:
    return Settings().ffmpeg_binary_path


def tools() -> runner.ClarifyTools:
    settings = Settings()
    return runner.ClarifyTools(settings.ffmpeg_binary_path, settings.ffprobe_binary_path)


def make_clip(path: Path) -> Path:
    # B-frames: el orden de decodificacion no es el de salida, y N tiene que ser el de salida.
    command = [
        str(ffmpeg_path()), "-hide_banner", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=352x288:rate=25,noise=alls=12:allf=t", "-t", "3",
        "-c:v", "libx264", "-threads", "1", "-x264-params", "threads=1", "-g", "25", "-bf", "3",
        "-b:v", "300k", str(path),
    ]  # fmt: skip
    subprocess.run(command, check=True, capture_output=True)
    return path


def run(command: list[str]) -> bytes:
    return subprocess.run(command, check=True, capture_output=True).stdout


def full_framehashes(source: Path) -> tuple[fe.FrameHash, ...]:
    return fe.parse_framehashes(run(fe.build_framehash_command(ffmpeg_path(), source)))


def extract_all(source: Path, directory: Path) -> list[Path]:
    directory.mkdir()
    pattern = directory / "f%05d.png"
    run([str(ffmpeg_path()), "-v", "error", "-i", str(source), "-map", "0:v:0", "-fps_mode", "passthrough",
         "-start_number", "0", str(pattern)])  # fmt: skip
    return sorted(directory.glob("f*.png"))


def pixels(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path))


@needs_ffmpeg
async def test_real_select_n_matches_the_full_extraction_and_its_framehash(tmp_path: Path) -> None:
    clip = make_clip(tmp_path / "camera.mkv")
    everything = extract_all(clip, tmp_path / "all")
    hashes = full_framehashes(clip)
    assert len(everything) == len(hashes) == 75

    for frame in (0, 1, 2, 26, 49, 74):
        still = tmp_path / f"still_{frame}.png"
        path, digest = await fe.export_still(tools(), fe.StillSource(clip, ()), frame, still)

        assert np.array_equal(pixels(path), pixels(everything[frame]))
        assert digest == hashes[frame].sha256
    assert len({h.sha256 for h in hashes}) == 75


@needs_ffmpeg
async def test_real_pairs_hash_the_original_and_the_processed_frame_through_the_trim(tmp_path: Path) -> None:
    clip = make_clip(tmp_path / "camera.mkv")
    session = tmp_path / "session"
    session.mkdir()
    settings = Settings()
    media = MediaTools(ffmpeg=settings.ffmpeg_binary_path, ffprobe=settings.ffprobe_binary_path)
    ingested = await ingest_working_copy(media, clip, session, sniff_file(clip))
    work = ingested.working_copy.path
    trim = {"id": "trim", "params": {"start_frame": 5, "end_frame": 54}}
    steps = steps_from_request([trim, {"id": "denoise", "params": {"filter": "hqdn3d"}}], "classic")
    plan = runner.ClarifyPlan(work, steps, FrameGeometry(352, 288), "yuv420p", ingested.index.frames, False, "9.9.9")
    out = tmp_path / "out"
    out.mkdir()
    result = await runner.run_clarify(tools(), plan, out, THREADS)
    times = tuple(entry.pts_time for entry in ingested.index.frames)
    frames = fe.checked_still_frames([30, 5, 54], first=5, last=54)
    still_request = fe.StillRequest(
        original=fe.StillSource(work, times),
        processed=fe.StillSource(result.analysis, ()),
        frames=frames,
        output_dir=out,
        trim_start=5,
    )

    pairs = await fe.export_still_pairs(tools(), still_request)

    work_hashes, analysis_hashes = full_framehashes(work), full_framehashes(result.analysis)
    assert [pair.frame for pair in pairs] == [5, 30, 54]
    for pair in pairs:
        assert pair.original.framehash == work_hashes[pair.frame].sha256
        assert pair.processed.frame == pair.frame - 5
        assert pair.processed.framehash == analysis_hashes[pair.frame - 5].sha256
        assert pair.processed.pts_time == pair.original.pts_time == times[pair.frame]
        assert pair.original.path.name == f"original_f{pair.frame}.png"
        assert pair.processed.path.name == f"processed_f{pair.frame}.png"
        assert Image.open(pair.original.path).size == Image.open(pair.processed.path).size == (352, 288)
        assert not pair.approximate
    assert pairs[1].original.framehash != pairs[1].processed.framehash
