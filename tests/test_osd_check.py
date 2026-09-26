from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.services.cctv_chain import CctvChainError, steps_from_request
from app.services.cctv_presets import PresetContext, preset_steps
from app.services.ffmpeg_filters import OSD_BOX_OUTSIDE_FRAME
from app.services.osd_check import (
    MAX_OSD_BOXES,
    MIN_CONTRAST,
    MIN_STATIC_FRACTION,
    OSD_BOX_ODD,
    OSD_CONFLICT,
    OSD_NOT_TEXT,
    OSD_UNCONFIRMED,
    SAMPLE_FRAMES,
    TOO_MANY_OSD_BOXES,
    OsdSelection,
    build_osd_sample_command,
    check_osd_boxes,
    osd_box_looks_like_text,
    osd_report,
    osd_warnings,
    run_osd_check,
    sample_osd_frames,
    split_gray_frames,
    temporal_filters_in,
    validate_osd_decision,
    validate_osd_selection,
)
from app.services.video_analysis import VideoAnalysisError
from ffmpeg_support import needs_ffmpeg

WIDTH, HEIGHT = 160, 96
CLIP_FPS = 5
CLIP_SECONDS = 12
OSD_BOX = (8, 8, 64, 16)
SCENE_BOX = (96, 56, 48, 32)
SECONDS_DIGIT = (60, 10, 8, 12)
INDEX_STRIPE_ROW = HEIGHT - 2


def glyph_mask() -> np.ndarray:
    columns = np.arange(OSD_BOX[2])
    rows = np.arange(OSD_BOX[3])
    strokes = (columns % 6 < 2)[None, :] | (rows % 8 < 2)[:, None]
    return strokes


def paint_osd(frame: np.ndarray, second: int) -> None:
    x, y, w, h = OSD_BOX
    frame[y : y + h, x : x + w] = np.where(glyph_mask(), 235, 16)
    dx, dy, dw, dh = SECONDS_DIGIT
    frame[dy : dy + dh, dx : dx + dw] = 235 if second % 2 else 16
    frame[dy + dh // 2, dx : dx + dw] = 16 if second % 2 else 235


def noisy_scene(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 256, size=(HEIGHT, WIDTH), dtype=np.uint8)


def synthetic_frame(index: int, rng: np.random.Generator, fps: int = 1) -> np.ndarray:
    frame = noisy_scene(rng)
    paint_osd(frame, index // fps)
    frame[INDEX_STRIPE_ROW:, :] = index * 4
    return frame


def synthetic_frames(count: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([synthetic_frame(index, rng) for index in range(count)])


def flat_frames(count: int) -> np.ndarray:
    return np.full((count, HEIGHT, WIDTH), 120, dtype=np.uint8)


def test_a_synthetic_osd_with_a_changing_seconds_digit_looks_like_text() -> None:
    check = osd_box_looks_like_text(synthetic_frames(SAMPLE_FRAMES), OSD_BOX)

    assert check.looks_like_text
    assert check.contrast >= MIN_CONTRAST
    assert check.static_fraction is not None
    assert MIN_STATIC_FRACTION <= check.static_fraction < 1.0
    assert check.frames == SAMPLE_FRAMES
    assert check.box == OSD_BOX


def test_a_box_over_a_noisy_scene_is_not_text_because_it_is_not_static() -> None:
    check = osd_box_looks_like_text(synthetic_frames(SAMPLE_FRAMES), SCENE_BOX)

    assert not check.looks_like_text
    assert check.contrast >= MIN_CONTRAST
    assert check.static_fraction is not None and check.static_fraction < MIN_STATIC_FRACTION


def test_a_static_box_without_contrast_is_not_text() -> None:
    check = osd_box_looks_like_text(flat_frames(SAMPLE_FRAMES), OSD_BOX)

    assert not check.looks_like_text
    assert check.contrast == 0.0
    assert check.static_fraction == 1.0


def test_a_single_frame_is_judged_by_contrast_alone() -> None:
    frames = synthetic_frames(1)

    assert osd_box_looks_like_text(frames, OSD_BOX).static_fraction is None
    assert osd_box_looks_like_text(frames, OSD_BOX).looks_like_text
    assert not osd_box_looks_like_text(flat_frames(1), OSD_BOX).looks_like_text


def test_the_check_needs_at_least_one_frame() -> None:
    with pytest.raises(ValueError):
        osd_box_looks_like_text(np.zeros((0, HEIGHT, WIDTH), np.uint8), OSD_BOX)


@pytest.mark.parametrize("box", [(150, 8, 16, 16), (-2, 0, 8, 8), (0, 0, 0, 8)])
def test_the_check_refuses_a_box_outside_the_frames(box) -> None:
    with pytest.raises(CctvChainError) as error:
        osd_box_looks_like_text(synthetic_frames(2), box)
    assert error.value.code == OSD_BOX_OUTSIDE_FRAME


def test_check_osd_boxes_reports_every_box_in_order() -> None:
    checks = check_osd_boxes(synthetic_frames(SAMPLE_FRAMES), (OSD_BOX, SCENE_BOX))

    assert [check.box for check in checks] == [OSD_BOX, SCENE_BOX]
    assert [check.looks_like_text for check in checks] == [True, False]


def test_osd_warnings_flag_a_box_that_does_not_look_like_text_once() -> None:
    frames = synthetic_frames(SAMPLE_FRAMES)

    assert osd_warnings(check_osd_boxes(frames, (OSD_BOX,))) == ()
    assert osd_warnings(check_osd_boxes(frames, (SCENE_BOX, SCENE_BOX))) == (OSD_NOT_TEXT,)


def confirmed(*boxes) -> OsdSelection:
    return OsdSelection(boxes=tuple(boxes), confirmed=True, no_osd=False)


@pytest.mark.parametrize(
    "selection",
    [
        OsdSelection(boxes=(), confirmed=False, no_osd=False),
        OsdSelection(boxes=(OSD_BOX,), confirmed=False, no_osd=False),
        OsdSelection(boxes=(), confirmed=True, no_osd=False),
    ],
)
def test_a_job_without_confirmed_boxes_nor_no_osd_is_rejected(selection: OsdSelection) -> None:
    with pytest.raises(CctvChainError) as error:
        validate_osd_selection(selection, WIDTH, HEIGHT)
    assert error.value.code == OSD_UNCONFIRMED


@pytest.mark.parametrize(
    "selection",
    [
        OsdSelection(boxes=(OSD_BOX,), confirmed=False, no_osd=True),
        OsdSelection(boxes=(OSD_BOX,), confirmed=True, no_osd=True),
        OsdSelection(boxes=(), confirmed=True, no_osd=True),
    ],
)
def test_no_osd_excludes_boxes_and_confirmation(selection: OsdSelection) -> None:
    with pytest.raises(CctvChainError) as error:
        validate_osd_selection(selection, WIDTH, HEIGHT)
    assert error.value.code == OSD_CONFLICT


def test_the_decision_alone_is_checked_before_the_frame_size_is_known() -> None:
    far_outside = (4000, 4000, 10, 10)
    validate_osd_decision(confirmed(far_outside))
    validate_osd_decision(OsdSelection(boxes=(), confirmed=False, no_osd=True))
    with pytest.raises(CctvChainError) as unconfirmed:
        validate_osd_decision(OsdSelection(boxes=(far_outside,), confirmed=False, no_osd=False))
    with pytest.raises(CctvChainError) as conflict:
        validate_osd_decision(OsdSelection(boxes=(far_outside,), confirmed=True, no_osd=True))
    assert (unconfirmed.value.code, conflict.value.code) == (OSD_UNCONFIRMED, OSD_CONFLICT)


def test_confirmed_boxes_or_no_osd_are_accepted() -> None:
    validate_osd_selection(confirmed(OSD_BOX, SCENE_BOX), WIDTH, HEIGHT)
    validate_osd_selection(OsdSelection(boxes=(), confirmed=False, no_osd=True), WIDTH, HEIGHT)


@pytest.mark.parametrize(
    "box", [(150, 8, 16, 16), (0, 90, 8, 8), (-2, 0, 8, 8), (0, 0, 0, 8), (0, 0, 8), (0, 0, True, 8), 8, "0088"]
)
def test_confirmed_boxes_must_sit_inside_the_frame(box) -> None:
    with pytest.raises(CctvChainError) as error:
        validate_osd_selection(confirmed(box), WIDTH, HEIGHT)
    assert error.value.code == OSD_BOX_OUTSIDE_FRAME


@pytest.mark.parametrize("box", [(0, 0, 7, 8), (0, 0, 8, 9)])
def test_confirmed_boxes_must_have_even_dimensions(box) -> None:
    with pytest.raises(CctvChainError) as error:
        validate_osd_selection(confirmed(box), WIDTH, HEIGHT)
    assert error.value.code == OSD_BOX_ODD


def test_the_number_of_boxes_is_bounded() -> None:
    boxes = [(0, 0, 8, 8)] * (MAX_OSD_BOXES + 1)
    with pytest.raises(CctvChainError) as error:
        validate_osd_selection(confirmed(*boxes), WIDTH, HEIGHT)
    assert error.value.code == TOO_MANY_OSD_BOXES


def classic_steps(preset_id: str):
    return steps_from_request(preset_steps(preset_id, "classic", PresetContext()), "classic")


def test_temporal_filters_are_named_for_the_presets_that_mix_frames() -> None:
    assert temporal_filters_in(classic_steps("night_ir")) == ("atadenoise",)
    assert temporal_filters_in(classic_steps("day")) == ("hqdn3d",)


FFTDNOIZ_CASES = [({}, ()), ({"prev": 1}, ("fftdnoiz",)), ({"next": 1}, ("fftdnoiz",))]


@pytest.mark.parametrize(("params", "expected"), FFTDNOIZ_CASES)
def test_fftdnoiz_mixes_frames_only_with_prev_or_next(params: dict, expected: tuple) -> None:
    steps = steps_from_request([{"id": "denoise", "params": {"filter": "fftdnoiz", **params}}], "classic")

    assert temporal_filters_in(steps) == expected


def test_the_report_records_boxes_confirmation_checks_and_warnings() -> None:
    selection = confirmed(OSD_BOX, SCENE_BOX)
    checks = check_osd_boxes(synthetic_frames(SAMPLE_FRAMES), selection.boxes)

    report = osd_report(selection, checks, classic_steps("night_ir"))

    assert report["boxes"] == [list(OSD_BOX), list(SCENE_BOX)]
    assert report["osdBoxesConfirmed"] is True
    assert report["noOsd"] is False
    assert [entry["looksLikeText"] for entry in report["checks"]] == [True, False]
    assert report["checks"][1]["box"] == list(SCENE_BOX)
    assert set(report["checks"][0]) == {"box", "looksLikeText", "contrast", "staticFraction", "framesChecked"}
    assert report["temporalFilters"] == ["atadenoise"]
    assert report["warnings"] == [OSD_NOT_TEXT]


def test_the_report_of_a_job_without_on_screen_text_is_empty_but_explicit() -> None:
    selection = OsdSelection(boxes=(), confirmed=False, no_osd=True)

    report = osd_report(selection, (), classic_steps("day"))

    assert report == {
        "boxes": [],
        "osdBoxesConfirmed": False,
        "noOsd": True,
        "checks": [],
        "temporalFilters": ["hqdn3d"],
        "warnings": [],
    }


def test_split_gray_frames_drops_a_trailing_partial_frame() -> None:
    buffer = bytes(range(6)) * 2 + b"\x01"

    frames = split_gray_frames(buffer, 3, 2)

    assert frames.shape == (2, 2, 3)
    assert frames[1, 1, 2] == 5


def test_the_sample_command_takes_ten_gray_frames_one_second_apart(tmp_path: Path) -> None:
    command = build_osd_sample_command(Path("ffmpeg.exe"), tmp_path / "work.mkv", start_seconds=2.5)

    assert command[0] == "ffmpeg.exe"
    assert command[command.index("-ss") + 1] == "2.500"
    assert command.index("-ss") < command.index("-i")
    expected_filter = "select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,0.99),format=gray"
    assert command[command.index("-vf") + 1] == expected_filter
    assert command[command.index("-fps_mode") + 1] == "passthrough"
    assert command[command.index("-frames:v") + 1] == str(SAMPLE_FRAMES)
    assert command[-5:] == ["-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]


def test_the_sample_command_starts_at_the_beginning_by_default(tmp_path: Path) -> None:
    assert "-ss" not in build_osd_sample_command(Path("ffmpeg.exe"), tmp_path / "work.mkv")


@pytest.mark.parametrize("start", [-1.0, float("nan"), float("inf")])
def test_the_sample_command_refuses_a_start_that_is_not_a_time(start: float, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_osd_sample_command(Path("ffmpeg.exe"), tmp_path / "work.mkv", start_seconds=start)


def ffmpeg_path() -> Path:
    return Settings().ffmpeg_binary_path


def write_clip(path: Path) -> Path:
    rng = np.random.default_rng(11)
    frames = [synthetic_frame(index, rng, CLIP_FPS) for index in range(CLIP_FPS * CLIP_SECONDS)]
    command = [
        str(ffmpeg_path()), "-hide_banner", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(CLIP_FPS), "-i", "pipe:0",
        "-pix_fmt", "yuv420p", "-c:v", "ffv1", str(path),
    ]  # fmt: skip
    result = subprocess.run(command, input=b"".join(frame.tobytes() for frame in frames), capture_output=True)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return path


def sampled_indices(frames: np.ndarray) -> list[int]:
    return [round(float(frame[INDEX_STRIPE_ROW:, :].mean()) / 4) for frame in frames]


@needs_ffmpeg
async def test_the_real_binary_samples_ten_frames_one_second_apart(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "work.mkv")

    frames = await sample_osd_frames(ffmpeg_path(), clip, WIDTH, HEIGHT)

    assert frames.shape == (SAMPLE_FRAMES, HEIGHT, WIDTH)
    assert sampled_indices(frames) == [second * CLIP_FPS for second in range(SAMPLE_FRAMES)]


@needs_ffmpeg
async def test_the_sample_can_start_later_in_the_clip(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "work.mkv")

    frames = await sample_osd_frames(ffmpeg_path(), clip, WIDTH, HEIGHT, start_seconds=4.0)

    assert sampled_indices(frames)[:3] == [20, 25, 30]
    assert len(frames) == CLIP_SECONDS - 4


@needs_ffmpeg
async def test_a_misplaced_box_with_a_temporal_preset_warns_and_the_report_records_it(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "work.mkv")
    selection = confirmed(OSD_BOX, SCENE_BOX)

    checks = await run_osd_check(ffmpeg_path(), clip, WIDTH, HEIGHT, selection.boxes)
    report = osd_report(selection, checks, classic_steps("night_ir"))

    assert [check.looks_like_text for check in checks] == [True, False]
    assert all(check.frames == SAMPLE_FRAMES for check in checks)
    assert report["warnings"] == [OSD_NOT_TEXT]
    assert report["checks"][1]["looksLikeText"] is False
    assert report["temporalFilters"] == ["atadenoise"]
    assert report["osdBoxesConfirmed"] is True


@needs_ffmpeg
async def test_run_osd_check_refuses_boxes_outside_the_frame_before_decoding(tmp_path: Path) -> None:
    with pytest.raises(CctvChainError) as error:
        await run_osd_check(ffmpeg_path(), tmp_path / "missing.mkv", WIDTH, HEIGHT, ((150, 8, 16, 16),))
    assert error.value.code == OSD_BOX_OUTSIDE_FRAME


@needs_ffmpeg
async def test_an_unreadable_video_fails_with_the_ffmpeg_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.mkv"
    broken.write_bytes(b"not a video")

    with pytest.raises(VideoAnalysisError):
        await sample_osd_frames(ffmpeg_path(), broken, WIDTH, HEIGHT)


@needs_ffmpeg
async def test_a_start_past_the_end_decodes_no_frames_and_fails(tmp_path: Path) -> None:
    clip = write_clip(tmp_path / "work.mkv")

    with pytest.raises(VideoAnalysisError):
        await sample_osd_frames(ffmpeg_path(), clip, WIDTH, HEIGHT, start_seconds=60.0)
