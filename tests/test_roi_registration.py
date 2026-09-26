from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.services import roi_frames
from app.services import roi_registration as rr
from app.services.cctv_frame_index import parse_frame_index
from app.services.cctv_ingest import build_frame_index_command
from ffmpeg_support import needs_ffmpeg

HEIGHT, WIDTH = 240, 320
PLATE_BOX = rr.RoiBox(120, 90, 80, 40)
MAX_CORNER_ERROR_PX = 0.1


def texture(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0, 255, (HEIGHT, WIDTH)).astype(np.float32)
    return np.clip(cv2.GaussianBlur(noise, (0, 0), 2.0) * 3.0 - 255.0, 0, 240).astype(np.float32)


def translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)


def about_center(matrix: np.ndarray, box: rr.RoiBox) -> np.ndarray:
    cx, cy = box.center
    return translation(cx, cy) @ matrix @ translation(-cx, -cy)


def perspective(dx: float, dy: float, angle: float, px: float, py: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    local = np.array([[cos, -sin, 0], [sin, cos, 0], [px, py, 1]], dtype=np.float64)
    return translation(dx, dy) @ about_center(local, PLATE_BOX)


def affine(dx: float, dy: float, angle: float, zoom: float) -> np.ndarray:
    cos, sin = np.cos(angle) * zoom, np.sin(angle) * zoom
    local = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]], dtype=np.float64)
    return translation(dx, dy) @ about_center(local, PLATE_BOX)


def render(base: np.ndarray, reference_to_frame: np.ndarray) -> np.ndarray:
    # Contenido del cuadro en x = contenido de la referencia en inv(M)·x.
    return cv2.warpPerspective(
        base, reference_to_frame, (WIDTH, HEIGHT), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT
    )


def frames_for(matrices: list[np.ndarray], base: np.ndarray | None = None) -> tuple[rr.RoiFrame, ...]:
    source = texture() if base is None else base
    return tuple(rr.RoiFrame(n, "P", render(source, m)) for n, m in enumerate(matrices))


def box_corners(box: rr.RoiBox) -> np.ndarray:
    right, bottom = box.x + box.w - 1, box.y + box.h - 1
    return np.array([[box.x, box.y], [right, box.y], [box.x, bottom], [right, bottom]], dtype=np.float64)


def mapped(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.c_[points, np.ones(len(points))] @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:]


def corner_error(estimated: tuple[tuple[float, ...], ...], truth: np.ndarray, box: rr.RoiBox) -> float:
    corners = box_corners(box)
    return float(np.max(np.linalg.norm(mapped(np.array(estimated), corners) - mapped(truth, corners), axis=1)))


def register(frames, reference: int = 0, box: rr.RoiBox = PLATE_BOX, motion: rr.Motion = "homography", **kw):
    return rr.register_roi_frames(frames, reference, box, motion, rr.RegistrationSettings(**kw))


# --- Tipo de movimiento y ventana ---


@pytest.mark.parametrize(
    ("kind", "box", "expected"),
    [
        ("plate", rr.RoiBox(0, 0, 80, 40), "homography"),
        ("face_or_object", rr.RoiBox(0, 0, 40, 48), "affine"),
        ("plate", rr.RoiBox(0, 0, 60, 22), "translation"),
        ("face_or_object", rr.RoiBox(0, 0, 20, 30), "translation"),
    ],
)
def test_the_motion_model_follows_the_kind_and_small_rois_fall_back_to_translation(kind, box, expected) -> None:
    assert rr.motion_for(kind, box) == expected


def test_the_search_window_grows_the_roi_by_half_around_its_center() -> None:
    assert rr.search_window(rr.RoiBox(100, 100, 40, 20), (240, 320), 0.5) == rr.RoiBox(90, 95, 60, 30)


def test_the_search_window_is_clamped_inside_the_frame() -> None:
    assert rr.search_window(rr.RoiBox(0, 0, 40, 20), (240, 320), 0.5) == rr.RoiBox(0, 0, 50, 25)


def test_a_box_outside_the_frame_is_refused() -> None:
    frames = frames_for([np.eye(3)])
    with pytest.raises(rr.RoiRegistrationError) as error:
        register(frames, box=rr.RoiBox(300, 200, 40, 40))
    assert error.value.key == rr.ROI_OUTSIDE_FRAME


def test_the_reference_has_to_be_one_of_the_frames() -> None:
    with pytest.raises(rr.RoiRegistrationError) as error:
        register(frames_for([np.eye(3)]), reference=9)
    assert error.value.key == rr.ROI_REFERENCE_MISSING


# --- Piso de ruido y copias del GOP ---


def test_the_noise_floor_estimates_the_sigma_of_white_noise() -> None:
    noise = np.random.default_rng(1).normal(128, 6.0, (200, 200)).astype(np.float32)
    assert rr.noise_floor(noise) == pytest.approx(6.0, rel=0.1)


def test_identical_frames_collapse_into_one_copy_group() -> None:
    patch = texture()[:60, :80]
    assert rr.group_copies([patch, patch.copy(), patch + 0.01], threshold=1.0) == (0, 0, 0)


def test_frames_with_independent_noise_are_separate_samples() -> None:
    rng = np.random.default_rng(3)
    clean = texture()[:60, :80]
    patches = [clean + rng.normal(0, 5.0, clean.shape).astype(np.float32) for _ in range(4)]
    threshold = 0.5 * rr.noise_floor(patches[0])
    assert rr.group_copies(patches, threshold) == (0, 1, 2, 3)


def test_a_change_after_a_run_of_copies_starts_a_new_group() -> None:
    first = texture(1)[:60, :80]
    second = texture(2)[:60, :80]
    assert rr.group_copies([first, first, second, second, first], threshold=1.0) == (0, 0, 1, 1, 2)


def test_copies_of_the_reference_count_as_one_sample_with_the_reference() -> None:
    frames = frames_for([np.eye(3), np.eye(3), translation(0.4, 0.3), translation(0.4, 0.3)])
    result = register(frames, reference=1)
    statuses = [sample.status for sample in result.samples]
    assert statuses == ["copy", "reference", "accepted", "copy"]
    assert [sample.copy_group for sample in result.samples] == [0, 0, 1, 1]
    assert result.effective_samples == 2


# --- Registro ---


def test_a_known_subpixel_homography_is_recovered_within_a_tenth_of_a_pixel() -> None:
    truths = [
        np.eye(3),
        perspective(1.37, -0.62, 0.012, 2e-5, -1e-5),
        perspective(-2.21, 1.48, -0.018, -3e-5, 2e-5),
        perspective(0.53, 2.76, 0.006, 1e-5, 3e-5),
    ]
    result = register(frames_for(truths), motion="homography")
    for sample, truth in zip(result.samples[1:], truths[1:], strict=True):
        assert sample.status == "accepted"
        assert corner_error(sample.matrix, truth, PLATE_BOX) < MAX_CORNER_ERROR_PX
        assert sample.ecc > 0.95


def test_a_known_subpixel_affine_is_recovered_within_a_tenth_of_a_pixel() -> None:
    truths = [np.eye(3), affine(0.71, -1.33, 0.015, 1.01), affine(-1.46, 0.28, -0.01, 0.99)]
    result = register(frames_for(truths), motion="affine")
    for sample, truth in zip(result.samples[1:], truths[1:], strict=True):
        assert corner_error(sample.matrix, truth, PLATE_BOX) < MAX_CORNER_ERROR_PX


def test_a_subpixel_translation_is_reported_as_the_shift_of_the_roi_center() -> None:
    result = register(frames_for([np.eye(3), translation(2.35, -1.6)]), motion="translation")
    dx, dy = result.samples[1].shift
    assert dx == pytest.approx(2.35, abs=0.05)
    assert dy == pytest.approx(-1.6, abs=0.05)
    assert result.samples[0].shift == (0.0, 0.0)


def test_a_frame_that_does_not_match_is_rejected_and_listed() -> None:
    frames = frames_for([np.eye(3), translation(0.5, 0.25)])
    unrelated = rr.RoiFrame(2, "P", texture(seed=99))
    result = register((*frames, unrelated), motion="translation")
    assert result.samples[2].status == "rejected"
    assert result.rejected_frames == (2,)
    assert result.effective_samples == 2


def test_the_ecc_threshold_is_configurable() -> None:
    frames = frames_for([np.eye(3), translation(0.5, 0.25)])
    grain = np.random.default_rng(5).normal(0, 40, (HEIGHT, WIDTH)).astype(np.float32)
    noisy = rr.RoiFrame(1, "P", frames[1].pixels + grain)
    lenient = register((frames[0], noisy), motion="translation", ecc_min=0.3)
    strict = register((frames[0], noisy), motion="translation", ecc_min=0.999)
    assert lenient.samples[1].status == "accepted"
    assert strict.samples[1].status == "rejected"


def test_registration_is_deterministic() -> None:
    truths = [np.eye(3), perspective(1.1, 0.4, 0.01, 1e-5, 0), perspective(-0.7, 1.9, -0.01, 0, 2e-5)]
    first = register(frames_for(truths))
    second = register(frames_for(truths))
    assert first == second


def test_opencv_threads_are_restored_after_registering() -> None:
    before = cv2.getNumThreads()
    register(frames_for([np.eye(3), translation(0.5, 0.5)]), motion="translation")
    assert cv2.getNumThreads() == before


# --- Dispersion subpixel y nearCopies ---


def test_fractional_parts_wrap_around_the_pixel_boundary() -> None:
    spread = rr.subpixel_spread([(0.0, 0.0), (1.97, 3.02), (-2.03, 0.98)])
    assert spread.spread < 0.05


def test_varied_subpixel_phases_have_a_wide_spread() -> None:
    spread = rr.subpixel_spread([(0.0, 0.0), (0.25, 0.5), (1.5, 2.75), (0.75, 0.25)])
    assert spread.spread > 0.2
    assert spread.fractions_x == (0.0, 0.25, 0.5, 0.75)


def test_one_varied_axis_is_enough_to_carry_new_information() -> None:
    spread = rr.subpixel_spread([(0.0, 0.0), (1.0, 0.3), (2.0, 0.6), (3.0, 0.8)])
    assert spread.spread_x < 0.01
    assert spread.spread > 0.2


def test_fewer_than_three_effective_samples_are_near_copies() -> None:
    result = register(frames_for([np.eye(3), translation(0.4, 0.3)]), motion="translation")
    assert result.near_copies
    assert rr.NEAR_COPIES in result.warnings


def test_whole_pixel_shifts_share_one_phase_and_are_near_copies() -> None:
    matrices = [np.eye(3), translation(1, 0), translation(2, 1), translation(-1, 2), translation(3, -2)]
    result = register(frames_for(matrices), motion="translation")
    assert result.effective_samples == 5
    assert result.spread.spread < 0.1
    assert result.near_copies


def test_varied_subpixel_shifts_are_not_near_copies() -> None:
    matrices = [np.eye(3), translation(0.5, 0.25), translation(1.25, 0.75), translation(-0.75, 1.5)]
    result = register(frames_for(matrices), motion="translation")
    assert not result.near_copies
    assert result.warnings == ()


def test_the_result_serializes_to_camel_case_for_the_report() -> None:
    result = register(frames_for([np.eye(3), translation(0.5, 0.25)]), motion="translation")
    payload = result.to_json()
    assert payload["effectiveSamples"] == 2
    assert payload["nearCopies"] is True
    sample = payload["samples"][1]
    assert set(sample) == {"frame", "pictType", "copyGroup", "status", "ecc", "shift", "matrix"}
    assert payload["spread"]["spreadPx"] == result.spread.spread


# --- Cuadro de referencia sugerido ---


def blurred(frame: rr.RoiFrame, sigma: float) -> rr.RoiFrame:
    return rr.RoiFrame(frame.n, frame.pict_type, cv2.GaussianBlur(frame.pixels, (0, 0), sigma))


def saturated(frame: rr.RoiFrame, box: rr.RoiBox, share: float) -> rr.RoiFrame:
    pixels = frame.pixels.copy()
    rows = int(round(box.h * share))
    pixels[box.y : box.y + rows, box.x : box.x + box.w] = 255.0
    return rr.RoiFrame(frame.n, frame.pict_type, pixels)


def test_the_sharpest_roi_is_suggested_as_reference() -> None:
    base = frames_for([np.eye(3)] * 3)
    frames = (blurred(base[0], 2.0), base[1], blurred(base[2], 1.0))
    assert rr.suggest_reference_frame(frames, PLATE_BOX) == 1
    assert [candidate.n for candidate in rr.reference_candidates(frames, PLATE_BOX)] == [1, 2, 0]


def test_a_clipped_roi_loses_against_a_softer_unclipped_one() -> None:
    base = frames_for([np.eye(3)] * 2)
    frames = (saturated(base[0], PLATE_BOX, 0.1), blurred(base[1], 1.0))
    candidates = rr.reference_candidates(frames, PLATE_BOX)
    assert [candidate.n for candidate in candidates] == [1, 0]
    assert candidates[1].clipped
    assert candidates[1].saturated_fraction == pytest.approx(0.1, abs=0.01)


def test_with_every_roi_clipped_the_sharpest_is_still_suggested() -> None:
    base = frames_for([np.eye(3)] * 2)
    frames = (saturated(blurred(base[0], 2.0), PLATE_BOX, 0.2), saturated(base[1], PLATE_BOX, 0.2))
    assert rr.suggest_reference_frame(frames, PLATE_BOX) == 1


def test_suggesting_a_reference_needs_frames() -> None:
    with pytest.raises(rr.RoiRegistrationError):
        rr.suggest_reference_frame((), PLATE_BOX)


def test_suggesting_a_reference_needs_the_box_inside_the_frame() -> None:
    with pytest.raises(rr.RoiRegistrationError) as error:
        rr.suggest_reference_frame(frames_for([np.eye(3)]), rr.RoiBox(300, 0, 40, 40))
    assert error.value.key == rr.ROI_OUTSIDE_FRAME


# --- Con ffmpeg real: clip x264 estatico con GOP largo ---


CLIP_FRAMES = 40
LONG_GOP_X264 = ["-c:v", "libx264", "-threads", "1", "-x264-params", "threads=1", "-g", "250", "-bf", "0"]


def ffmpeg_command(*args: str) -> list[str]:
    return [str(Settings().ffmpeg_binary_path), "-hide_banner", "-v", "error", "-y", *args]


def make_static_clip(tmp_path: Path) -> Path:
    still = tmp_path / "still.png"
    noise = np.random.default_rng(11).normal(0, 6.0, (HEIGHT, WIDTH)).astype(np.float32)
    cv2.imwrite(str(still), np.clip(texture() + noise, 0, 255).astype(np.uint8))
    clip = tmp_path / "static.mp4"
    command = ffmpeg_command(
        "-loop", "1", "-framerate", "25", "-i", str(still), "-frames:v", str(CLIP_FRAMES),
        *LONG_GOP_X264, "-crf", "23", "-pix_fmt", "yuv420p", str(clip),
    )  # fmt: skip
    subprocess.run(command, check=True, capture_output=True)
    return clip


def make_noisy_static_clip(tmp_path: Path) -> Path:
    # Ruido de sensor distinto en cada cuadro, pero a bajo bitrate como un DVR.
    rng = np.random.default_rng(2)
    scene = texture()
    noisy_frames = (np.clip(scene + rng.normal(0, 4.0, scene.shape), 0, 255) for _ in range(CLIP_FRAMES))
    raw = b"".join(frame.astype(np.uint8).tobytes() for frame in noisy_frames)
    clip = tmp_path / "noisy.mp4"
    command = ffmpeg_command(
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{WIDTH}x{HEIGHT}", "-r", "25", "-i", "-",
        *LONG_GOP_X264, "-b:v", "60k", "-pix_fmt", "yuv420p", str(clip),
    )  # fmt: skip
    subprocess.run(command, input=raw, check=True, capture_output=True)
    return clip


def frame_index(clip: Path):
    command = build_frame_index_command(Settings().ffprobe_binary_path, clip)
    return parse_frame_index(subprocess.run(command, check=True, capture_output=True).stdout.decode())


@needs_ffmpeg
async def test_a_static_x264_clip_with_a_long_gop_is_near_copies(tmp_path: Path) -> None:
    clip = make_static_clip(tmp_path)
    index = frame_index(clip)
    stack = await roi_frames.decode_roi_frames(Settings().ffmpeg_binary_path, clip, 5, 34, WIDTH, HEIGHT)
    frames = roi_frames.roi_frames_from(stack, 5, index)

    result = register(frames, reference=20)

    assert len(result.samples) == 30
    assert {sample.pict_type for sample in result.samples} == {"P"}
    assert result.effective_samples < 3
    assert result.near_copies
    assert rr.NEAR_COPIES in result.warnings


@needs_ffmpeg
async def test_a_static_scene_with_sensor_noise_shares_one_subpixel_phase(tmp_path: Path) -> None:
    clip = make_noisy_static_clip(tmp_path)
    stack = await roi_frames.decode_roi_frames(Settings().ffmpeg_binary_path, clip, 5, 34, WIDTH, HEIGHT)
    frames = roi_frames.roi_frames_from(stack, 5, frame_index(clip))

    result = register(frames, reference=20)

    assert result.spread.spread < 0.1
    assert result.near_copies
    assert rr.NEAR_COPIES in result.warnings
