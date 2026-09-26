from __future__ import annotations

import csv
import hashlib
import io
from fractions import Fraction

import cv2
import numpy as np
import pytest

from app.services import roi_fusion as rf
from app.services import roi_registration as rr

HEIGHT, WIDTH = 240, 320
BOX = rr.RoiBox(120, 90, 80, 40)
NOISE_SIGMA = 8.0
SHIFTS = [
    (0.0, 0.0), (0.37, -0.21), (-0.64, 0.48), (1.13, 0.71), (-0.29, -0.83), (0.52, 1.36),
    (-1.18, 0.14), (0.81, -0.57), (-0.46, -1.24), (1.42, -0.33), (0.09, 0.93), (-0.92, 1.07),
]  # fmt: skip


def texture(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0, 255, (HEIGHT, WIDTH)).astype(np.float32)
    return np.clip(cv2.GaussianBlur(noise, (0, 0), 2.0) * 3.0 - 255.0, 0, 240).astype(np.float32)


def translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float64)


def render(base: np.ndarray, reference_to_frame: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(
        base, reference_to_frame, (WIDTH, HEIGHT), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT
    )


def noisy_frames(shifts=SHIFTS, sigma: float = NOISE_SIGMA, seed: int = 11) -> tuple[rr.RoiFrame, ...]:
    rng = np.random.default_rng(seed)
    base = texture()
    return tuple(
        rr.RoiFrame(n, "P", render(base, translation(dx, dy)) + rng.normal(0, sigma, base.shape).astype(np.float32))
        for n, (dx, dy) in enumerate(shifts)
    )


def register(frames, motion: rr.Motion = "affine") -> rr.RegistrationResult:
    return rr.register_roi_frames(frames, 0, BOX, motion)


def identity() -> rr.Matrix:
    return rr.as_matrix(np.eye(3))


def snr_db(image: np.ndarray, truth: np.ndarray) -> float:
    inner = (slice(8, -8), slice(8, -8))
    error = image[inner] - truth[inner]
    return float(10 * np.log10(np.var(truth[inner]) / np.mean(error**2)))


def fused_bytes(fusion: rf.RoiFusion) -> bytes:
    return rf.png_bytes(rf.to_uint16(fusion.fused)) + rf.png_bytes(fusion.agreement) + rf.stack_gray16le(fusion.stack)


# --- Fusion ---


@pytest.mark.parametrize("method", ["median", "trimmed_mean"])
def test_with_independent_noise_the_fused_roi_beats_the_best_single_frame(method: str) -> None:
    frames = noisy_frames()
    registration = register(frames)
    assert registration.effective_samples == len(SHIFTS)
    fusion = rf.fuse_roi(frames, registration, BOX, rf.FusionSettings(scale=2, method=method))

    truth = rf.warp_to_grid(texture(), identity(), BOX, 2, "affine")
    singles = [snr_db(layer, truth) for layer in fusion.stack]
    assert snr_db(fusion.fused, truth) > max(singles) + 3.0


def test_the_fused_grid_is_the_roi_times_the_scale() -> None:
    frames = noisy_frames(SHIFTS[:4])
    fusion = rf.fuse_roi(frames, register(frames), BOX, rf.FusionSettings(scale=3))
    assert fusion.fused.shape == (BOX.h * 3, BOX.w * 3)
    assert fusion.reference_nearest.shape == (BOX.h * 3, BOX.w * 3)
    assert fusion.agreement.shape == (BOX.h * 3, BOX.w * 3, 3)


def test_the_identity_warp_of_the_reference_matches_a_lanczos_upscale_of_the_roi() -> None:
    base = texture()
    warped = rf.warp_to_grid(base, identity(), BOX, 2, "homography")
    resized = cv2.resize(BOX.crop(base), (BOX.w * 2, BOX.h * 2), interpolation=cv2.INTER_LANCZOS4)
    inner = (slice(6, -6), slice(6, -6))
    assert np.max(np.abs(warped[inner] - np.clip(resized, 0, 255)[inner])) < 0.5


def test_a_known_shift_is_undone_on_the_grid() -> None:
    base = texture()
    shifted = render(base, translation(1.5, -0.75))
    warped = rf.warp_to_grid(shifted, rr.as_matrix(translation(1.5, -0.75)), BOX, 2, "affine")
    truth = rf.warp_to_grid(base, identity(), BOX, 2, "affine")
    inner = (slice(8, -8), slice(8, -8))
    assert np.mean(np.abs(warped[inner] - truth[inner])) < 1.0


def test_the_median_ignores_one_wild_frame() -> None:
    stack = np.stack([np.full((2, 2), 10.0), np.full((2, 2), 12.0), np.full((2, 2), 250.0)])
    assert np.array_equal(rf.median_fuse(stack), np.full((2, 2), 12.0))


def test_the_trimmed_mean_drops_a_tenth_from_each_end() -> None:
    values = [0.0, *[10.0] * 8, 255.0]
    stack = np.array(values).reshape(10, 1, 1)
    assert rf.trim_per_side(10, rf.TRIMMED_SHARE) == 1
    assert rf.trimmed_mean_fuse(stack)[0, 0] == pytest.approx(10.0)


def test_the_trimmed_mean_of_a_few_frames_is_the_plain_mean() -> None:
    stack = np.array([0.0, 30.0, 60.0]).reshape(3, 1, 1)
    assert rf.trimmed_mean_fuse(stack)[0, 0] == pytest.approx(30.0)


# --- Copias del GOP y mapa de acuerdo ---


def test_copies_of_the_gop_are_left_out_of_the_fusion_and_the_agreement_map() -> None:
    unique = noisy_frames(SHIFTS[:4])
    copies = tuple(rr.RoiFrame(4 + i, "P", unique[3].pixels.copy()) for i in range(6))
    frames = (*unique, *copies)
    registration = register(frames)

    assert [sample.status for sample in registration.samples[4:]] == ["copy"] * 6
    fusion = rf.fuse_roi(frames, registration, BOX, rf.FusionSettings())
    assert fusion.stack_frames == (0, 1, 2, 3)
    unique_only = rf.fuse_roi(unique, register(unique), BOX, rf.FusionSettings())
    assert np.array_equal(fusion.mad, unique_only.mad)
    assert np.array_equal(fusion.fused, unique_only.fused)


def test_a_rejected_frame_is_left_out() -> None:
    frames = (*noisy_frames(SHIFTS[:3]), rr.RoiFrame(3, "P", texture(seed=99)))
    registration = register(frames)
    fusion = rf.fuse_roi(frames, registration, BOX, rf.FusionSettings())
    assert registration.rejected_frames == (3,)
    assert fusion.stack_frames == (0, 1, 2)


def test_the_agreement_map_is_green_where_frames_agree_and_red_where_they_do_not() -> None:
    mad = np.array([[0.0, rf.AGREEMENT_FULL_SCALE / 2, rf.AGREEMENT_FULL_SCALE * 3]])
    image = rf.agreement_image(mad)
    assert image[0, 0].tolist() == [0, 255, 0]
    assert image[0, 1].tolist() == [0, 255, 255]
    assert image[0, 2].tolist() == [0, 0, 255]


def test_the_agreement_mad_is_the_median_absolute_deviation_per_pixel() -> None:
    stack = np.array([1.0, 2.0, 3.0, 4.0, 100.0]).reshape(5, 1, 1)
    assert rf.agreement_mad(stack)[0, 0] == pytest.approx(1.0)


# --- Determinismo y salidas ---


def test_two_runs_give_the_same_bytes() -> None:
    frames = noisy_frames(SHIFTS[:6])
    first = rf.fuse_roi(frames, register(frames), BOX, rf.FusionSettings(method="trimmed_mean"))
    second = rf.fuse_roi(frames, register(frames), BOX, rf.FusionSettings(method="trimmed_mean"))
    assert hashlib.sha256(fused_bytes(first)).digest() == hashlib.sha256(fused_bytes(second)).digest()


def test_the_fused_png_is_16_bit_and_the_reference_is_nearest_neighbor() -> None:
    frames = noisy_frames(SHIFTS[:3])
    fusion = rf.fuse_roi(frames, register(frames), BOX, rf.FusionSettings(scale=2))
    decoded = cv2.imdecode(np.frombuffer(rf.png_bytes(rf.to_uint16(fusion.fused)), np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.dtype == np.uint16 and decoded.shape == (BOX.h * 2, BOX.w * 2)
    reference = np.clip(np.round(BOX.crop(frames[0].pixels)), 0, 255).astype(np.uint8)
    assert np.array_equal(fusion.reference_nearest[::2, ::2], reference)
    assert np.array_equal(fusion.reference_nearest[1::2, 1::2], reference)


def test_the_stack_is_raw_gray16le_one_layer_per_effective_sample() -> None:
    stack = np.array([[[0.0, 255.0]], [[1.0, 128.0]]])
    raw = rf.stack_gray16le(stack)
    assert len(raw) == 2 * 2 * 2
    assert np.frombuffer(raw, "<u2").tolist() == [0, 65535, 257, 32896]


def test_the_samples_csv_lists_frame_type_group_ecc_and_shift() -> None:
    samples = (
        rr.RoiSample(10, "I", 0, "reference", 1.0, (0.0, 0.0), identity()),
        rr.RoiSample(11, "P", 0, "copy", 1.0, (0.0, 0.0), identity()),
        rr.RoiSample(12, "P", 1, "rejected", 0.41, None, None),
    )
    rows = list(csv.reader(io.StringIO(rf.samples_csv_text(samples))))
    assert rows[0] == list(rf.SAMPLES_HEADER)
    assert rows[1] == ["10", "I", "0", "reference", "1.000000", "0.000000", "0.000000"]
    assert rows[3] == ["12", "P", "1", "rejected", "0.410000", "", ""]


def test_output_names_carry_the_scale() -> None:
    assert rf.output_names(3)["fused"] == "roi_fused_x3.png"
    assert rf.output_names(3)["reference"] == "roi_reference_nearest_x3.png"


# --- Avisos ---


def test_face_density_is_measured_in_stored_pixels_even_when_the_display_is_twice_as_wide() -> None:
    density = rf.roi_density("face_or_object", rr.RoiBox(0, 0, 36, 48), Fraction(2))
    assert (density.stored_px, density.display_px) == (36, 72)
    assert rf.density_notices(density) == (rf.RoiNotice(rf.DENSITY_FACE, {"px": 36}),)


def test_a_face_40_stored_pixels_wide_has_no_density_notice() -> None:
    assert rf.density_notices(rf.roi_density("face_or_object", rr.RoiBox(0, 0, 40, 48), Fraction(1))) == ()


def test_a_plate_always_reports_its_stored_height() -> None:
    density = rf.roi_density("plate", rr.RoiBox(0, 0, 120, 18), Fraction(2))
    assert density.axis == "height"
    assert rf.density_notices(density) == (rf.RoiNotice(rf.DENSITY_PLATE, {"px": 18}),)


def test_clipping_counts_frames_with_more_than_a_fifth_of_the_roi_saturated() -> None:
    box = rr.RoiBox(0, 0, 10, 10)
    dark = np.zeros((10, 10), dtype=np.float32)
    fifth = dark.copy()
    fifth[:2] = 255.0
    third = dark.copy()
    third[:3] = 250.0
    frames = [rr.RoiFrame(n, "P", pixels) for n, pixels in enumerate([dark, fifth, third, third])]
    pct = rf.clipped_frames_pct(frames, box)
    assert pct == 50.0
    assert rf.clipped_notices(pct) == (rf.RoiNotice(rf.CLIPPED, {"pct": 50.0}),)
    assert rf.clipped_notices(0.0) == ()


def test_frames_used_counts_copies_but_not_rejected_frames() -> None:
    unique = noisy_frames(SHIFTS[:3])
    frames = (*unique, rr.RoiFrame(3, "P", unique[2].pixels.copy()), rr.RoiFrame(4, "P", texture(seed=99)))
    registration = register(frames)
    notice = rf.frames_used_notice(registration)
    assert notice == rf.RoiNotice(rf.FRAMES_USED, {"used": 4, "total": 5, "effective": 3})


def test_near_copies_are_announced_after_frames_used() -> None:
    frames = tuple(rr.RoiFrame(n, "P", texture()) for n in range(4))
    registration = register(frames)
    notices = rf.fusion_notices(registration, rf.roi_density("plate", BOX, Fraction(1)), 0.0)
    assert [notice.key for notice in notices] == [rf.FRAMES_USED, rr.NEAR_COPIES, rf.DENSITY_PLATE]
    assert notices[0].to_json() == {"key": rf.FRAMES_USED, "params": {"used": 4, "total": 4, "effective": 1}}
