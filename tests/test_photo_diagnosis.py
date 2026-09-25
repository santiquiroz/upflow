from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.services.photo_diagnosis import (
    BLUR_MAX_SHARPNESS,
    DEFAULT_STEP_COSTS,
    FACE_ANALYSIS_SIDE,
    FACE_MIN_SCORE,
    Blocking,
    DetectedFace,
    Finding,
    PhotoDiagnosis,
    StepCost,
    analyze_pattern,
    classify_tone,
    damage_stats,
    detect_faces,
    diagnose_photo,
    estimate_noise_sigma,
    estimate_restore_time,
    measure_blocking,
    measure_color_cast,
    measure_sharpness,
)
from app.services.photo_restore_chain import RESTORE_CHAIN, step_ids
from app.services.photo_restore_presets import FADED_CAST_MIN_DE, NOISE_MIN_SIGMA, resolve_preset


def smooth_field(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    coarse = rng.random((cells, cells), dtype=np.float32)
    return np.clip(cv2.resize(coarse, (size, size), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)


def natural_scene(seed: int, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    hue = smooth_field(rng, size, 6) * 360.0
    saturation = smooth_field(rng, size, 5) ** 2 * 0.8
    value = 0.08 + 0.87 * smooth_field(rng, size, 7)
    hsv = np.stack([hue, saturation, value], axis=-1).astype(np.float32)
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0.0, 1.0)


def as_gray(rgb: np.ndarray) -> np.ndarray:
    return np.repeat(rgb.mean(axis=-1, keepdims=True), 3, axis=-1).astype(np.float32)


def gray_scene(seed: int, size: int = 256) -> np.ndarray:
    return as_gray(natural_scene(seed, size))


def detailed_scene(seed: int, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed + 100)
    scene = natural_scene(seed, size).copy()
    for _ in range(40):
        x0, y0 = (int(value) for value in rng.integers(0, size - 8, 2))
        x1, y1 = (int(value) for value in rng.integers(8, 64, 2))
        color = tuple(float(value) for value in rng.uniform(0.05, 0.95, 3))
        cv2.rectangle(scene, (x0, y0), (x0 + x1, y0 + y1), color, thickness=-1)
    return scene


def lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2Lab)


def from_lab(lab_image: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(lab_image.astype(np.float32), cv2.COLOR_Lab2RGB), 0.0, 1.0)


def shift_chroma(rgb: np.ndarray, a: float, b: float, where: np.ndarray | None = None) -> np.ndarray:
    lab_image = lab(rgb)
    region = np.ones(rgb.shape[:2], bool) if where is None else where
    lab_image[region, 1] += a
    lab_image[region, 2] += b
    return from_lab(lab_image)


def sepia(rgb: np.ndarray) -> np.ndarray:
    return shift_chroma(as_gray(rgb), 6.0, 18.0)


def disk(size: int, fraction: float) -> np.ndarray:
    rows, cols = np.mgrid[:size, :size]
    radius = size * np.sqrt(fraction / np.pi)
    return np.hypot(rows - size * 0.4, cols - size * 0.55) < radius


def hand_tinted(base: np.ndarray, fraction: float = 0.12) -> np.ndarray:
    return shift_chroma(base, 28.0, 14.0, disk(base.shape[0], fraction))


def with_noise(rgb: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    grain = np.random.default_rng(seed).standard_normal(rgb.shape[:2]).astype(np.float32) * sigma
    return (rgb + grain[..., None]).astype(np.float32)


def jpeg_roundtrip(rgb: np.ndarray, quality: int) -> np.ndarray:
    bgr = cv2.cvtColor(np.round(rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def periodic_texture(photo: np.ndarray, period: float = 7.0, levels: float = 18.0) -> np.ndarray:
    rows, cols = np.mgrid[: photo.shape[0], : photo.shape[1]].astype(np.float32)
    texture = sum(
        np.cos(2.0 * np.pi * (cols * np.cos(angle) + rows * np.sin(angle)) / period)
        for angle in np.deg2rad([0.0, 60.0, 120.0])
    )
    return np.clip(photo + (levels / 255.0 / 3.0) * texture[..., None], 0.0, 1.0).astype(np.float32)


def am_halftone(photo: np.ndarray, period: float = 5.0) -> np.ndarray:
    rows, cols = np.mgrid[: photo.shape[0], : photo.shape[1]].astype(np.float32)
    u, v = (cols + rows) / np.sqrt(2.0), (cols - rows) / np.sqrt(2.0)
    screen = 0.5 + 0.25 * (np.cos(2.0 * np.pi * u / period) + np.cos(2.0 * np.pi * v / period))
    return (photo > screen[..., None]).astype(np.float32)


def finding(diagnosis: PhotoDiagnosis, key: str) -> Finding:
    return next(item for item in diagnosis.findings if item.key == key)


def finding_keys(diagnosis: PhotoDiagnosis) -> set[str]:
    return {item.key for item in diagnosis.findings}


def proposals(item: Finding) -> dict:
    return {proposal.step_id: proposal for proposal in item.proposes}


class FakeDamageDetector:
    def __init__(self, coverage: float, shape: tuple[int, int] = (64, 80)) -> None:
        self.coverage = coverage
        self.shape = shape
        self.calls = 0

    def __call__(self, rgb: np.ndarray) -> np.ndarray:
        self.calls += 1
        probability = np.zeros(self.shape, np.float32)
        probability.reshape(-1)[: int(round(self.coverage * probability.size))] = 0.9
        return probability


class FakeFaceDetector:
    def __init__(self, faces: list[DetectedFace]) -> None:
        self.faces = faces
        self.shapes: list[tuple[int, ...]] = []

    def __call__(self, rgb: np.ndarray) -> list[DetectedFace]:
        self.shapes.append(rgb.shape)
        return self.faces


@pytest.mark.parametrize("seed", range(3))
def test_tone_a_gray_photo_looks_black_and_white(seed: int) -> None:
    assert classify_tone(gray_scene(seed)).kind == "mono"


@pytest.mark.parametrize("seed", range(3))
def test_tone_a_sepia_print_looks_toned(seed: int) -> None:
    assert classify_tone(sepia(natural_scene(seed))).kind == "toned"


@pytest.mark.parametrize("seed", range(3))
def test_tone_a_color_photo_is_color(seed: int) -> None:
    assert classify_tone(natural_scene(seed)).kind == "color"


@pytest.mark.parametrize("base", ["gray", "sepia"])
def test_tone_localized_color_over_a_neutral_or_toned_print_is_hand_tinted(base: str) -> None:
    photo = gray_scene(4) if base == "gray" else sepia(natural_scene(4))

    analysis = classify_tone(hand_tinted(photo))

    assert analysis.kind == "hand_tinted"
    assert 0.05 < analysis.tinted_fraction <= 0.25


def test_tone_a_colored_area_over_a_quarter_of_the_photo_is_not_hand_tinted() -> None:
    assert classify_tone(hand_tinted(gray_scene(4), fraction=0.45)).kind == "color"


def test_tone_mono_finding_keeps_the_tone_and_only_suggests_colorize() -> None:
    tone = finding(diagnose_photo(gray_scene(1)), "tone")

    assert tone.reason_key == "restore.diag.mono"
    assert tone.value == "mono"
    offered = proposals(tone)
    assert offered["tone"].enabled and offered["tone"].options["keep_tone"] is True
    assert offered["tone"].options["fix_faded"] is False
    assert offered["colorize"].enabled is False


def test_tone_sepia_finding_names_the_toning() -> None:
    assert finding(diagnose_photo(sepia(natural_scene(1))), "tone").reason_key == "restore.diag.sepia"


def test_tone_hand_tinted_leaves_fix_faded_colors_off() -> None:
    diagnosis = diagnose_photo(hand_tinted(gray_scene(4)))

    tone = finding(diagnosis, "tone")
    assert tone.reason_key == "restore.diag.handTinted"
    assert tone.proposes == ()
    assert "faded" not in finding_keys(diagnosis)
    assert "tone" not in resolve_preset("gentle", diagnosis.facts).steps


def test_tone_color_finding_proposes_nothing_by_itself() -> None:
    tone = finding(diagnose_photo(natural_scene(2)), "tone")

    assert tone.reason_key == "restore.diag.color"
    assert tone.proposes == ()


@pytest.mark.parametrize("sigma255", [4.0, 8.0, 15.0])
def test_noise_sigma_is_within_twenty_percent(sigma255: float) -> None:
    noisy = with_noise(gray_scene(3), sigma255 / 255.0)

    assert estimate_noise_sigma(noisy) == pytest.approx(sigma255 / 255.0, rel=0.2)


def test_noise_a_clean_photo_stays_under_the_threshold() -> None:
    assert estimate_noise_sigma(natural_scene(3)) < NOISE_MIN_SIGMA


def test_noise_is_measured_at_native_resolution_on_a_large_scan() -> None:
    noisy = with_noise(gray_scene(3, size=2400), 8.0 / 255.0)

    assert estimate_noise_sigma(noisy) == pytest.approx(8.0 / 255.0, rel=0.2)


def test_noise_finding_proposes_denoise_with_strength_growing_with_sigma() -> None:
    light = finding(diagnose_photo(with_noise(gray_scene(3), 4.0 / 255.0)), "noise")
    heavy = finding(diagnose_photo(with_noise(gray_scene(3), 15.0 / 255.0)), "noise")

    light_denoise, heavy_denoise = proposals(light)["denoise"], proposals(heavy)["denoise"]
    assert 0.0 < light_denoise.options["strength"] < heavy_denoise.options["strength"] <= 1.0
    assert light_denoise.options["keep_grain"] == pytest.approx(0.25)
    assert "noise" not in finding_keys(diagnose_photo(natural_scene(3)))


@pytest.mark.parametrize("seed", range(3))
def test_blocking_jpeg_q20_is_blocky_and_q95_is_not(seed: int) -> None:
    scene = with_noise(detailed_scene(seed), 2.0 / 255.0, seed)

    assert measure_blocking(jpeg_roundtrip(scene, 20)).visible
    assert not measure_blocking(jpeg_roundtrip(scene, 95)).visible


def test_blocking_survives_a_crop_that_shifts_the_grid() -> None:
    blocky = jpeg_roundtrip(detailed_scene(1), 20)[3:, 5:]

    assert measure_blocking(blocky).visible


def test_blocking_fractional_steps_of_a_high_quality_jpeg_are_not_visible() -> None:
    assert not Blocking(ratio=2.4, step=0.14).visible
    assert Blocking(ratio=1.6, step=0.9).visible


def test_blocking_finding_needs_a_jpeg_origin() -> None:
    blocky = jpeg_roundtrip(detailed_scene(1), 20)

    from_jpeg = finding(diagnose_photo(blocky, jpeg_origin=True), "jpeg_blocking")
    assert from_jpeg.reason_key == "restore.diag.jpeg"
    assert proposals(from_jpeg)["deblock"].enabled
    assert "jpeg_blocking" not in finding_keys(diagnose_photo(blocky, jpeg_origin=False))


def test_pattern_paper_texture_is_texture() -> None:
    pattern = analyze_pattern(periodic_texture(gray_scene(3)))

    assert pattern.kind == "texture"


@pytest.mark.parametrize("period", [5.0, 7.0])
def test_pattern_binary_dot_screen_is_halftone_with_its_period(period: float) -> None:
    pattern = analyze_pattern(am_halftone(gray_scene(3), period))

    assert pattern.kind == "halftone"
    assert pattern.period == pytest.approx(period, rel=0.1)


@pytest.mark.parametrize("seed", range(3))
def test_pattern_a_clean_photo_has_none(seed: int) -> None:
    assert analyze_pattern(natural_scene(seed)).kind is None


def test_pattern_findings_propose_descreen_in_the_matching_mode() -> None:
    halftone = finding(diagnose_photo(am_halftone(gray_scene(3))), "pattern")
    texture = finding(diagnose_photo(periodic_texture(gray_scene(3))), "pattern")

    assert halftone.reason_key == "restore.diag.halftone"
    assert proposals(halftone)["descreen"].options["mode"] == "halftone"
    assert proposals(halftone)["descreen"].options["period"] == pytest.approx(5.0, rel=0.1)
    assert texture.reason_key == "restore.diag.texture"
    assert proposals(texture)["descreen"].options["mode"] == "texture"


@pytest.mark.parametrize("seed", range(3))
def test_blur_a_blurred_photo_scores_under_the_threshold(seed: int) -> None:
    sharp = detailed_scene(seed)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 3.0)

    assert measure_sharpness(sharp) > BLUR_MAX_SHARPNESS
    assert measure_sharpness(blurred) < BLUR_MAX_SHARPNESS


def test_blur_is_informative_only() -> None:
    blurred = cv2.GaussianBlur(detailed_scene(1), (0, 0), 3.0)

    soft = finding(diagnose_photo(blurred), "blur")
    assert soft.reason_key == "restore.diag.blur"
    assert soft.proposes == ()
    assert "blur" not in finding_keys(diagnose_photo(detailed_scene(1)))


def test_blur_a_flat_photo_is_not_reported_as_blurry() -> None:
    assert measure_sharpness(np.full((64, 64, 3), 0.5, np.float32)) > BLUR_MAX_SHARPNESS


def test_damage_coverage_uses_the_probability_threshold() -> None:
    probability = np.array([[0.39, 0.41], [0.9, 0.0]], np.float32)

    assert damage_stats(probability, (2, 2)).coverage == pytest.approx(0.5)


def test_damage_large_areas_are_measured_at_native_resolution() -> None:
    probability = np.zeros((100, 100), np.float32)
    probability[10:12, 10:90] = 1.0
    probability[40:60, 40:60] = 1.0

    at_native = damage_stats(probability, (100, 100))
    upscaled = damage_stats(probability, (1000, 1000))

    assert at_native.large_areas == 0
    assert upscaled.large_areas == 1


def test_damage_finding_reports_the_percentage_and_proposes_repair() -> None:
    detector = FakeDamageDetector(0.05)

    diagnosis = diagnose_photo(natural_scene(1), damage_detector=detector)

    damage = finding(diagnosis, "damage")
    assert detector.calls == 1
    assert damage.reason_key == "restore.diag.damage"
    assert damage.params["pct"] == pytest.approx(5.0, abs=0.1)
    assert proposals(damage)["repair"].enabled
    assert diagnosis.facts.damage_coverage == pytest.approx(0.05, abs=0.001)
    assert diagnosis.proposed_preset == "heavy_damage"


def test_damage_under_the_threshold_is_not_reported() -> None:
    diagnosis = diagnose_photo(natural_scene(1), damage_detector=FakeDamageDetector(0.001))

    assert "damage" not in finding_keys(diagnosis)


def test_damage_without_the_pack_says_which_pack_is_missing() -> None:
    damage = finding(diagnose_photo(natural_scene(1)), "damage")

    assert damage.missing_pack == "restore-core"
    assert damage.reason_key == "restore.diag.packMissing"
    assert damage.message
    assert damage.proposes == ()


def test_faces_run_on_a_copy_of_at_most_1280_and_come_back_in_photo_pixels() -> None:
    photo = natural_scene(1, size=2560)
    detector = FakeFaceDetector(
        [
            DetectedFace(box=(100.0, 100.0, 200.0, 220.0), score=0.95, eye_px=20.0),
            DetectedFace(box=(400.0, 400.0, 440.0, 450.0), score=0.5, eye_px=8.0),
            DetectedFace(box=(600.0, 100.0, 900.0, 460.0), score=0.99, eye_px=60.0),
        ]
    )

    faces = detect_faces(photo, detector)

    assert max(detector.shapes[0][:2]) == FACE_ANALYSIS_SIDE
    assert [face.eye_px for face in faces] == [pytest.approx(120.0), pytest.approx(40.0)]
    assert faces[1].box == pytest.approx((200.0, 200.0, 400.0, 440.0))
    assert all(face.score >= FACE_MIN_SCORE for face in faces)


def test_faces_a_small_photo_is_not_enlarged() -> None:
    detector = FakeFaceDetector([])

    detect_faces(natural_scene(1, size=300), detector)

    assert detector.shapes[0][:2] == (300, 300)


def test_faces_finding_suggests_portrait_but_never_turns_faces_on() -> None:
    detector = FakeFaceDetector([DetectedFace(box=(10.0, 10.0, 90.0, 110.0), score=0.97, eye_px=40.0)])

    diagnosis = diagnose_photo(natural_scene(1), face_detector=detector)

    faces = finding(diagnosis, "faces")
    assert faces.reason_key == "restore.diag.faces"
    assert faces.params["count"] == 1
    assert proposals(faces)["faces"].enabled is False
    assert diagnosis.facts.max_eye_px == pytest.approx(40.0)
    assert "portrait" in diagnosis.suggested_presets
    assert diagnosis.proposed_preset != "portrait"
    assert len(diagnosis.faces) == 1


def test_faces_without_the_pack_say_which_pack_is_missing() -> None:
    faces = finding(diagnose_photo(natural_scene(1)), "faces")

    assert faces.missing_pack == "restore-faces"
    assert faces.message


@pytest.mark.parametrize(
    ("a", "b", "name"),
    [(10.0, 0.0, "reddish"), (0.0, 12.0, "yellow"), (-8.0, -8.0, "cyan/green"), (0.0, -12.0, "blue")],
)
def test_faded_cast_is_measured_and_named(a: float, b: float, name: str) -> None:
    cast = measure_color_cast(shift_chroma(natural_scene(5), a, b))

    assert cast.delta_e > FADED_CAST_MIN_DE
    assert cast.name == name


def test_faded_an_unfaded_color_photo_has_a_small_cast() -> None:
    assert measure_color_cast(natural_scene(5)).delta_e < FADED_CAST_MIN_DE


def test_faded_finding_proposes_fix_faded_colors_with_the_cast_named() -> None:
    diagnosis = diagnose_photo(shift_chroma(natural_scene(5), 10.0, 0.0))

    faded = finding(diagnosis, "faded")
    assert faded.reason_key == "restore.diag.faded"
    assert faded.params["cast"] == "reddish"
    tone = proposals(faded)["tone"]
    assert tone.enabled and tone.options["fix_faded"] is True
    assert diagnosis.facts.color_cast_de > FADED_CAST_MIN_DE


def test_faded_is_not_reported_for_a_sepia_print() -> None:
    assert "faded" not in finding_keys(diagnose_photo(sepia(natural_scene(5))))


def test_time_estimate_adds_megapixels_times_cost_and_per_item_costs() -> None:
    costs = {
        "denoise": StepCost(gpu_seconds_per_mpx=1.0, cpu_seconds_per_mpx=10.0),
        "faces": StepCost(0.0, 0.0, gpu_seconds_per_item=0.5, cpu_seconds_per_item=2.0),
    }

    estimate = estimate_restore_time(4.0, ("denoise", "faces"), costs, items={"faces": 3})

    assert estimate.gpu_seconds == pytest.approx(4.0 + 1.5)
    assert estimate.cpu_seconds == pytest.approx(40.0 + 6.0)


def test_time_estimate_of_no_steps_is_zero() -> None:
    estimate = estimate_restore_time(12.0, ())

    assert (estimate.gpu_seconds, estimate.cpu_seconds) == (0.0, 0.0)


def test_time_estimate_rejects_a_step_without_a_cost() -> None:
    with pytest.raises(ValueError, match="denoise"):
        estimate_restore_time(1.0, ("denoise",), costs={})


def test_time_estimate_every_chain_step_has_a_default_cost_and_gpu_is_not_slower() -> None:
    assert set(DEFAULT_STEP_COSTS) == set(step_ids(RESTORE_CHAIN))
    for cost in DEFAULT_STEP_COSTS.values():
        assert cost.gpu_seconds_per_mpx <= cost.cpu_seconds_per_mpx
        assert cost.gpu_seconds_per_item <= cost.cpu_seconds_per_item


def test_time_estimate_follows_the_proposed_steps() -> None:
    clean = diagnose_photo(natural_scene(1))
    noisy = diagnose_photo(with_noise(gray_scene(1, size=512), 12.0 / 255.0))

    assert clean.estimate.cpu_seconds == pytest.approx(0.0)
    assert "denoise" in noisy.proposed_steps
    assert noisy.estimate.cpu_seconds > noisy.estimate.gpu_seconds > 0.0


def test_diagnosis_proposes_the_newspaper_preset_for_a_halftone() -> None:
    assert diagnose_photo(am_halftone(gray_scene(3))).proposed_preset == "newspaper"


def test_diagnosis_of_a_clean_photo_proposes_gentle_with_nothing_on() -> None:
    diagnosis = diagnose_photo(natural_scene(1))

    assert diagnosis.proposed_preset == "gentle"
    assert diagnosis.proposed_steps == ()


def test_diagnosis_does_not_mutate_the_photo() -> None:
    photo = with_noise(hand_tinted(gray_scene(2)), 5.0 / 255.0)
    snapshot = photo.copy()

    diagnose_photo(photo, jpeg_origin=True, damage_detector=FakeDamageDetector(0.02))

    np.testing.assert_array_equal(photo, snapshot)


def test_diagnosis_rejects_an_image_without_three_channels() -> None:
    with pytest.raises(ValueError, match="3 channels"):
        diagnose_photo(np.zeros((32, 32), np.float32))
