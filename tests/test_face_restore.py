from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.services.engines.face_restore import (
    BEFORE_FACES_NAME,
    FACE_MODEL_ID,
    FACES_MANIFEST_NAME,
    LOW_DISK_WARNING,
    WRITE_FAILED_WARNING,
    FacePatch,
    FaceRestoreContext,
    FaceTarget,
    blend_face,
    disk_free_bytes,
    face_model_input,
    match_face_tone,
    output_scale,
    paste_faces,
    read_unit_png16,
    recompose_artifact_bytes,
    restore_artifact_dir,
    restore_faces,
    save_recompose_artifacts,
)
from app.services.engines.photo_restore_engine import NonFiniteOutputError, PhotoRestoreEngine
from app.services.engines.tiled_restore_runner import TDR_BUDGET_REASON, CalibrationCache, RestoreCancelled
from app.services.face_geometry import (
    FACE_SIZE,
    TEMPLATE_FFHQ_512,
    align_face,
    align_matrix,
    inverse_paste_matrix,
    paste_box,
    upscaled_points,
)
from app.services.restore_models import RestoreModelSpec

GPU = "dml:0"
CPU = "cpu"
SEPIA_AB = (6.0, 20.0)


class OrtLikeSession:
    def __init__(self, transform) -> None:
        self.transform = transform
        self.batches: list[np.ndarray] = []

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def get_outputs(self):
        return [SimpleNamespace(name="output")]

    def run(self, output_names, feeds):
        batch = feeds["input"]
        self.batches.append(batch.copy())
        return [self.transform(batch)]


class SessionFactory:
    def __init__(self, transform, clock=None, ms_per_call: float = 0.0) -> None:
        self.transform = transform
        self.clock = clock
        self.ms_per_call = ms_per_call
        self.sessions: list[tuple[str, OrtLikeSession]] = []

    def __call__(self, model_path: str, device: str, settings: Settings, **kwargs) -> OrtLikeSession:
        session = OrtLikeSession(self._timed)
        self.sessions.append((device, session))
        return session

    def _timed(self, batch: np.ndarray) -> np.ndarray:
        if self.clock is not None:
            self.clock.advance_ms(self.ms_per_call)
        return self.transform(batch)

    def batches(self) -> list[np.ndarray]:
        return [batch for _, session in self.sessions for batch in session.batches]

    def devices(self) -> list[str]:
        return [device for device, _ in self.sessions]


class FakeClock:
    def __init__(self) -> None:
        self.seconds = 0.0

    def __call__(self) -> float:
        return self.seconds

    def advance_ms(self, ms: float) -> None:
        self.seconds += ms / 1000.0


class NullCoordinator:
    def register(self, owner: object) -> None:
        pass

    def acquire(self, device: str, owner: object) -> None:
        pass

    def invalidate_device(self, device: str) -> None:
        pass


def checker(amplitude: float = 0.05) -> np.ndarray:
    ys, xs = np.mgrid[0:FACE_SIZE, 0:FACE_SIZE]
    return (((ys // 2 + xs // 2) % 2) * 2.0 - 1.0).astype(np.float32) * amplitude


def fake_face_restorer_512(brightness: float = 0.1, tint=(0.0, 0.0, 0.0), detail: np.ndarray | None = None):
    # FakeFaceRestorer512: identidad + brillo (+ dominante y detalle fino opcionales).
    pattern = checker() if detail is None else detail
    offsets = np.asarray(tint, dtype=np.float32).reshape(1, 3, 1, 1) + np.float32(brightness)

    def transform(batch: np.ndarray) -> np.ndarray:
        assert batch.shape == (1, 3, FACE_SIZE, FACE_SIZE)
        return batch + offsets + pattern[np.newaxis, np.newaxis]

    return transform


def gfpgan_spec() -> RestoreModelSpec:
    return RestoreModelSpec(
        id=FACE_MODEL_ID,
        name="GFPGAN v1.4",
        bundle="faces",
        filename="gfpgan-v1.4.onnx",
        license_spdx="Apache-2.0",
        license_url="https://example.com/LICENSE",
        copyright="Copyright (c) test",
        attribution="Test model",
        data_lineage="D1b+D1c",
        commercial_use="yes",
        source_url="https://example.com/model",
        source_revision="abc123",
        source_sha256="a" * 64,
        modifications=("exported to ONNX",),
        tile_min=FACE_SIZE,
        fixed_shape=True,
        overlap=0,
    )


def make_context(tmp_path: Path, factory: SessionFactory, clock=None, **extra) -> FaceRestoreContext:
    model_dir = tmp_path / "restore"
    model_dir.mkdir(exist_ok=True)
    spec = gfpgan_spec()
    (model_dir / spec.filename).write_bytes(b"\0" * 1024)
    settings = Settings(_env_file=None, RESTORE_MODEL_DIR=str(model_dir), RUNTIME_DIR=str(tmp_path / "runtime"))
    engine = PhotoRestoreEngine(settings, NullCoordinator(), models={spec.id: spec}, create_session=factory)
    engine.begin_phase(GPU)
    engine.begin_phase(CPU)
    return FaceRestoreContext(
        engine=engine,
        calibrations=CalibrationCache(),
        budget_ms=float(settings.restore_call_budget_ms),
        clock=clock or FakeClock(),
        **extra,
    )


def tilted_landmarks(center=(300.0, 220.0), scale=0.35, angle_deg=12.0) -> np.ndarray:
    radians = math.radians(angle_deg)
    rotation = np.array([[math.cos(radians), -math.sin(radians)], [math.sin(radians), math.cos(radians)]])
    return ((TEMPLATE_FFHQ_512 - FACE_SIZE / 2) @ rotation.T) * scale + np.asarray(center)


def target(index: int = 0, *, center=(300.0, 220.0), blend: float = 1.0, enabled: bool = True) -> FaceTarget:
    points = tuple((float(x), float(y)) for x, y in tilted_landmarks(center=center))
    return FaceTarget(index=index, landmarks=points, blend=blend, enabled=enabled)


def smooth_luma(height: int = 400, width: int = 600) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    return 0.3 + 0.2 * xs / width + 0.15 * np.sin(ys / 37.0) * np.cos(xs / 53.0)


def color_photo(height: int = 400, width: int = 600) -> np.ndarray:
    luma = smooth_luma(height, width)
    return np.clip(np.stack([luma * 1.1, luma, luma * 0.8], axis=2), 0.0, 1.0).astype(np.float32)


def gray_photo(height: int = 400, width: int = 600) -> np.ndarray:
    return np.repeat(smooth_luma(height, width)[:, :, np.newaxis], 3, axis=2).astype(np.float32)


def sepia_photo(height: int = 400, width: int = 600) -> np.ndarray:
    lab = np.empty((height, width, 3), dtype=np.float32)
    lab[..., 0] = smooth_luma(height, width) * 100.0
    lab[..., 1], lab[..., 2] = SEPIA_AB
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0.0, 1.0)


def face_core(image: np.ndarray, center=(300.0, 220.0), radius: int = 30) -> np.ndarray:
    x, y = (int(round(v)) for v in center)
    return image[y - radius : y + radius, x - radius : x + radius]


def mean_ab(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2Lab)
    return lab[..., 1:].reshape(-1, 2).mean(axis=0)


def box_of(face: FaceTarget, shape: tuple[int, int], scale: float = 1.0):
    return paste_box(inverse_paste_matrix(align_matrix(np.asarray(face.landmarks)), scale), shape)


def outside_box(image: np.ndarray, box) -> np.ndarray:
    x0, y0, x1, y1 = box
    keep = np.ones(image.shape[:2], dtype=bool)
    keep[y0:y1, x0:x1] = False
    return image[keep]


# ---------------------------------------------------------------- tono de la entrada y de la salida


@pytest.mark.parametrize("tone_kind", ["mono", "toned"])
def test_a_monochrome_or_toned_photo_feeds_the_model_a_neutral_crop(tmp_path, tone_kind) -> None:
    factory = SessionFactory(fake_face_restorer_512())
    photo = sepia_photo() if tone_kind == "toned" else gray_photo()

    restore_faces(make_context(tmp_path, factory), photo, photo, [target()], tone_kind=tone_kind, device=CPU)

    (batch,) = factory.batches()
    np.testing.assert_allclose(batch[0, 0], batch[0, 1], atol=1e-6)
    np.testing.assert_allclose(batch[0, 1], batch[0, 2], atol=1e-6)


@pytest.mark.parametrize("tone_kind", ["color", "hand_tinted"])
def test_a_color_photo_feeds_the_model_the_aligned_crop_as_is(tmp_path, tone_kind) -> None:
    factory = SessionFactory(fake_face_restorer_512())
    photo = color_photo()
    face = target()

    restore_faces(make_context(tmp_path, factory), photo, photo, [face], tone_kind=tone_kind, device=CPU)

    (batch,) = factory.batches()
    aligned = align_face(photo, align_matrix(np.asarray(face.landmarks)))
    np.testing.assert_array_equal(np.transpose(batch[0], (1, 2, 0)), aligned)


def test_a_mono_photo_comes_back_gray_even_if_the_model_adds_a_cast(tmp_path) -> None:
    factory = SessionFactory(fake_face_restorer_512(tint=(0.08, 0.0, -0.05)))
    photo = gray_photo()

    result = restore_faces(make_context(tmp_path, factory), photo, photo, [target()], tone_kind="mono", device=CPU)

    (face,) = result.faces
    assert np.ptp(face.restored, axis=2).max() < 1e-5
    assert np.ptp(result.image, axis=2).max() < 1e-5


def test_the_low_frequency_transfer_gives_a_sepia_face_its_tone_back(tmp_path) -> None:
    factory = SessionFactory(fake_face_restorer_512())
    photo = sepia_photo()

    result = restore_faces(make_context(tmp_path, factory), photo, photo, [target()], tone_kind="toned", device=CPU)

    np.testing.assert_allclose(mean_ab(face_core(result.image)), SEPIA_AB, atol=1.0)
    np.testing.assert_allclose(mean_ab(face_core(result.image)), mean_ab(face_core(photo)), atol=0.5)


def test_the_transfer_drops_the_model_brightness_shift_and_keeps_its_fine_detail(tmp_path) -> None:
    factory = SessionFactory(fake_face_restorer_512(brightness=0.1))
    photo = color_photo()

    result = restore_faces(make_context(tmp_path, factory), photo, photo, [target()], tone_kind="color", device=CPU)

    (face,) = result.faces
    core = (slice(160, 448), slice(128, 384))
    assert abs(float(face.restored[core].mean() - face.aligned[core].mean())) < 0.01
    detail = face.restored[core] - face.aligned[core]
    assert float(np.abs(detail).mean()) > 0.03


def test_match_face_tone_on_a_color_crop_is_the_wavelet_transfer_alone() -> None:
    aligned = np.full((64, 64, 3), (0.5, 0.4, 0.3), dtype=np.float32)
    restored = np.full((64, 64, 3), (0.6, 0.6, 0.6), dtype=np.float32)

    matched = match_face_tone(restored, aligned, monochrome=False)

    np.testing.assert_allclose(matched, aligned, atol=1e-5)


def test_face_model_input_is_neutral_only_for_monochrome() -> None:
    crop = np.full((8, 8, 3), (0.5, 0.4, 0.3), dtype=np.float32)

    assert face_model_input(crop, monochrome=False) is crop
    neutral = face_model_input(crop, monochrome=True)
    assert np.ptp(neutral, axis=2).max() < 1e-6


# ---------------------------------------------------------------- mezcla


def test_blend_face_mixes_alpha_restored_with_one_minus_alpha_aligned() -> None:
    restored = np.full((4, 4, 3), 0.8, dtype=np.float32)
    aligned = np.full((4, 4, 3), 0.4, dtype=np.float32)

    np.testing.assert_allclose(blend_face(restored, aligned, 0.25), 0.25 * 0.8 + 0.75 * 0.4, atol=1e-7)
    np.testing.assert_array_equal(blend_face(restored, aligned, 0.0), aligned)
    np.testing.assert_array_equal(blend_face(restored, aligned, 1.0), restored)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan")])
def test_blend_outside_zero_to_one_is_rejected(alpha) -> None:
    crop = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        blend_face(crop, crop, alpha)


def test_the_pasted_change_scales_with_the_face_blend(tmp_path) -> None:
    photo = color_photo()

    def pasted(alpha: float) -> np.ndarray:
        context = make_context(tmp_path, SessionFactory(fake_face_restorer_512()))
        face = target(blend=alpha)
        return restore_faces(context, photo, photo, [face], tone_kind="color", device=CPU).image

    none, half, full = pasted(0.0), pasted(0.5), pasted(1.0)
    np.testing.assert_allclose(half - none, 0.5 * (full - none), atol=1e-5)
    assert float(np.abs(full - none).max()) > 0.03


# ---------------------------------------------------------------- pegado


def test_pasting_only_changes_pixels_inside_the_face_box(tmp_path) -> None:
    photo = color_photo()
    face = target()
    context = make_context(tmp_path, SessionFactory(fake_face_restorer_512()))

    result = restore_faces(context, photo, photo, [face], tone_kind="color", device=CPU)

    box = box_of(face, photo.shape[:2])
    np.testing.assert_array_equal(outside_box(result.image, box), outside_box(photo, box))
    assert not np.array_equal(result.image, photo)


def test_unselected_faces_are_left_untouched_and_never_reach_the_model(tmp_path) -> None:
    factory = SessionFactory(fake_face_restorer_512())
    photo = color_photo(400, 900)
    chosen, skipped = target(0, center=(220.0, 200.0)), target(1, center=(680.0, 200.0), enabled=False)
    context = make_context(tmp_path, factory)

    result = restore_faces(context, photo, photo, [chosen, skipped], tone_kind="color", device=CPU)

    assert len(factory.batches()) == 1
    assert [face.index for face in result.faces] == [0]
    x0, y0, x1, y1 = box_of(skipped, photo.shape[:2])
    np.testing.assert_array_equal(result.image[y0:y1, x0:x1], photo[y0:y1, x0:x1])


def test_without_selected_faces_no_model_is_loaded_and_the_base_is_returned(tmp_path) -> None:
    factory = SessionFactory(fake_face_restorer_512())
    photo = color_photo()

    result = restore_faces(
        make_context(tmp_path, factory), photo, photo, [target(enabled=False)], tone_kind="color", device=CPU
    )

    assert factory.sessions == []
    assert result.faces == ()
    np.testing.assert_array_equal(result.image, photo)


def test_paste_faces_does_not_modify_the_base() -> None:
    base = color_photo()
    before = base.copy()
    matrix = align_matrix(tilted_landmarks())

    paste_faces(base, [FacePatch(np.ones((FACE_SIZE, FACE_SIZE, 3), np.float32), matrix)], 1.0)

    np.testing.assert_array_equal(base, before)


def test_a_face_outside_the_output_is_skipped() -> None:
    base = color_photo()
    matrix = align_matrix(tilted_landmarks(center=(3000.0, 3000.0)))

    pasted = paste_faces(base, [FacePatch(np.ones((FACE_SIZE, FACE_SIZE, 3), np.float32), matrix)], 1.0)

    np.testing.assert_array_equal(pasted, base)


def blob_detail(sigma: float = 3.0, amplitude: float = 0.3) -> np.ndarray:
    ys, xs = np.mgrid[0:FACE_SIZE, 0:FACE_SIZE].astype(np.float32)
    detail = np.zeros((FACE_SIZE, FACE_SIZE), dtype=np.float32)
    for x, y in TEMPLATE_FFHQ_512:
        detail += np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma**2))
    return detail * amplitude


def centroid(image: np.ndarray, near: np.ndarray, radius: int) -> np.ndarray:
    x0, y0 = (int(round(v)) - radius for v in near)
    window = np.clip(image[y0 : y0 + 2 * radius + 1, x0 : x0 + 2 * radius + 1].astype(np.float64), 0.0, None)
    ys, xs = np.mgrid[0 : window.shape[0], 0 : window.shape[1]]
    total = window.sum()
    return np.array([(xs * window).sum() / total + x0, (ys * window).sum() / total + y0])


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_faces_are_pasted_at_the_output_scale_with_the_half_pixel_center(tmp_path, scale) -> None:
    photo = gray_photo(300, 500)
    base = cv2.resize(photo, (500 * scale, 300 * scale), interpolation=cv2.INTER_LINEAR)
    face = target(center=(250.0, 150.0))
    factory = SessionFactory(fake_face_restorer_512(brightness=0.0, detail=blob_detail()))

    result = restore_faces(make_context(tmp_path, factory), photo, base, [face], tone_kind="color", device=CPU)

    assert result.scale == scale
    change = (result.image - base).mean(axis=2)
    for expected in upscaled_points(np.asarray(face.landmarks), scale):
        found = centroid(change, expected, radius=3 * scale)
        assert np.hypot(*(found - expected)) < 0.5


def test_output_scale_is_the_uniform_ratio_between_the_images() -> None:
    assert output_scale((300, 500, 3), (600, 1000, 3)) == 2.0
    assert output_scale((300, 500, 3), (300, 500, 3)) == 1.0


def test_output_scale_rejects_a_non_uniform_ratio() -> None:
    with pytest.raises(ValueError):
        output_scale((300, 500, 3), (600, 1200, 3))


def test_faces_need_rgb_float_images(tmp_path) -> None:
    context = make_context(tmp_path, SessionFactory(fake_face_restorer_512()))
    photo = (color_photo() * 255).astype(np.uint8)
    with pytest.raises(ValueError):
        restore_faces(context, photo, photo, [target()], tone_kind="color", device=CPU)


# ---------------------------------------------------------------- modelo: guardia, clamp y presupuesto


def test_the_restored_crop_is_clamped_on_the_host(tmp_path) -> None:
    factory = SessionFactory(lambda batch: batch * 3.0 - 1.0)
    photo = color_photo()

    result = restore_faces(make_context(tmp_path, factory), photo, photo, [target()], tone_kind="color", device=CPU)

    (face,) = result.faces
    assert face.restored.min() >= 0.0 and face.restored.max() <= 1.0


def test_a_nan_from_the_face_model_fails_instead_of_being_clamped(tmp_path) -> None:
    def with_nan(batch: np.ndarray) -> np.ndarray:
        out = batch.copy()
        out[0, 0, 100, 100] = np.nan
        return out

    photo = color_photo()
    with pytest.raises(NonFiniteOutputError):
        restore_faces(
            make_context(tmp_path, SessionFactory(with_nan)), photo, photo, [target()], tone_kind="color", device=CPU
        )


def test_a_gpu_face_model_within_budget_runs_on_the_gpu(tmp_path) -> None:
    clock = FakeClock()
    factory = SessionFactory(fake_face_restorer_512(), clock=clock, ms_per_call=100.0)
    photo = color_photo()
    context = make_context(tmp_path, factory, clock)

    result = restore_faces(context, photo, photo, [target()], tone_kind="color", device=GPU)

    assert (result.device, result.precision, result.cpu_fallback) == (GPU, "fp32", None)
    assert set(factory.devices()) == {GPU}


def test_a_gpu_face_model_over_the_tdr_budget_falls_back_to_cpu(tmp_path) -> None:
    clock = FakeClock()
    factory = SessionFactory(fake_face_restorer_512(), clock=clock, ms_per_call=900.0)
    photo = color_photo()
    context = make_context(tmp_path, factory, clock)

    result = restore_faces(context, photo, photo, [target()], tone_kind="color", device=GPU)

    assert result.device == CPU
    assert result.cpu_fallback is not None
    assert result.cpu_fallback.to_metadata() == {"model": FACE_MODEL_ID, "reason": TDR_BUDGET_REASON}
    assert factory.devices()[-1] == CPU
    assert result.faces


# ---------------------------------------------------------------- progreso y cancelacion


def test_progress_is_reported_once_per_restored_face(tmp_path) -> None:
    seen: list[tuple[int, int]] = []
    factory = SessionFactory(fake_face_restorer_512())
    context = make_context(tmp_path, factory, on_progress=lambda done, total: seen.append((done, total)))
    photo = color_photo(400, 900)
    faces = [target(0, center=(220.0, 200.0)), target(1, center=(680.0, 200.0))]

    restore_faces(context, photo, photo, faces, tone_kind="color", device=CPU)

    assert seen == [(1, 2), (2, 2)]


def test_a_cancelled_job_stops_before_the_next_face(tmp_path) -> None:
    cancel = threading.Event()
    cancel.set()
    factory = SessionFactory(fake_face_restorer_512())
    context = make_context(tmp_path, factory, cancel_event=cancel)
    photo = color_photo()

    with pytest.raises(RestoreCancelled):
        restore_faces(context, photo, photo, [target()], tone_kind="color", device=CPU)
    assert factory.batches() == []


# ---------------------------------------------------------------- artefactos de recomposicion


def restored_result(tmp_path: Path, scale: int = 2):
    photo = color_photo(300, 500)
    base = cv2.resize(photo, (500 * scale, 300 * scale), interpolation=cv2.INTER_LINEAR)
    faces = [target(0, center=(150.0, 150.0), blend=0.6), target(1, center=(360.0, 150.0), blend=0.4)]
    context = make_context(tmp_path, SessionFactory(fake_face_restorer_512()))
    return base, restore_faces(context, photo, base, faces, tone_kind="color", device=CPU)


def test_the_artifact_directory_is_the_job_id_dot_restore_inside_outputs(tmp_path) -> None:
    assert restore_artifact_dir(tmp_path / "outputs", "job-1") == tmp_path / "outputs" / "job-1.restore"


def test_recompose_artifacts_are_written(tmp_path) -> None:
    base, result = restored_result(tmp_path)
    directory = restore_artifact_dir(tmp_path / "outputs", "job-1")

    saved = save_recompose_artifacts(directory, base, result, free_bytes=lambda path: 10**12)

    assert saved.available and saved.warning is None
    manifest = json.loads((directory / FACES_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["schemaVersion"] == 1
    assert manifest["scale"] == 2.0
    assert manifest["beforeFaces"] == BEFORE_FACES_NAME
    assert manifest["size"] == {"width": 1000, "height": 600}
    np.testing.assert_allclose(read_unit_png16(directory / BEFORE_FACES_NAME), base, atol=1 / 65535)
    for entry, face in zip(manifest["faces"], result.faces, strict=True):
        assert entry["index"] == face.index
        assert entry["blend"] == face.blend
        np.testing.assert_allclose(np.asarray(entry["matrix"]), face.matrix)
        np.testing.assert_allclose(read_unit_png16(directory / entry["aligned"]), face.aligned, atol=1 / 65535)
        np.testing.assert_allclose(read_unit_png16(directory / entry["restored"]), face.restored, atol=1 / 65535)


def test_recompose_from_the_artifacts_reproduces_the_result(tmp_path) -> None:
    base, result = restored_result(tmp_path)
    directory = restore_artifact_dir(tmp_path / "outputs", "job-1")
    save_recompose_artifacts(directory, base, result, free_bytes=lambda path: 10**12)
    manifest = json.loads((directory / FACES_MANIFEST_NAME).read_text(encoding="utf-8"))

    def patch(entry: dict) -> FacePatch:
        restored = read_unit_png16(directory / entry["restored"])
        aligned = read_unit_png16(directory / entry["aligned"])
        return FacePatch(blend_face(restored, aligned, entry["blend"]), np.asarray(entry["matrix"]))

    patches = [patch(entry) for entry in manifest["faces"]]
    rebuilt = paste_faces(read_unit_png16(directory / BEFORE_FACES_NAME), patches, manifest["scale"])

    np.testing.assert_allclose(rebuilt, result.image, atol=2 / 65535)


def test_low_disk_disables_recompose_without_writing(tmp_path) -> None:
    base, result = restored_result(tmp_path)
    directory = restore_artifact_dir(tmp_path / "outputs", "job-1")
    needed = recompose_artifact_bytes(base.shape, len(result.faces))

    saved = save_recompose_artifacts(directory, base, result, free_bytes=lambda path: needed - 1)

    assert not saved.available
    assert saved.warning == LOW_DISK_WARNING
    assert not directory.exists()


def test_an_unmeasurable_disk_still_writes_the_artifacts(tmp_path) -> None:
    base, result = restored_result(tmp_path)
    directory = restore_artifact_dir(tmp_path / "outputs", "job-1")

    saved = save_recompose_artifacts(directory, base, result, free_bytes=lambda path: None)

    assert saved.available
    assert (directory / FACES_MANIFEST_NAME).is_file()


def test_a_failed_write_removes_the_partial_artifacts(tmp_path, monkeypatch) -> None:
    base, result = restored_result(tmp_path)
    directory = restore_artifact_dir(tmp_path / "outputs", "job-1")
    real_write = Path.write_bytes
    calls = {"n": 0}

    def failing_write(self: Path, data: bytes) -> int:
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk full")
        return real_write(self, data)

    monkeypatch.setattr(Path, "write_bytes", failing_write)

    saved = save_recompose_artifacts(directory, base, result, free_bytes=lambda path: 10**12)

    assert not saved.available
    assert saved.warning == WRITE_FAILED_WARNING
    assert not directory.exists()


def test_a_failed_write_keeps_files_it_did_not_write(tmp_path, monkeypatch) -> None:
    base, result = restored_result(tmp_path)
    directory = restore_artifact_dir(tmp_path / "outputs", "job-1")
    directory.mkdir(parents=True)
    (directory / "ab_512.npy").write_bytes(b"ab")
    monkeypatch.setattr(Path, "write_text", lambda self, *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))

    saved = save_recompose_artifacts(directory, base, result, free_bytes=lambda path: 10**12)

    assert saved.warning == WRITE_FAILED_WARNING
    assert sorted(path.name for path in directory.iterdir()) == ["ab_512.npy"]


def test_blends_are_checked_before_any_face_reaches_the_model(tmp_path) -> None:
    factory = SessionFactory(fake_face_restorer_512())
    photo = color_photo(400, 900)
    faces = [target(0, center=(220.0, 200.0)), target(1, center=(680.0, 200.0), blend=1.5)]

    with pytest.raises(ValueError):
        restore_faces(make_context(tmp_path, factory), photo, photo, faces, tone_kind="color", device=CPU)
    assert factory.batches() == []


def test_the_disk_estimate_covers_the_base_and_two_crops_per_face() -> None:
    assert recompose_artifact_bytes((600, 1000, 3), 2) >= 600 * 1000 * 6 + 2 * 2 * FACE_SIZE * FACE_SIZE * 6


def test_disk_free_bytes_measures_the_nearest_existing_folder(tmp_path) -> None:
    free = disk_free_bytes(tmp_path / "outputs" / "job-1.restore")

    assert free is not None and free > 0
