from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.config import Settings
from app.services.engines.colorize import (
    AB_ARTIFACT_NAME,
    AB_LIMIT,
    COLORIZE_MODEL_ID,
    COLORIZE_NEEDS_MONO_CODE,
    FP16_REJECTED_REASON,
    FP16_UNAVAILABLE_REASON,
    RENDER_SIZE,
    ColorizeContext,
    ColorizeNeedsMonoError,
    ColorizeOptions,
    ab_artifact_bytes,
    apply_ab,
    colorize,
    colorize_band,
    colorize_metadata,
    load_ab_artifact,
    model_input,
    require_colorizable,
    relative_luminance,
    save_ab_artifact,
    srgb_to_linear,
    upsample_ab,
)
from app.services.engines.face_restore import (
    LOW_DISK_WARNING,
    WRITE_FAILED_WARNING,
    RecomposeArtifacts,
    restore_artifact_dir,
)
from app.services.engines.photo_restore_engine import NonFiniteOutputError, PhotoRestoreEngine
from app.services.engines.tiled_restore_runner import TDR_BUDGET_REASON, CalibrationCache, RestoreCancelled
from app.services.photo_dsp import neutral_gray
from app.services.restore_models import RestoreModelSpec

GPU = "dml:0"
CPU = "cpu"
GRAPH_NAME = "ddcolor-tiny-fp16.onnx"
WARM_AB = (18.0, 32.0)
LIGHTNESS_ATOL = 1e-3
LUMA_ROW = np.array([0.212671, 0.715160, 0.072169])


class OrtLikeSession:
    def __init__(self, transform, device: str) -> None:
        self.transform = transform
        self.device = device
        self.batches: list[np.ndarray] = []

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def get_outputs(self):
        return [SimpleNamespace(name="ab")]

    def run(self, output_names, feeds):
        batch = feeds["input"]
        self.batches.append(batch.copy())
        return [self.transform(batch, self.device)]


class SessionFactory:
    def __init__(self, transform, clock=None, ms_per_call: float = 0.0) -> None:
        self.transform = transform
        self.clock = clock
        self.ms_per_call = ms_per_call
        self.sessions: list[OrtLikeSession] = []

    def __call__(self, model_path: str, device: str, settings: Settings, **kwargs) -> OrtLikeSession:
        session = OrtLikeSession(self._timed, device)
        self.sessions.append(session)
        return session

    def _timed(self, batch: np.ndarray, device: str) -> np.ndarray:
        if self.clock is not None:
            self.clock.advance_ms(self.ms_per_call)
        return self.transform(batch, device)

    def batches(self) -> list[np.ndarray]:
        return [batch for session in self.sessions for batch in session.batches]

    def devices(self) -> list[str]:
        return [session.device for session in self.sessions]

    def calls_on(self, device: str) -> int:
        return sum(len(session.batches) for session in self.sessions if session.device == device)


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


def fake_ddcolor_session(ab=WARM_AB, gpu_shift: float = 0.0):
    # FakeDDColorSession: ab constante (con un corrimiento opcional solo en la GPU, para el canario).
    def transform(batch: np.ndarray, device: str) -> np.ndarray:
        assert batch.shape == (1, 3, RENDER_SIZE, RENDER_SIZE)
        shift = gpu_shift if device != CPU else 0.0
        values = np.asarray(ab, dtype=np.float32).reshape(1, 2, 1, 1) + np.float32(shift)
        return np.broadcast_to(values, (1, 2, RENDER_SIZE, RENDER_SIZE)).copy()

    return transform


def ab_field_session(ab_512: np.ndarray):
    def transform(batch: np.ndarray, device: str) -> np.ndarray:
        return np.transpose(ab_512, (2, 0, 1))[np.newaxis].copy()

    return transform


def ddcolor_spec(*, fp16: bool = True) -> RestoreModelSpec:
    return RestoreModelSpec(
        id=COLORIZE_MODEL_ID,
        name="DDColor-tiny",
        bundle="colorize",
        filename=GRAPH_NAME,
        fp16_filename=GRAPH_NAME if fp16 else None,
        license_spdx="Apache-2.0",
        license_url="https://example.com/LICENSE",
        copyright="Copyright (c) test",
        attribution="Test model",
        data_lineage="D1a",
        commercial_use="yes",
        source_url="https://example.com/model",
        source_revision="abc123",
        source_sha256="a" * 64,
        modifications=("exported to ONNX",),
        tile_min=RENDER_SIZE,
        fixed_shape=True,
        overlap=0,
    )


def make_context(
    tmp_path: Path, factory: SessionFactory, clock=None, *, fp16_file: bool = True, prefer_fp16: bool = True, **extra
) -> ColorizeContext:
    model_dir = tmp_path / "restore"
    model_dir.mkdir(exist_ok=True)
    spec = ddcolor_spec(fp16=fp16_file)
    (model_dir / GRAPH_NAME).write_bytes(b"\0" * 1024)
    settings = Settings(
        _env_file=None,
        RESTORE_MODEL_DIR=str(model_dir),
        RUNTIME_DIR=str(tmp_path / "runtime"),
        ONNX_PREFER_FP16=prefer_fp16,
    )
    engine = PhotoRestoreEngine(settings, NullCoordinator(), models={spec.id: spec}, create_session=factory)
    engine.begin_phase(GPU)
    engine.begin_phase(CPU)
    return ColorizeContext(
        engine=engine,
        calibrations=CalibrationCache(),
        budget_ms=float(settings.restore_call_budget_ms),
        clock=clock or FakeClock(),
        **extra,
    )


def textured_luma(height: int = 300, width: int = 420) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    smooth = 0.35 + 0.2 * xs / width + 0.1 * np.sin(ys / 23.0)
    fine = 0.04 * (((ys // 1 + xs // 1) % 2) * 2.0 - 1.0)
    return np.clip(smooth + fine, 0.0, 1.0)


def gray_photo(height: int = 300, width: int = 420) -> np.ndarray:
    return np.repeat(textured_luma(height, width)[:, :, np.newaxis], 3, axis=2).astype(np.float32)


def sepia_photo(height: int = 300, width: int = 420) -> np.ndarray:
    lab = np.empty((height, width, 3), dtype=np.float32)
    lab[..., 0] = textured_luma(height, width) * 100.0
    lab[..., 1], lab[..., 2] = 6.0, 20.0
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB), 0.0, 1.0)


def to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2Lab)


def exact_lightness(rgb: np.ndarray) -> np.ndarray:
    # CIE L* en float64: el Lab en float de cv2 se corre ~0,2 de L por su cuenta.
    values = rgb.astype(np.float64)
    luminance = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4) @ LUMA_ROW
    return np.where(luminance > 0.008856, 116.0 * np.cbrt(luminance) - 16.0, 903.3 * luminance)


def chroma(rgb: np.ndarray) -> np.ndarray:
    lab = to_lab(rgb)
    return np.hypot(lab[..., 1], lab[..., 2])


def constant_ab(a: float, b: float) -> np.ndarray:
    ab = np.empty((RENDER_SIZE, RENDER_SIZE, 2), dtype=np.float32)
    ab[..., 0], ab[..., 1] = a, b
    return ab


def varying_ab() -> np.ndarray:
    ys, xs = np.mgrid[0:RENDER_SIZE, 0:RENDER_SIZE].astype(np.float32)
    a = 30.0 * np.sin(xs / 9.0) * np.cos(ys / 13.0)
    b = 25.0 * np.cos(xs / 17.0 + ys / 11.0)
    return np.stack([a, b], axis=2).astype(np.float32)


# ---------------------------------------------------------------- solo fotos mono o viradas


@pytest.mark.parametrize("tone_kind", ["color", "hand_tinted"])
def test_a_photo_that_is_not_mono_or_toned_is_rejected_before_any_model_runs(tmp_path, tone_kind) -> None:
    factory = SessionFactory(fake_ddcolor_session())

    with pytest.raises(ColorizeNeedsMonoError) as caught:
        colorize(make_context(tmp_path, factory), gray_photo(), tone_kind=tone_kind, device=CPU)

    assert caught.value.code == COLORIZE_NEEDS_MONO_CODE == "restore.error.colorizeNeedsMono"
    assert factory.sessions == []


@pytest.mark.parametrize("tone_kind", ["mono", "toned"])
def test_mono_and_toned_photos_can_be_colorized(tone_kind) -> None:
    require_colorizable(tone_kind)


# ---------------------------------------------------------------- entrada del modelo


def test_the_model_sees_the_lab_gray_of_the_photo_resized_to_512(tmp_path) -> None:
    factory = SessionFactory(fake_ddcolor_session())
    photo = sepia_photo()

    colorize(make_context(tmp_path, factory), photo, tone_kind="toned", device=CPU)

    (batch,) = factory.batches()
    assert batch.shape == (1, 3, RENDER_SIZE, RENDER_SIZE)
    assert batch.dtype == np.float32
    np.testing.assert_array_equal(batch[0, 0], batch[0, 1])
    np.testing.assert_array_equal(batch[0, 0], batch[0, 2])
    expected = neutral_gray(cv2.resize(photo, (RENDER_SIZE, RENDER_SIZE), interpolation=cv2.INTER_LINEAR))
    np.testing.assert_allclose(np.transpose(batch[0], (1, 2, 0)), expected, atol=1e-6)


def test_a_large_photo_is_area_averaged_down_to_the_render_size() -> None:
    photo = gray_photo(1200, 1600)

    sample = model_input(photo)

    expected = neutral_gray(cv2.resize(photo, (RENDER_SIZE, RENDER_SIZE), interpolation=cv2.INTER_AREA))
    assert sample.shape == (RENDER_SIZE, RENDER_SIZE, 3)
    np.testing.assert_allclose(sample, expected, atol=1e-6)


# ---------------------------------------------------------------- post: recorte, bicubica, saturacion, L


def test_the_predicted_ab_is_clipped_to_plus_minus_110(tmp_path) -> None:
    factory = SessionFactory(fake_ddcolor_session(ab=(150.0, -140.0)))

    result = colorize(make_context(tmp_path, factory), gray_photo(), tone_kind="mono", device=CPU)

    assert AB_LIMIT == 110.0
    assert result.ab_512.shape == (RENDER_SIZE, RENDER_SIZE, 2)
    assert float(result.ab_512[..., 0].max()) == 110.0
    assert float(result.ab_512[..., 1].min()) == -110.0


def test_the_ab_is_upsampled_with_bicubic_not_nearest() -> None:
    ab = varying_ab()

    upsampled = upsample_ab(ab, (700, 900, 3))

    bicubic = cv2.resize(ab, (900, 700), interpolation=cv2.INTER_CUBIC)
    nearest = cv2.resize(ab, (900, 700), interpolation=cv2.INTER_NEAREST)
    assert upsampled.shape == (700, 900, 2)
    np.testing.assert_allclose(upsampled, bicubic, atol=1e-5)
    assert float(np.abs(upsampled - nearest).max()) > 1.0


def test_the_output_carries_the_upsampled_ab(tmp_path) -> None:
    ab = varying_ab() * np.float32(0.5)
    factory = SessionFactory(ab_field_session(ab))
    photo = np.clip(gray_photo(600, 800) + np.float32(0.15), 0.0, 1.0)

    result = colorize(make_context(tmp_path, factory), photo, tone_kind="mono", device=CPU)

    expected = cv2.resize(ab, (800, 600), interpolation=cv2.INTER_CUBIC)
    np.testing.assert_allclose(to_lab(result.image)[..., 1:], expected, atol=0.6)


def test_the_final_lightness_is_kept(tmp_path) -> None:
    factory = SessionFactory(ab_field_session(varying_ab()))
    photo = gray_photo()

    result = colorize(make_context(tmp_path, factory), photo, tone_kind="mono", device=CPU)

    assert float(chroma(result.image).mean()) > 10.0
    np.testing.assert_allclose(exact_lightness(result.image), exact_lightness(photo), atol=LIGHTNESS_ATOL)


def test_the_lightness_of_a_toned_photo_is_kept_and_its_tint_replaced(tmp_path) -> None:
    factory = SessionFactory(fake_ddcolor_session(ab=(-10.0, -14.0)))
    photo = sepia_photo()

    result = colorize(make_context(tmp_path, factory), photo, tone_kind="toned", device=CPU)

    lab = to_lab(result.image)
    np.testing.assert_allclose(exact_lightness(result.image), exact_lightness(photo), atol=LIGHTNESS_ATOL)
    np.testing.assert_allclose(lab[..., 1:].reshape(-1, 2).mean(axis=0), (-10.0, -14.0), atol=0.5)


def test_saturation_scales_the_ab() -> None:
    photo = gray_photo()
    ab = constant_ab(*WARM_AB)

    half = apply_ab(photo, ab, ColorizeOptions(saturation=0.5))

    np.testing.assert_allclose(to_lab(half)[..., 1:].reshape(-1, 2).mean(axis=0), (9.0, 16.0), atol=0.5)


def test_color_strength_zero_gives_the_gray_version() -> None:
    photo = sepia_photo()

    gray = apply_ab(photo, constant_ab(*WARM_AB), ColorizeOptions(strength=0.0))

    np.testing.assert_allclose(gray, neutral_gray(photo), atol=2e-3)
    np.testing.assert_array_equal(gray[..., 0], gray[..., 1])
    np.testing.assert_array_equal(gray[..., 0], gray[..., 2])


def test_color_strength_mixes_with_the_gray_version_in_linear_light() -> None:
    photo = gray_photo()
    ab = constant_ab(*WARM_AB)
    colored = srgb_to_linear(apply_ab(photo, ab, ColorizeOptions()))

    mixed = srgb_to_linear(apply_ab(photo, ab, ColorizeOptions(strength=0.4)))

    gray = relative_luminance(photo)[..., np.newaxis]
    np.testing.assert_allclose(mixed, gray + 0.4 * (colored - gray), atol=2e-4)


@pytest.mark.parametrize("strength", [0.0, 0.3, 0.7, 1.0])
def test_every_color_strength_keeps_the_lightness(strength) -> None:
    photo = gray_photo()

    mixed = apply_ab(photo, varying_ab(), ColorizeOptions(strength=strength))

    np.testing.assert_allclose(exact_lightness(mixed), exact_lightness(photo), atol=LIGHTNESS_ATOL)


def test_an_out_of_gamut_ab_is_desaturated_without_moving_the_lightness() -> None:
    photo = gray_photo()

    fitted = apply_ab(photo, constant_ab(-AB_LIMIT, -AB_LIMIT), ColorizeOptions())

    assert float(fitted.min()) >= 0.0 and float(fitted.max()) <= 1.0
    np.testing.assert_allclose(exact_lightness(fitted), exact_lightness(photo), atol=LIGHTNESS_ATOL)
    lab = to_lab(fitted)
    assert float(lab[..., 1].max()) < 0.0 and float(lab[..., 2].max()) < 0.0


def test_bands_join_without_seams() -> None:
    photo = gray_photo(2100, 60)
    ab = varying_ab()

    banded = apply_ab(photo, ab, ColorizeOptions())

    whole = colorize_band(photo, upsample_ab(ab, photo.shape), 1.0)
    np.testing.assert_array_equal(banded, whole)


def test_apply_ab_does_not_modify_its_inputs() -> None:
    photo = gray_photo()
    ab = varying_ab()
    photo_before, ab_before = photo.copy(), ab.copy()

    apply_ab(photo, ab, ColorizeOptions(saturation=1.5, strength=0.5))

    np.testing.assert_array_equal(photo, photo_before)
    np.testing.assert_array_equal(ab, ab_before)


@pytest.mark.parametrize(
    "options",
    [
        {"strength": -0.1},
        {"strength": 1.1},
        {"strength": float("nan")},
        {"saturation": -0.5},
        {"saturation": 2.5},
        {"saturation": float("inf")},
    ],
)
def test_options_outside_their_range_are_rejected(options) -> None:
    with pytest.raises(ValueError):
        ColorizeOptions(**options)


def test_colorize_needs_an_rgb_float_image(tmp_path) -> None:
    factory = SessionFactory(fake_ddcolor_session())
    context = make_context(tmp_path, factory)

    with pytest.raises(ValueError):
        colorize(context, (gray_photo() * 255).astype(np.uint8), tone_kind="mono", device=CPU)
    with pytest.raises(ValueError):
        colorize(context, gray_photo()[..., 0], tone_kind="mono", device=CPU)
    assert factory.sessions == []


def test_a_model_output_that_is_not_ab_512_is_rejected(tmp_path) -> None:
    factory = SessionFactory(lambda batch, device: batch)

    with pytest.raises(ValueError):
        colorize(make_context(tmp_path, factory), gray_photo(), tone_kind="mono", device=CPU)


def test_a_nan_from_the_model_fails_instead_of_being_clipped(tmp_path) -> None:
    factory = SessionFactory(fake_ddcolor_session(ab=(float("nan"), 10.0)))

    with pytest.raises(NonFiniteOutputError):
        colorize(make_context(tmp_path, factory), gray_photo(), tone_kind="mono", device=CPU)


def test_a_cancelled_job_stops_before_the_model_runs(tmp_path) -> None:
    cancel = threading.Event()
    cancel.set()
    factory = SessionFactory(fake_ddcolor_session())

    with pytest.raises(RestoreCancelled):
        colorize(make_context(tmp_path, factory, cancel_event=cancel), gray_photo(), tone_kind="mono", device=CPU)
    assert factory.batches() == []


# ---------------------------------------------------------------- device: GPU solo con fp16 sano, si no CPU


def test_on_the_cpu_the_single_graph_runs_without_fallback(tmp_path) -> None:
    factory = SessionFactory(fake_ddcolor_session())

    result = colorize(make_context(tmp_path, factory), gray_photo(), tone_kind="mono", device=CPU)

    assert (result.device, result.precision, result.cpu_fallback) == (CPU, "fp32", None)
    assert factory.devices() == [CPU]


def test_a_healthy_fp16_within_budget_runs_on_the_gpu(tmp_path) -> None:
    clock = FakeClock()
    factory = SessionFactory(fake_ddcolor_session(), clock=clock, ms_per_call=100.0)
    context = make_context(tmp_path, factory, clock)

    result = colorize(context, gray_photo(), tone_kind="mono", device=GPU)

    assert (result.device, result.precision, result.cpu_fallback) == (GPU, "fp16", None)
    assert context.engine.fp16_rejections() == ()
    # El canario compara contra una sesion efimera en CPU: DDColor no tiene grafo fp32 en la GPU.
    assert factory.calls_on(CPU) == 1


def test_an_fp16_rejected_by_the_canary_falls_back_to_the_cpu(tmp_path) -> None:
    factory = SessionFactory(fake_ddcolor_session(gpu_shift=5.0))
    context = make_context(tmp_path, factory)

    result = colorize(context, gray_photo(), tone_kind="mono", device=GPU)

    assert result.device == CPU and result.precision == "fp32"
    assert result.cpu_fallback.to_metadata() == {"model": COLORIZE_MODEL_ID, "reason": FP16_REJECTED_REASON}
    (rejection,) = context.engine.fp16_rejections()
    assert (rejection.model_id, rejection.device) == (COLORIZE_MODEL_ID, GPU)
    assert factory.calls_on(GPU) == 1
    np.testing.assert_allclose(result.ab_512[0, 0], WARM_AB)


@pytest.mark.parametrize(("fp16_file", "prefer_fp16"), [(False, True), (True, False)])
def test_without_fp16_on_the_gpu_it_runs_on_the_cpu_never_as_gpu_fp32(tmp_path, fp16_file, prefer_fp16) -> None:
    factory = SessionFactory(fake_ddcolor_session())
    context = make_context(tmp_path, factory, fp16_file=fp16_file, prefer_fp16=prefer_fp16)

    result = colorize(context, gray_photo(), tone_kind="mono", device=GPU)

    assert result.device == CPU
    assert result.cpu_fallback.to_metadata() == {"model": COLORIZE_MODEL_ID, "reason": FP16_UNAVAILABLE_REASON}
    assert GPU not in factory.devices()


def test_a_gpu_call_over_the_tdr_budget_falls_back_to_the_cpu(tmp_path) -> None:
    clock = FakeClock()
    factory = SessionFactory(fake_ddcolor_session(), clock=clock, ms_per_call=900.0)
    context = make_context(tmp_path, factory, clock)

    result = colorize(context, gray_photo(), tone_kind="mono", device=GPU)

    assert result.device == CPU
    assert result.cpu_fallback.to_metadata() == {"model": COLORIZE_MODEL_ID, "reason": TDR_BUDGET_REASON}
    assert factory.devices()[-1] == CPU


# ---------------------------------------------------------------- ab_512.npy para recomponer sin inferir


def colorized(tmp_path: Path):
    factory = SessionFactory(ab_field_session(varying_ab()))
    photo = gray_photo()
    options = ColorizeOptions(strength=0.8, saturation=1.2)
    result = colorize(make_context(tmp_path, factory), photo, tone_kind="mono", device=CPU, options=options)
    return photo, result, factory


def test_the_ab_512_is_saved_next_to_the_recompose_artifacts(tmp_path) -> None:
    _, result, _ = colorized(tmp_path)
    directory = restore_artifact_dir(tmp_path / "outputs", "job-1")

    saved = save_ab_artifact(directory, result.ab_512, free_bytes=lambda path: 10**12)

    assert saved == RecomposeArtifacts(directory)
    assert (directory / AB_ARTIFACT_NAME) == tmp_path / "outputs" / "job-1.restore" / "ab_512.npy"
    stored = load_ab_artifact(directory)
    assert stored.dtype == np.float32
    np.testing.assert_array_equal(stored, result.ab_512)


def test_recompose_reuses_the_saved_ab_without_inference(tmp_path) -> None:
    photo, result, factory = colorized(tmp_path)
    directory = restore_artifact_dir(tmp_path / "outputs", "job-1")
    save_ab_artifact(directory, result.ab_512, free_bytes=lambda path: 10**12)
    calls = len(factory.batches())

    rebuilt = apply_ab(photo, load_ab_artifact(directory), result.options)

    np.testing.assert_array_equal(rebuilt, result.image)
    assert len(factory.batches()) == calls


def test_low_disk_skips_the_ab_artifact(tmp_path) -> None:
    _, result, _ = colorized(tmp_path)
    directory = restore_artifact_dir(tmp_path / "outputs", "job-1")

    saved = save_ab_artifact(directory, result.ab_512, free_bytes=lambda path: ab_artifact_bytes() - 1)

    assert saved == RecomposeArtifacts(None, LOW_DISK_WARNING)
    assert not directory.exists()


def test_a_failed_ab_write_removes_only_its_own_file(tmp_path, monkeypatch) -> None:
    _, result, _ = colorized(tmp_path)
    directory = restore_artifact_dir(tmp_path / "outputs", "job-1")
    directory.mkdir(parents=True)
    (directory / "faces.json").write_text("{}", encoding="utf-8")

    def failing_save(file, array, allow_pickle=True):
        file.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(np, "save", failing_save)

    saved = save_ab_artifact(directory, result.ab_512, free_bytes=lambda path: 10**12)

    assert saved == RecomposeArtifacts(None, WRITE_FAILED_WARNING)
    assert sorted(path.name for path in directory.iterdir()) == ["faces.json"]


def test_a_failed_ab_write_in_an_empty_folder_leaves_nothing(tmp_path, monkeypatch) -> None:
    _, result, _ = colorized(tmp_path)
    directory = restore_artifact_dir(tmp_path / "outputs", "job-1")
    monkeypatch.setattr(np, "save", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))

    saved = save_ab_artifact(directory, result.ab_512, free_bytes=lambda path: 10**12)

    assert saved.warning == WRITE_FAILED_WARNING
    assert not directory.exists()


def test_an_ab_artifact_of_the_wrong_shape_is_rejected(tmp_path) -> None:
    directory = tmp_path / "job-1.restore"
    directory.mkdir()
    with (directory / AB_ARTIFACT_NAME).open("wb") as handle:
        np.save(handle, np.zeros((256, 256, 2), dtype=np.float32))

    with pytest.raises(ValueError):
        load_ab_artifact(directory)


def test_the_ab_disk_estimate_covers_the_array() -> None:
    assert ab_artifact_bytes() >= RENDER_SIZE * RENDER_SIZE * 2 * 4


# ---------------------------------------------------------------- metadata


def test_the_metadata_names_model_render_size_options_and_ab_reuse(tmp_path) -> None:
    _, result, _ = colorized(tmp_path)

    saved = colorize_metadata(result, RecomposeArtifacts(tmp_path))
    skipped = colorize_metadata(result, RecomposeArtifacts(None, LOW_DISK_WARNING))

    assert saved == {
        "model": COLORIZE_MODEL_ID,
        "renderSize": RENDER_SIZE,
        "strength": 0.8,
        "saturation": 1.2,
        "abReusedOnRecompose": True,
    }
    assert skipped["abReusedOnRecompose"] is False
