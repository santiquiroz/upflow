from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.services import photo_restore_runners as runners
from app.services.engines.colorize import ColorizeResult
from app.services.engines.drunet_restore import CpuFallback, DrunetResult
from app.services.engines.face_detect import FaceDetection
from app.services.engines.face_restore import FaceRestoreResult
from app.services.engines.photo_restore_engine import Fp16Rejection
from app.services.engines.scratch_detect import SCRATCH_MODEL_ID
from app.services.face_geometry import TEMPLATE_FFHQ_512
from app.services.photo_dsp import ToneSettings, apply_tone
from app.services.photo_restore_chain import RESTORE_CHAIN, restore_step
from app.services.photo_restore_pipeline import FaceSelection, RestoreHints, RestoreRequest, StepCall
from app.services.photo_restore_runners import (
    RunnerDeps,
    build_step_runners,
    descreen_mode,
    detected_selections,
    face_targets,
    mask_without_boxes,
    migan_or_none,
    run_colorize,
    run_deblock,
    run_denoise,
    run_descreen,
    run_faces,
    run_repair,
    run_tone,
    tone_settings,
)


def photo(height: int = 64, width: int = 80) -> np.ndarray:
    coarse = np.random.default_rng(0).random((4, 5, 3), dtype=np.float32)
    return np.clip(cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)


def fake_deps(*, migan: bool = False, rejections: tuple[Fp16Rejection, ...] = ()) -> RunnerDeps:
    engine = SimpleNamespace(
        settings=SimpleNamespace(migan_available=lambda: migan),
        fp16_rejections=lambda: rejections,
    )
    return RunnerDeps(engine=engine, calibrations=SimpleNamespace(), budget_ms=1200.0)


def call_for(step_id: str, image: np.ndarray, *, params=None, source=None, events=None, **request) -> StepCall:
    tone_kind = request.pop("tone_kind", "mono")
    job = RestoreRequest(image, (step_id,), tone_kind, "dml:0", params={step_id: params or {}}, **request)
    sink = events.append if events is not None else (lambda *event: None)
    return StepCall(restore_step(step_id), job, source, None, lambda *event: sink(event))


def test_every_chain_step_has_a_runner() -> None:
    assert set(build_step_runners(fake_deps())) == {spec.id for spec in RESTORE_CHAIN}


@pytest.mark.parametrize(
    ("requested", "kind", "expected"),
    [("auto", "halftone", "halftone"), ("auto", None, "texture"), ("texture", "halftone", "texture")],
)
def test_descreen_mode_follows_the_diagnosis_when_automatic(requested: str, kind: str | None, expected: str) -> None:
    assert descreen_mode(requested, RestoreHints(pattern_kind=kind)) == expected


def test_descreen_mode_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError):
        descreen_mode("moire", RestoreHints())


def test_descreen_halftone_uses_the_measured_screen_period(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}
    def fake_halftone(image, period, strength):
        seen.update(period=period, strength=strength)
        return image.copy()

    monkeypatch.setattr(runners, "descreen_halftone", fake_halftone)
    image = photo()

    hints = RestoreHints(pattern_kind="halftone", pattern_period=6.0)

    outcome = run_descreen(image, call_for("descreen", image, params={"mode": "auto", "strength": 0.7}, hints=hints))

    assert seen == {"period": 6.0, "strength": 0.7}
    assert outcome.details == {"mode": "halftone"}
    assert outcome.model is None


def test_tone_settings_drop_keep_tone_when_colorizing() -> None:
    settings = tone_settings({"strength": 0.8, "fix_faded": True, "gray_point": [3.0, 4.0]}, ("tone", "colorize"))

    assert settings == ToneSettings(strength=0.8, gray_point=(3, 4), keep_tone=False, fix_faded=True)


def test_run_tone_is_the_tone_of_photo_dsp() -> None:
    image = photo()

    outcome = run_tone(image, call_for("tone", image, params={"local_contrast": True}))

    np.testing.assert_allclose(outcome.image, apply_tone(image, ToneSettings(local_contrast=True)), atol=2e-6)


def test_repair_with_the_user_mask_fills_only_the_painted_pixels() -> None:
    image = photo()
    user_mask = np.zeros(image.shape[:2], dtype=bool)
    user_mask[20:23, 10:60] = True
    probability = np.zeros(image.shape[:2], dtype=np.float32)
    probability[40:42, :] = 0.9
    hints = RestoreHints(damage_probability=probability, user_mask=user_mask)
    events: list = []

    call = call_for("repair", image, params={"engine": "classic"}, hints=hints, events=events)

    outcome = run_repair(fake_deps(), image, call)

    changed = np.abs(outcome.image - image).max(axis=-1) > 0
    assert changed.any() and not changed[~user_mask].any()
    assert outcome.details["userEdited"] is True
    assert outcome.details["detectedCoverage"] == pytest.approx(probability.astype(bool).mean())
    assert outcome.details["engine"] == "classic"
    assert outcome.model is None and outcome.aux_models == ()
    assert ("restore_repair_detect", 1, 1) in events


def test_repair_without_a_saved_probability_runs_the_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    image = photo()
    probability = np.zeros(image.shape[:2], dtype=np.float32)
    probability[30:32, 5:70] = 0.95
    monkeypatch.setattr(runners, "scratch_detector", lambda engine: lambda source: probability)

    outcome = run_repair(fake_deps(), image, call_for("repair", image, params={"engine": "fast", "sensitivity": 0.5}))

    assert outcome.aux_models[0].model_id == SCRATCH_MODEL_ID
    assert outcome.details["engine"] == "classic"
    assert outcome.details["userEdited"] is False
    assert outcome.details["finalCoverage"] == pytest.approx(probability.astype(bool).mean())


def test_repair_can_leave_the_faces_unrepaired() -> None:
    image = photo()
    user_mask = np.zeros(image.shape[:2], dtype=bool)
    user_mask[10:20, 10:20] = True
    face = FaceSelection(0, ((0.0, 0.0),) * 5, 0.6, box=(8.0, 8.0, 22.0, 22.0))

    hints = RestoreHints(damage_probability=np.zeros(image.shape[:2], np.float32), user_mask=user_mask)
    params = {"engine": "classic", "leave_faces": True}

    outcome = run_repair(fake_deps(), image, call_for("repair", image, params=params, hints=hints, faces=(face,)))

    np.testing.assert_array_equal(outcome.image, image)
    assert outcome.details["touchesFaces"] is False


def test_mask_without_boxes_clears_whole_pixels_under_each_box() -> None:
    mask = np.ones((10, 10), dtype=bool)

    cleared = mask_without_boxes(mask, [(2.5, 3.2, 4.1, 5.0)])

    assert not cleared[3:5, 2:5].any()
    assert cleared.sum() == 100 - 6
    assert mask.all()


def test_fast_repair_uses_migan_only_when_its_pack_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runners, "migan_infer", lambda engine, device: ("migan", device))

    assert migan_or_none(fake_deps(migan=False), "fast", "dml:0") is None
    assert migan_or_none(fake_deps(migan=True), "classic", "dml:0") is None
    assert migan_or_none(fake_deps(migan=True), "fast", "dml:0") == ("migan", "dml:0")


def test_deblock_passes_strength_and_device_and_reports_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_deblock(context, image, strength, device):
        seen.update(strength=strength, device=device, stage=context.on_progress)
        context.on_progress(2, 4)
        return DrunetResult(image.copy(), "drunet-deblock-color", "dml:0", "fp16", 384)

    monkeypatch.setattr(runners, "deblock_photo", fake_deblock)
    image, events = photo(), []
    rejection = Fp16Rejection("drunet-deblock-color", "dml:0", "PSNR 30 dB < 50 dB")

    call = call_for("deblock", image, params={"strength": 0.8}, events=events)

    outcome = run_deblock(fake_deps(rejections=(rejection,)), image, call)

    assert (seen["strength"], seen["device"]) == (0.8, "dml:0")
    assert ("restore_deblock", 2, 4) in events
    expected = {"id": "drunet-deblock-color", "device": "dml:0", "precision": "fp16", "tile": 384}
    assert outcome.model.to_metadata() == expected
    assert outcome.warnings == ("fp16Rejected:drunet-deblock-color:PSNR 30 dB < 50 dB",)


def test_denoise_uses_the_diagnosed_sigma_and_keep_grain(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_denoise(context, image, strength, sigma, device, *, keep_grain_amount):
        seen.update(strength=strength, sigma=sigma, grain=keep_grain_amount)
        return DrunetResult(image.copy(), "drunet-color", "cpu", "fp32", 512, CpuFallback("drunet-color", "tdrBudget"))

    monkeypatch.setattr(runners, "denoise_photo", fake_denoise)
    image = photo()

    call = call_for("denoise", image, params={"strength": 0.5}, hints=RestoreHints(noise_sigma=0.02))

    outcome = run_denoise(fake_deps(), image, call)

    assert seen == {"strength": 0.5, "sigma": 0.02, "grain": 0.25}
    assert outcome.cpu_fallbacks == (CpuFallback("drunet-color", "tdrBudget"),)


def face_selection(index: int, **kwargs) -> FaceSelection:
    landmarks = tuple((float(x), float(y)) for x, y in TEMPLATE_FFHQ_512 * 0.1)
    return FaceSelection(index, landmarks, 0.6, **kwargs)


def test_face_targets_apply_the_selection_and_per_face_blends() -> None:
    faces = (face_selection(0), face_selection(1, enabled=False), face_selection(2))

    targets = face_targets(faces, {"selected": [1, 2], "per_face": {"2": 0.3}})

    assert [(t.index, t.enabled, t.blend) for t in targets] == [(0, False, 0.6), (1, True, 0.6), (2, True, 0.3)]


def test_face_targets_keep_the_session_choice_without_a_selection() -> None:
    faces = (face_selection(0), face_selection(1, enabled=False))

    assert [t.enabled for t in face_targets(faces, {})] == [True, False]


def test_run_faces_uses_the_confirmed_faces_and_reports_the_restored_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}

    def fake_restore(context, source, base, targets, *, tone_kind, device, model_id):
        seen.update(targets=targets, tone_kind=tone_kind, source=source)
        return FaceRestoreResult(base.copy(), (SimpleNamespace(index=0),), 1.0, model_id, device, "fp32")

    monkeypatch.setattr(runners, "restore_faces", fake_restore)
    image, source = photo(), photo()

    call = call_for("faces", image, source=source, faces=(face_selection(0),), tone_kind="toned")

    outcome = run_faces(fake_deps(), image, call)

    assert seen["source"] is source and seen["tone_kind"] == "toned"
    assert [t.index for t in seen["targets"]] == [0]
    assert outcome.details == {"restored": [0]}
    assert outcome.model.model_id == "gfpgan-v1.4"
    assert outcome.faces is not None


def test_run_faces_detects_when_the_analysis_did_not_confirm_faces(monkeypatch: pytest.MonkeyPatch) -> None:
    landmarks = tuple((float(x), float(y)) for x, y in TEMPLATE_FFHQ_512 * 0.25)
    detection = FaceDetection(box=(40.0, 50.0, 90.0, 110.0), score=0.97, landmarks=landmarks)
    monkeypatch.setattr(runners, "landmarked_face_detector", lambda engine: lambda image: (detection,))
    monkeypatch.setattr(
        runners,
        "restore_faces",
        lambda context, source, base, targets, **kwargs: FaceRestoreResult(base.copy(), (), 1.0),
    )
    image = photo(160, 160)

    outcome = run_faces(fake_deps(), image, call_for("faces", image, source=image, params={"blend": 0.7}))

    assert outcome.aux_models[0].model_id == "retinaface-r34"
    assert outcome.model is None
    assert outcome.details == {"restored": []}


def test_detected_selections_follow_the_face_policy() -> None:
    blurry = np.full((200, 200, 3), 0.5, dtype=np.float32)
    big = FaceDetection((0, 0, 1, 1), 0.99, tuple((float(x), float(y)) for x, y in TEMPLATE_FFHQ_512 * 0.3))
    small = FaceDetection((0, 0, 1, 1), 0.95, tuple((float(x), float(y)) for x, y in TEMPLATE_FFHQ_512 * 0.1))

    selections = detected_selections(blurry, (small, big), blend=0.7)

    assert [(s.index, s.score, s.enabled, s.blend) for s in selections] == [(0, 0.99, True, 0.7), (1, 0.95, False, 0.4)]


def test_run_colorize_passes_the_options_and_keeps_the_predicted_ab(monkeypatch: pytest.MonkeyPatch) -> None:
    ab_512 = np.zeros((512, 512, 2), dtype=np.float32)

    def fake_colorize(context, image, *, tone_kind, device, options, model_id):
        fallback = CpuFallback(model_id, "fp16Rejected")
        return ColorizeResult(image.copy(), ab_512, options, model_id, "cpu", "fp32", fallback)

    monkeypatch.setattr(runners, "colorize", fake_colorize)
    image = photo()

    outcome = run_colorize(fake_deps(), image, call_for("colorize", image, params={"strength": 0.5, "saturation": 1.5}))

    assert outcome.ab_512 is ab_512
    assert outcome.details == {"model": "ddcolor-tiny", "strength": 0.5, "saturation": 1.5}
    assert outcome.cpu_fallbacks[0].reason == "fp16Rejected"


def test_a_painted_mask_without_a_saved_probability_skips_the_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_detector(engine):
        raise AssertionError("the painted mask replaces the automatic one")

    monkeypatch.setattr(runners, "scratch_detector", no_detector)
    image = photo()
    painted = np.zeros(image.shape[:2], dtype=bool)
    painted[5:8, 5:40] = True
    call = call_for("repair", image, params={"engine": "classic"}, hints=RestoreHints(user_mask=painted))

    outcome = run_repair(fake_deps(), image, call)

    assert outcome.details["userEdited"] is True
    assert outcome.details["detectedCoverage"] is None
    assert outcome.aux_models == ()
