from __future__ import annotations

import asyncio
import hashlib
import threading
import tracemalloc
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.engines.colorize import AB_ARTIFACT_NAME, ColorizeOptions, apply_ab
from app.services.engines.face_restore import (
    FACES_MANIFEST_NAME,
    FaceRestoreResult,
    RestoredFace,
    blended_patch,
    output_scale,
    paste_faces,
)
from app.services.engines.photo_restore_engine import run_cancellable
from app.services.engines.tiled_restore_runner import RestoreCancelled
from app.services.face_geometry import TEMPLATE_FFHQ_512, align_face, align_matrix
from app.services.photo_dsp import ToneSettings, apply_tone
from app.services.photo_restore_chain import UnknownRestoreStep
from app.services.photo_restore_pipeline import (
    PREVIEW_MARGIN_PX,
    STAGE_ORDER,
    FaceSelection,
    InvalidPreviewCrop,
    ModelUse,
    PhotoRestorePipeline,
    PixelLimits,
    RestoreRequest,
    RestoreTooLarge,
    StepCall,
    StepOutcome,
    needs_upscale,
    restore_metadata,
)
from app.services.photo_restore_runners import run_tone
from app.services.restore_models import MIGAN_MODEL_ID
from app.services.restore_provenance import (
    InputInfo,
    ModelCatalog,
    OutputFile,
    SidecarContext,
    UpscaleInfo,
    build_sidecar,
    default_model_catalog,
    digital_source_type,
    facts_from_records,
    sha256_file,
)
from app.services.xmp_packet import DIGITAL_SOURCE_COMPOSITE, DIGITAL_SOURCE_ENHANCED

LIMITS = PixelLimits(max_input_pixels=40_000_000, max_output_pixels=100_000_000)
ALL_STEPS = ("descreen", "repair", "deblock", "denoise", "tone", "faces", "colorize")
AB_SEPIA = (6.0, 18.0)


def photo(height: int = 96, width: int = 128, seed: int = 0) -> np.ndarray:
    coarse = np.random.default_rng(seed).random((6, 8, 3), dtype=np.float32)
    return np.clip(cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC), 0.0, 1.0)


class Journal:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[int, ...]]] = []
        self.faces: list[tuple[FaceSelection, ...]] = []

    def runner(self, step_id: str, **outcome):
        def run(image: np.ndarray, call: StepCall) -> StepOutcome:
            self.calls.append((step_id, image.shape))
            call.progress(STAGE_ORDER[STAGE_ORDER.index(f"restore_{_stage(step_id)}")], 1, 2)
            shifted = image.copy()
            shifted += np.float32(0.001)
            return StepOutcome(np.clip(shifted, 0.0, 1.0, out=shifted), **outcome)

        return run

    def faces_runner(self, image: np.ndarray, call: StepCall) -> StepOutcome:
        self.calls.append(("faces", image.shape))
        self.faces.append(call.request.faces)
        return fake_faces(image, call)

    def colorize_runner(self, image: np.ndarray, call: StepCall) -> StepOutcome:
        self.calls.append(("colorize", image.shape))
        return fake_colorize(image, call)

    def runners(self, **overrides) -> dict:
        runners = {step: self.runner(step) for step in ALL_STEPS}
        runners["faces"] = self.faces_runner
        runners["colorize"] = self.colorize_runner
        return {**runners, **overrides}


def _stage(step_id: str) -> str:
    return "repair_fill" if step_id == "repair" else step_id


def fake_faces(image: np.ndarray, call: StepCall) -> StepOutcome:
    scale = output_scale(call.source.shape, image.shape)
    faces = tuple(_fake_restored(call.source, face) for face in call.request.faces if face.enabled)
    pasted = paste_faces(image, [blended_patch(face) for face in faces], scale)
    result = FaceRestoreResult(pasted, faces, scale, "gfpgan-v1.4", "cpu", "fp32")
    return StepOutcome(
        pasted,
        model=ModelUse("gfpgan-v1.4", "cpu", "fp32"),
        details={"restored": [face.index for face in faces]},
        faces=result,
    )


def _fake_restored(source: np.ndarray, face: FaceSelection) -> RestoredFace:
    matrix = align_matrix(np.asarray(face.landmarks))
    aligned = align_face(source, matrix)
    restored = aligned.copy()
    restored *= np.float32(0.5)
    return RestoredFace(face.index, matrix, face.blend, aligned, restored)


def fake_colorize(image: np.ndarray, call: StepCall) -> StepOutcome:
    ab_512 = np.empty((512, 512, 2), dtype=np.float32)
    ab_512[...] = AB_SEPIA
    colored = apply_ab(image, ab_512, ColorizeOptions(), cancel_event=call.cancel_event)
    return StepOutcome(
        colored,
        model=ModelUse("ddcolor-tiny", "cpu", "fp32"),
        details={"model": "ddcolor-tiny", "strength": 1.0, "saturation": 1.0},
        ab_512=ab_512,
    )


def template_face(offset: tuple[float, float], scale: float) -> tuple[tuple[float, float], ...]:
    points = TEMPLATE_FFHQ_512 * scale + np.asarray(offset)
    return tuple((float(x), float(y)) for x, y in points)


def face(index: int, offset=(10.0, 8.0), scale=0.12, **kwargs) -> FaceSelection:
    landmarks = template_face(offset, scale)
    xs, ys = [x for x, _ in landmarks], [y for _, y in landmarks]
    box = (min(xs) - 5, min(ys) - 10, max(xs) + 5, max(ys) + 8)
    return FaceSelection(index, landmarks, 0.6, box=box, score=0.99, eye_px=15.0, sharpness=0.01, **kwargs)


def request(image: np.ndarray, steps=ALL_STEPS, **kwargs) -> RestoreRequest:
    return RestoreRequest(image=image, steps=tuple(steps), tone_kind="mono", device="cpu", **kwargs)


def test_steps_run_in_catalog_order_split_between_the_two_phases() -> None:
    journal = Journal()
    pipeline = PhotoRestorePipeline(journal.runners(), limits=LIMITS)
    shuffled = ("colorize", "tone", "repair", "faces", "descreen", "denoise", "deblock")

    pre = pipeline.run_pre(request(photo(), shuffled, faces=(face(0),)))
    native = [step for step, _ in journal.calls]
    post = pipeline.run_post(pre)

    assert native == ["descreen", "repair", "deblock", "denoise", "tone"]
    assert [step for step, _ in journal.calls] == list(ALL_STEPS)
    assert [r.step_id for r in pre.records] == native
    assert [r.step_id for r in post.records] == ["faces", "colorize"]


def test_progress_is_monotonic_across_the_steps() -> None:
    events: list[tuple[str, int, int]] = []

    def regressing(image: np.ndarray, call: StepCall) -> StepOutcome:
        call.progress("restore_denoise", 3, 4)
        call.progress("restore_denoise", 1, 4)
        call.progress("restore_deblock", 9, 9)
        return StepOutcome(image.copy())

    journal = Journal()
    runners = journal.runners(denoise=regressing)
    pipeline = PhotoRestorePipeline(runners, limits=LIMITS, on_progress=lambda *e: events.append(e))

    pipeline.run(request(photo(), faces=(face(0),)))

    positions = [(STAGE_ORDER.index(stage), done) for stage, done, _ in events]
    assert positions == sorted(positions)
    assert ("restore_denoise", 1, 4) not in events
    assert ("restore_deblock", 9, 9) not in events
    assert ("restore_denoise", 4, 4) in events
    assert {stage for stage, _, _ in events} == set(STAGE_ORDER)
    assert events[-1] == ("restore_colorize", 1, 1)


async def test_cancelling_waits_for_the_worker_thread_before_returning() -> None:
    started, finished = threading.Event(), threading.Event()
    journal = Journal()

    def slow_deblock(image: np.ndarray, call: StepCall) -> StepOutcome:
        started.set()
        call.cancel_event.wait(timeout=5)
        finished.set()
        return StepOutcome(image.copy())

    pipeline = PhotoRestorePipeline(journal.runners(deblock=slow_deblock), limits=LIMITS)
    task = asyncio.ensure_future(run_cancellable(pipeline.run_pre, request(photo(), ("deblock", "denoise"))))
    await asyncio.to_thread(started.wait, 5)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished.is_set()
    assert "denoise" not in [step for step, _ in journal.calls]


def test_a_cancelled_event_stops_before_the_next_step() -> None:
    cancel = threading.Event()
    journal = Journal()

    def cancelling(image: np.ndarray, call: StepCall) -> StepOutcome:
        cancel.set()
        return StepOutcome(image.copy())

    pipeline = PhotoRestorePipeline(journal.runners(deblock=cancelling), limits=LIMITS)

    with pytest.raises(RestoreCancelled):
        pipeline.run_pre(request(photo(), ("deblock", "denoise")), cancel)

    assert journal.calls == []


def test_scale_one_skips_the_upscaler() -> None:
    calls: list[tuple[int, ...]] = []
    pipeline = PhotoRestorePipeline(Journal().runners(), limits=LIMITS)
    job = request(photo(), ("tone",))

    _, post = pipeline.run(job, upscale=lambda image, cancel: calls.append(image.shape))

    assert not needs_upscale(job)
    assert calls == []
    assert post.scale == 1.0


def test_an_upscaled_job_runs_the_output_steps_on_the_upscaled_image() -> None:
    journal = Journal()
    pipeline = PhotoRestorePipeline(journal.runners(), limits=LIMITS)
    image = photo()

    def double(source: np.ndarray, cancel) -> np.ndarray:
        return cv2.resize(source, (source.shape[1] * 2, source.shape[0] * 2), interpolation=cv2.INTER_LINEAR)

    job = request(image, ("tone", "faces"), scale=2.0, upscale_mode="ai", faces=(face(0),))

    pre, post = pipeline.run(job, upscale=double)

    assert journal.calls[-1] == ("faces", (192, 256, 3))
    assert pre.image.shape == image.shape
    assert post.image.shape == (192, 256, 3)
    assert post.scale == 2.0


def test_an_upscaled_job_needs_an_upscaler() -> None:
    pipeline = PhotoRestorePipeline(Journal().runners(), limits=LIMITS)

    with pytest.raises(ValueError, match="upscaler"):
        pipeline.run(request(photo(), ("tone",), scale=2.0, upscale_mode="classic"))


@pytest.mark.parametrize(("scale", "mode"), [(2.0, "none"), (1.0, "ai"), (0.5, "none"), (2.0, "magic")])
def test_scale_and_upscale_mode_must_agree(scale: float, mode: str) -> None:
    pipeline = PhotoRestorePipeline(Journal().runners(), limits=LIMITS)

    with pytest.raises(ValueError):
        pipeline.run_pre(request(photo(), ("tone",), scale=scale, upscale_mode=mode))


def test_the_input_pixel_cap_is_a_too_large_error() -> None:
    pipeline = PhotoRestorePipeline(Journal().runners(), limits=PixelLimits(96 * 128 - 1, 10**9))

    with pytest.raises(RestoreTooLarge) as raised:
        pipeline.run_pre(request(photo(), ("tone",)))

    assert raised.value.code == "restore.error.tooLarge"


def test_the_output_pixel_cap_counts_the_upscale() -> None:
    pipeline = PhotoRestorePipeline(Journal().runners(), limits=PixelLimits(10**9, 96 * 128 * 4 - 1))

    with pytest.raises(RestoreTooLarge):
        pipeline.run_pre(request(photo(), ("tone",), scale=2.0, upscale_mode="classic"))


def test_unknown_steps_and_missing_runners_are_rejected() -> None:
    journal = Journal()
    runners = journal.runners()
    del runners["tone"]
    pipeline = PhotoRestorePipeline(runners, limits=LIMITS)

    with pytest.raises(UnknownRestoreStep):
        pipeline.run_pre(request(photo(), ("sharpen",)))
    with pytest.raises(ValueError, match="No runner"):
        pipeline.run_pre(request(photo(), ("tone",)))


def test_preview_crop_processes_only_the_area_plus_its_margin() -> None:
    journal = Journal()
    pipeline = PhotoRestorePipeline(journal.runners(), limits=LIMITS)
    image = photo(400, 600)
    crop = (100, 150, 200, 120)

    pre, post = pipeline.run(request(image, ("deblock", "tone"), preview_crop=crop))

    region_shape = (120 + 2 * PREVIEW_MARGIN_PX, 200 + 2 * PREVIEW_MARGIN_PX, 3)
    assert journal.calls[0] == ("deblock", region_shape)
    assert pre.region == (36, 86, 364, 334)
    assert post.preview.shape == (120, 200, 3)
    np.testing.assert_allclose(post.preview, post.image[64:184, 64:264])


def test_preview_crop_is_clamped_at_the_photo_border_and_scaled_with_the_upscale() -> None:
    pipeline = PhotoRestorePipeline(Journal().runners(), limits=LIMITS)
    image = photo(300, 300)

    def double(source: np.ndarray, cancel) -> np.ndarray:
        return cv2.resize(source, (source.shape[1] * 2, source.shape[0] * 2), interpolation=cv2.INTER_NEAREST)

    job = request(image, ("deblock",), preview_crop=(0, 0, 100, 50), scale=2.0, upscale_mode="classic")

    pre, post = pipeline.run(job, upscale=double)

    assert pre.region == (0, 0, 164, 114)
    assert post.preview.shape == (100, 200, 3)


@pytest.mark.parametrize("crop", [(0, 0, 513, 100), (250, 0, 100, 100), (-1, 0, 10, 10), (0, 0, 0, 10)])
def test_invalid_preview_crops_are_rejected(crop: tuple[int, int, int, int]) -> None:
    pipeline = PhotoRestorePipeline(Journal().runners(), limits=LIMITS)

    with pytest.raises(InvalidPreviewCrop):
        pipeline.run_pre(request(photo(300, 300), ("tone",), preview_crop=crop))


def test_preview_crop_writes_no_recompose_artifacts(tmp_path: Path) -> None:
    pipeline = PhotoRestorePipeline(Journal().runners(), limits=LIMITS)
    image = photo(300, 300)

    job = request(image, ("faces", "colorize"), faces=(face(0),), preview_crop=(0, 0, 150, 150))

    _, post = pipeline.run(job, artifact_dir=tmp_path / "job.restore")

    assert post.artifacts is None
    assert not (tmp_path / "job.restore").exists()


def test_faces_receive_landmarks_scaled_to_the_working_copy() -> None:
    journal = Journal()
    pipeline = PhotoRestorePipeline(journal.runners(), limits=LIMITS)
    measured = face(0, offset=(5.0, 4.0), scale=0.06, reference_size=(48, 64))

    pipeline.run(request(photo(96, 128), ("faces",), faces=(measured,)))

    received = journal.faces[0][0]
    np.testing.assert_allclose(received.landmarks, np.asarray(measured.landmarks) * 2.0)
    np.testing.assert_allclose(received.box, np.asarray(measured.box) * 2.0)
    assert received.eye_px == measured.eye_px * 2.0


def test_faces_in_a_preview_are_moved_into_the_area_and_the_others_dropped() -> None:
    journal = Journal()
    pipeline = PhotoRestorePipeline(journal.runners(), limits=LIMITS)
    inside, outside = face(0, offset=(210.0, 220.0), scale=0.12), face(1, offset=(10.0, 10.0), scale=0.12)

    pipeline.run(request(photo(400, 400), ("faces",), faces=(inside, outside), preview_crop=(200, 200, 150, 150)))

    received = journal.faces[0]
    assert [f.index for f in received] == [0]
    np.testing.assert_allclose(received[0].landmarks, np.asarray(inside.landmarks) - 136.0)


def test_a_face_measured_on_another_aspect_ratio_is_rejected() -> None:
    pipeline = PhotoRestorePipeline(Journal().runners(), limits=LIMITS)
    wrong = face(0, reference_size=(48, 48))

    with pytest.raises(ValueError, match="not a rescale"):
        pipeline.run_pre(request(photo(96, 128), ("faces",), faces=(wrong,)))


def test_output_steps_save_the_recompose_artifacts(tmp_path: Path) -> None:
    pipeline = PhotoRestorePipeline(Journal().runners(), limits=LIMITS)
    directory = tmp_path / "job.restore"

    pre, post = pipeline.run(request(photo(), ("faces", "colorize"), faces=(face(0),)), artifact_dir=directory)

    assert (directory / FACES_MANIFEST_NAME).is_file()
    assert (directory / AB_ARTIFACT_NAME).is_file()
    assert post.artifacts.available
    assert post.before_faces is pre.image
    np.testing.assert_array_equal(post.uncolored, post.faces.image)
    assert restore_metadata(pre, post)["colorize"]["abReusedOnRecompose"] is True


def test_the_restore_metadata_is_complete() -> None:
    journal = Journal()
    runners = journal.runners(
        repair=journal.runner(
            "repair",
            model=ModelUse(MIGAN_MODEL_ID, "cpu", "fp32", 512),
            details={"finalCoverage": 0.002, "touchesFaces": True, "largeHoles": 0},
            warnings=("large_holes",),
        )
    )
    pipeline = PhotoRestorePipeline(runners, limits=LIMITS)

    pre, post = pipeline.run(request(photo(), ALL_STEPS, faces=(face(0), face(1, enabled=False))))
    metadata = restore_metadata(pre, post)

    assert [step["id"] for step in metadata["steps"]] == list(ALL_STEPS)
    assert metadata["steps"][1]["model"] == {"id": MIGAN_MODEL_ID, "device": "cpu", "precision": "fp32", "tile": 512}
    assert metadata["steps"][4]["strategy"] == "dsp"
    assert metadata["damage"] == {"finalCoverage": 0.002, "touchesFaces": True, "largeHoles": 0}
    faces = [(f["index"], f["enabled"], f["restored"]) for f in metadata["faces"]]
    assert faces == [(0, True, True), (1, False, False)]
    assert metadata["colorize"]["abReusedOnRecompose"] is False
    assert metadata["calibration"][MIGAN_MODEL_ID]["tile"] == 512
    assert metadata["warnings"] == ["large_holes"]
    assert metadata["scale"] == 1.0
    assert set(metadata) >= {"cpuFallback", "previewCrop", "recomposeAvailable"}


def test_real_tone_runner_matches_the_whole_image_tone() -> None:
    pipeline = PhotoRestorePipeline({"tone": run_tone}, limits=LIMITS)
    image = photo(200, 150)

    pre = pipeline.run_pre(request(image, ("tone",), params={"tone": {"keep_tone": True}}))

    np.testing.assert_allclose(pre.image, apply_tone(image, ToneSettings()), atol=2e-6)


@pytest.mark.parametrize(
    ("details", "upscale", "expected"),
    [
        ({"finalCoverage": 0.004, "touchesFaces": False}, UpscaleInfo(), DIGITAL_SOURCE_ENHANCED),
        ({"finalCoverage": 0.0005, "touchesFaces": True}, UpscaleInfo(), DIGITAL_SOURCE_COMPOSITE),
        ({"finalCoverage": 0.02, "touchesFaces": False}, UpscaleInfo(), DIGITAL_SOURCE_COMPOSITE),
        ({"finalCoverage": 0.004}, UpscaleInfo("ai", 2.0, "realesrgan-x2", True), DIGITAL_SOURCE_COMPOSITE),
        ({"finalCoverage": 0.004}, UpscaleInfo("ai", 2.0, "swinir", False), DIGITAL_SOURCE_ENHANCED),
    ],
)
def test_the_digital_source_type_follows_the_fill_and_the_upscale(
    details: dict, upscale: UpscaleInfo, expected: str
) -> None:
    journal = Journal()
    pipeline = PhotoRestorePipeline(journal.runners(repair=journal.runner("repair", details=details)), limits=LIMITS)

    pre, post = pipeline.run(request(photo(), ("repair", "tone")))

    assert digital_source_type(facts_from_records((*pre.records, *post.records), upscale)) == expected


def test_the_sidecar_hashes_the_input_and_outputs_and_names_the_licenses(tmp_path: Path) -> None:
    journal = Journal()
    migan = ModelUse(MIGAN_MODEL_ID, "cpu", "fp32")
    runners = journal.runners(repair=journal.runner("repair", model=migan, details={"finalCoverage": 0.001}))
    pipeline = PhotoRestorePipeline(runners, limits=LIMITS)
    source = tmp_path / "abuela.jpg"
    source.write_bytes(b"scan")
    pre, post = pipeline.run(request(photo(), ("repair", "tone")))
    output = tmp_path / "job.png"
    output.write_bytes(b"result")
    context = SidecarContext(
        input=InputInfo("abuela.jpg", sha256_file(source), 4, 8, False, 1),
        outputs=(OutputFile("restored", output),),
        output_bit_depth=8,
        app_version="0.99.0",
    )

    sidecar = build_sidecar(pre, post, context, default_model_catalog())

    assert sidecar["input"]["sha256"] == hashlib.sha256(b"scan").hexdigest()
    assert sidecar["outputs"][0]["sha256"] == hashlib.sha256(b"result").hexdigest()
    assert sidecar["steps"][0]["model"]["license"]["spdx"] == "MIT"
    assert sidecar["digitalSourceType"] == DIGITAL_SOURCE_ENHANCED
    assert isinstance(ModelCatalog({}, {}).license_of("missing"), type(None))


def test_peak_memory_stays_under_six_float32_copies_of_an_eight_megapixel_photo() -> None:
    image = np.ascontiguousarray(np.tile(photo(612, 816), (4, 4, 1)))
    assert image.shape[0] * image.shape[1] >= 7_990_000

    def lean(image: np.ndarray, call: StepCall) -> StepOutcome:
        output = image.copy()
        output *= np.float32(0.99)
        return StepOutcome(output)

    runners = {"deblock": lean, "denoise": lean, "tone": run_tone, "faces": fake_faces, "colorize": fake_colorize}
    pipeline = PhotoRestorePipeline(runners, limits=LIMITS)
    big_face = face(0, offset=(900.0, 700.0), scale=0.5)
    job = request(image, ("deblock", "denoise", "tone", "faces", "colorize"), faces=(big_face,))

    tracemalloc.start()
    try:
        pipeline.run(job)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 6 * image.nbytes, f"peak {peak / image.nbytes:.2f} copies"


def test_recompose_is_unavailable_when_the_colors_could_not_be_saved(tmp_path: Path) -> None:
    pipeline = PhotoRestorePipeline(Journal().runners(), limits=LIMITS)
    directory = tmp_path / "job.restore"
    (directory / AB_ARTIFACT_NAME).mkdir(parents=True)

    pre, post = pipeline.run(request(photo(), ("faces", "colorize"), faces=(face(0),)), artifact_dir=directory)
    metadata = restore_metadata(pre, post)

    assert (directory / FACES_MANIFEST_NAME).is_file()
    assert not post.artifacts.available
    assert post.artifacts.warning == "recompose_disabled_write_failed"
    assert metadata["colorize"]["abReusedOnRecompose"] is False
    assert metadata["recomposeAvailable"] is False
    assert metadata["warnings"] == ["recompose_disabled_write_failed"]


def test_no_recompose_artifacts_are_needed_without_restored_faces(tmp_path: Path) -> None:
    pipeline = PhotoRestorePipeline(Journal().runners(), limits=LIMITS)

    _, post = pipeline.run(request(photo(), ("colorize",)), artifact_dir=tmp_path / "job.restore")

    assert post.artifacts is None
    assert post.saved_artifacts == ("ab",)
