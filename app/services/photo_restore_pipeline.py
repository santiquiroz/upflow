from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from app.services.engines.colorize import RENDER_SIZE, save_ab_artifact
from app.services.engines.drunet_restore import CpuFallback
from app.services.engines.face_restore import (
    FaceRestoreResult,
    RecomposeArtifacts,
    output_scale,
    save_recompose_artifacts,
)
from app.services.engines.tiled_restore_runner import Clock, RestoreCancelled
from app.services.photo_dsp import PeriodicPeaks
from app.services.photo_geometry import Crop
from app.services.photo_restore_chain import RESTORE_CHAIN, RestoreStepSpec, steps_from_selection
from app.services.photo_restore_chain import STEP_STAGES as CHAIN_STEP_STAGES
from app.services.photo_restore_presets import ToneKind

PREVIEW_MARGIN_PX = 64
PREVIEW_MAX_SIDE = 512
TOO_LARGE_CODE = "restore.error.tooLarge"
UPSCALE_MODES = ("none", "classic", "ai")
GRAY_POINT_OUTSIDE_PREVIEW = "grayPointOutsidePreview"
FACES_ARTIFACT = "faces"
AB_ARTIFACT = "ab"
ARTIFACT_OF_STEP: Mapping[str, str] = MappingProxyType({"faces": FACES_ARTIFACT, "colorize": AB_ARTIFACT})
RGB_CHANNELS = 3

STEP_STAGES: Mapping[str, tuple[str, ...]] = MappingProxyType(CHAIN_STEP_STAGES)
STAGE_ORDER: tuple[str, ...] = tuple(stage for step in RESTORE_CHAIN for stage in STEP_STAGES[step.id])

Point = tuple[float, float]
FaceBox = tuple[float, float, float, float]
Region = tuple[int, int, int, int]
StageProgress = Callable[[str, int, int], None]


class RestoreTooLarge(ValueError):
    code = TOO_LARGE_CODE


class InvalidPreviewCrop(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PixelLimits:
    max_input_pixels: int
    max_output_pixels: int

    @classmethod
    def from_settings(cls, settings: Any) -> PixelLimits:
        return cls(settings.restore_max_input_pixels, settings.restore_max_output_pixels)


@dataclass(frozen=True, slots=True)
class FaceSelection:
    index: int
    landmarks: tuple[Point, ...]
    blend: float
    enabled: bool = True
    box: FaceBox | None = None
    score: float | None = None
    eye_px: float | None = None
    sharpness: float | None = None
    # (alto, ancho) de la imagen donde se midieron los puntos; None = la copia de trabajo.
    reference_size: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class RestoreHints:
    noise_sigma: float = 0.0
    pattern_kind: str | None = None
    pattern_period: float | None = None
    peaks: PeriodicPeaks | None = None
    damage_probability: np.ndarray | None = None
    damage_detector: ModelUse | None = None
    user_mask: np.ndarray | None = None
    face_detector: ModelUse | None = None


@dataclass(frozen=True, slots=True)
class RestoreRequest:
    image: np.ndarray
    steps: tuple[str, ...]
    tone_kind: ToneKind
    device: str
    params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    bit_depth: int = 8
    scale: float = 1.0
    upscale_mode: str = "none"
    faces: tuple[FaceSelection, ...] = ()
    hints: RestoreHints = RestoreHints()
    preview_crop: Crop | None = None

    def params_for(self, step_id: str) -> Mapping[str, Any]:
        return self.params.get(step_id, {})


@dataclass(frozen=True, slots=True)
class StepCall:
    spec: RestoreStepSpec
    request: RestoreRequest
    source: np.ndarray | None
    cancel_event: threading.Event | None
    progress: StageProgress

    @property
    def params(self) -> Mapping[str, Any]:
        return self.request.params_for(self.spec.id)


@dataclass(frozen=True, slots=True)
class ModelUse:
    model_id: str
    device: str
    precision: str
    tile: int | None = None

    def to_metadata(self) -> dict[str, object]:
        return {"id": self.model_id, "device": self.device, "precision": self.precision, "tile": self.tile}


@dataclass(frozen=True, slots=True)
class StepOutcome:
    image: np.ndarray
    model: ModelUse | None = None
    aux_models: tuple[ModelUse, ...] = ()
    tiles: int | None = None
    warnings: tuple[str, ...] = ()
    cpu_fallbacks: tuple[CpuFallback, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
    invents_detail: bool | None = None
    faces: FaceRestoreResult | None = None
    ab_512: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class StepRecord:
    step_id: str
    strategy: str
    params: Mapping[str, Any]
    seconds: float
    invents_detail: bool
    model: ModelUse | None = None
    aux_models: tuple[ModelUse, ...] = ()
    tiles: int | None = None
    warnings: tuple[str, ...] = ()
    cpu_fallbacks: tuple[CpuFallback, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, object]:
        return {
            "id": self.step_id,
            "strategy": self.strategy,
            "model": None if self.model is None else self.model.to_metadata(),
            "auxiliaryModels": [model.to_metadata() for model in self.aux_models],
            "params": dict(self.params),
            "seconds": round(self.seconds, 3),
            "tiles": self.tiles,
            "inventsDetail": self.invents_detail,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class PreResult:
    image: np.ndarray
    records: tuple[StepRecord, ...]
    request: RestoreRequest
    region: Region | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PostResult:
    image: np.ndarray
    records: tuple[StepRecord, ...]
    scale: float
    uncolored: np.ndarray | None = None
    before_faces: np.ndarray | None = None
    faces: FaceRestoreResult | None = None
    ab_512: np.ndarray | None = None
    artifacts: RecomposeArtifacts | None = None
    saved_artifacts: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    preview: np.ndarray | None = None


StepRunner = Callable[[np.ndarray, StepCall], StepOutcome]
Upscaler = Callable[[np.ndarray, threading.Event | None], np.ndarray]


def check_pixel_limits(height: int, width: int, scale: float, limits: PixelLimits) -> None:
    if height * width > limits.max_input_pixels:
        raise RestoreTooLarge(
            f"The photo has {height * width:,} pixels; the limit is {limits.max_input_pixels:,}."
        )
    output = round(height * scale) * round(width * scale)
    if output > limits.max_output_pixels:
        raise RestoreTooLarge(
            f"At {scale}x the result would have {output:,} pixels; the limit is {limits.max_output_pixels:,}."
        )


def check_upscale(scale: float, upscale_mode: str) -> None:
    if upscale_mode not in UPSCALE_MODES:
        raise ValueError(f"Unknown upscale mode {upscale_mode!r}. Valid modes: {', '.join(UPSCALE_MODES)}.")
    if not math.isfinite(scale) or scale < 1:
        raise ValueError(f"Scale must be at least 1, got {scale}")
    if (upscale_mode == "none") != (scale == 1):
        raise ValueError(f"Upscale mode {upscale_mode!r} does not match scale {scale}")


def needs_upscale(request: RestoreRequest) -> bool:
    return request.upscale_mode != "none" and request.scale > 1


def check_request(request: RestoreRequest, limits: PixelLimits) -> None:
    _require_rgb_float(request.image)
    steps_from_selection(request.steps)
    check_upscale(request.scale, request.upscale_mode)
    height, width = request.image.shape[:2]
    check_pixel_limits(height, width, request.scale, limits)
    if request.preview_crop is not None:
        preview_region((height, width), request.preview_crop)


def preview_region(shape: tuple[int, int], crop: Crop) -> Region:
    x, y, width, height = crop
    image_height, image_width = shape
    if width <= 0 or height <= 0 or max(width, height) > PREVIEW_MAX_SIDE:
        raise InvalidPreviewCrop(
            f"The preview area must be at most {PREVIEW_MAX_SIDE}x{PREVIEW_MAX_SIDE} px, got {width}x{height}"
        )
    if x < 0 or y < 0 or x + width > image_width or y + height > image_height:
        raise InvalidPreviewCrop(f"The preview area {crop} is outside the {image_width}x{image_height} photo")
    margin = PREVIEW_MARGIN_PX
    return (
        max(0, x - margin),
        max(0, y - margin),
        min(image_width, x + width + margin),
        min(image_height, y + height + margin),
    )


def crop_region(array: np.ndarray | None, region: Region) -> np.ndarray | None:
    if array is None:
        return None
    x0, y0, x1, y1 = region
    return np.array(array[y0:y1, x0:x1], copy=True)


def preview_output(image: np.ndarray, region: Region, crop: Crop, scale: float) -> np.ndarray:
    x, y, width, height = crop
    left, top = round((x - region[0]) * scale), round((y - region[1]) * scale)
    return np.array(image[top : top + round(height * scale), left : left + round(width * scale)], copy=True)


def faces_for_image(faces: Sequence[FaceSelection], shape: tuple[int, int]) -> tuple[FaceSelection, ...]:
    return tuple(_face_in_image_coords(face, shape) for face in faces)


def faces_in_region(faces: Sequence[FaceSelection], region: Region) -> tuple[FaceSelection, ...]:
    shifted = (_translated_face(face, -region[0], -region[1]) for face in faces)
    width, height = region[2] - region[0], region[3] - region[1]
    return tuple(face for face in shifted if _landmarks_inside(face.landmarks, width, height))


def request_for_region(request: RestoreRequest, region: Region) -> tuple[RestoreRequest, tuple[str, ...]]:
    params, warnings = _params_for_region(request.params, region)
    hints = replace(
        request.hints,
        damage_probability=crop_region(request.hints.damage_probability, region),
        user_mask=crop_region(request.hints.user_mask, region),
    )
    cropped = replace(
        request,
        image=crop_region(request.image, region),
        params=params,
        hints=hints,
        faces=faces_in_region(request.faces, region),
    )
    return cropped, warnings


def calibration_metadata(records: Sequence[StepRecord]) -> dict[str, dict[str, object]]:
    models = (model for record in records for model in (record.model, *record.aux_models) if model is not None)
    return {m.model_id: {"tile": m.tile, "precision": m.precision, "device": m.device} for m in models}


def restore_metadata(pre: PreResult, post: PostResult) -> dict[str, object]:
    records = (*pre.records, *post.records)
    return {
        "steps": [record.to_metadata() for record in records],
        "scale": post.scale,
        "damage": _step_details(records, "repair"),
        "faces": faces_metadata(pre.request.faces, records),
        "colorize": colorize_metadata(records, post),
        "cpuFallback": [fallback.to_metadata() for record in records for fallback in record.cpu_fallbacks],
        "calibration": calibration_metadata(records),
        "warnings": [*pre.warnings, *(w for record in records for w in record.warnings), *post.warnings],
        "previewCrop": None if pre.request.preview_crop is None else list(pre.request.preview_crop),
        "recomposeAvailable": post.artifacts is not None and post.artifacts.available,
    }


def colorize_metadata(records: Sequence[StepRecord], post: PostResult) -> dict[str, object] | None:
    details = _step_details(records, "colorize")
    if details is None:
        return None
    return {**details, "renderSize": RENDER_SIZE, "abReusedOnRecompose": AB_ARTIFACT in post.saved_artifacts}


def faces_metadata(selections: Sequence[FaceSelection], records: Sequence[StepRecord]) -> list[dict[str, object]]:
    restored = set((_step_details(records, "faces") or {}).get("restored", ()))
    return [_face_metadata(face, face.index in restored) for face in selections]


class MonotonicProgress:
    def __init__(self, sink: StageProgress | None) -> None:
        self._sink = sink
        self._position = (-1, 0)
        self._totals: dict[str, int] = {}

    def report(self, stage: str, done: int, total: int) -> None:
        position = (STAGE_ORDER.index(stage), done)
        if position < self._position:
            return
        self._position = position
        self._totals[stage] = max(total, 1)
        if self._sink is not None:
            self._sink(stage, done, total)

    def finish(self, stage: str) -> None:
        total = self._totals.get(stage, 1)
        self.report(stage, total, total)


class PhotoRestorePipeline:
    def __init__(
        self,
        runners: Mapping[str, StepRunner],
        *,
        limits: PixelLimits,
        on_progress: StageProgress | None = None,
        clock: Clock = time.perf_counter,
    ) -> None:
        self._runners = runners
        self._limits = limits
        self._progress = MonotonicProgress(on_progress)
        self._clock = clock

    def run_pre(self, request: RestoreRequest, cancel_event: threading.Event | None = None) -> PreResult:
        check_request(request, self._limits)
        working, region, warnings = _working_request(request)
        image, records = self._run_steps(working.image, _phase_specs(request, "native"), working, None, cancel_event)
        return PreResult(image, records, working, region, warnings)

    def run_post(
        self,
        pre: PreResult,
        upscaled: np.ndarray | None = None,
        cancel_event: threading.Event | None = None,
        artifact_dir: Path | None = None,
    ) -> PostResult:
        base = pre.image if upscaled is None else upscaled
        _require_rgb_float(base)
        directory = artifact_dir if pre.region is None else None
        state = _PostState(image=base, scale=output_scale(pre.image.shape, base.shape))
        for spec in _phase_specs(pre.request, "output"):
            _raise_if_cancelled(cancel_event)
            state = self._run_output_step(spec, state, pre, cancel_event, directory)
        return _post_result(state, pre, directory)

    def run(
        self,
        request: RestoreRequest,
        upscale: Upscaler | None = None,
        cancel_event: threading.Event | None = None,
        artifact_dir: Path | None = None,
    ) -> tuple[PreResult, PostResult]:
        pre = self.run_pre(request, cancel_event)
        upscaled = _upscaled(pre, upscale, cancel_event)
        return pre, self.run_post(pre, upscaled, cancel_event, artifact_dir)

    def _run_steps(
        self,
        image: np.ndarray,
        specs: Sequence[RestoreStepSpec],
        request: RestoreRequest,
        source: np.ndarray | None,
        cancel_event: threading.Event | None,
    ) -> tuple[np.ndarray, tuple[StepRecord, ...]]:
        records = []
        for spec in specs:
            _raise_if_cancelled(cancel_event)
            call = StepCall(spec, request, source, cancel_event, self._progress.report)
            outcome, record = self._run_step(spec, image, call)
            image = outcome.image
            records.append(record)
        return image, tuple(records)

    def _run_output_step(
        self,
        spec: RestoreStepSpec,
        state: _PostState,
        pre: PreResult,
        cancel_event: threading.Event | None,
        directory: Path | None,
    ) -> _PostState:
        call = StepCall(spec, pre.request, pre.image, cancel_event, self._progress.report)
        outcome, record = self._run_step(spec, state.image, call)
        absorbed = _absorbed(state, spec.id, outcome, record)
        return _with_saved_artifacts(absorbed, spec.id, directory)

    def _run_step(self, spec: RestoreStepSpec, image: np.ndarray, call: StepCall) -> tuple[StepOutcome, StepRecord]:
        runner = self._runners.get(spec.id)
        if runner is None:
            raise ValueError(f"No runner for restore step {spec.id!r}")
        stages = STEP_STAGES[spec.id]
        self._progress.report(stages[0], 0, 1)
        started = self._clock()
        outcome = runner(image, call)
        _require_rgb_float(outcome.image)
        self._progress.finish(stages[-1])
        return outcome, step_record(spec, call.params, outcome, self._clock() - started)


def step_record(spec: RestoreStepSpec, params: Mapping[str, Any], outcome: StepOutcome, seconds: float) -> StepRecord:
    invents = spec.invents_detail if outcome.invents_detail is None else outcome.invents_detail
    return StepRecord(
        step_id=spec.id,
        strategy="model" if outcome.model is not None else "dsp",
        params=MappingProxyType(dict(params)),
        seconds=seconds,
        invents_detail=invents,
        model=outcome.model,
        aux_models=outcome.aux_models,
        tiles=outcome.tiles,
        warnings=outcome.warnings,
        cpu_fallbacks=outcome.cpu_fallbacks,
        details=MappingProxyType(dict(outcome.details)),
    )


@dataclass(frozen=True, slots=True)
class _PostState:
    image: np.ndarray
    scale: float
    records: tuple[StepRecord, ...] = ()
    before_faces: np.ndarray | None = None
    faces: FaceRestoreResult | None = None
    uncolored: np.ndarray | None = None
    ab_512: np.ndarray | None = None
    saved: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def _absorbed(state: _PostState, step_id: str, outcome: StepOutcome, record: StepRecord) -> _PostState:
    return replace(
        state,
        image=outcome.image,
        records=(*state.records, record),
        before_faces=state.image if step_id == "faces" else state.before_faces,
        faces=outcome.faces if outcome.faces is not None else state.faces,
        uncolored=state.image if step_id == "colorize" else state.uncolored,
        ab_512=outcome.ab_512 if outcome.ab_512 is not None else state.ab_512,
    )


def _with_saved_artifacts(state: _PostState, step_id: str, directory: Path | None) -> _PostState:
    saved = _save_step_artifacts(state, step_id, directory)
    if saved is None:
        return state
    kinds = state.saved if not saved.available else (*state.saved, ARTIFACT_OF_STEP[step_id])
    warnings = state.warnings if saved.warning is None else (*state.warnings, saved.warning)
    return replace(state, saved=kinds, warnings=warnings)


def _save_step_artifacts(state: _PostState, step_id: str, directory: Path | None) -> RecomposeArtifacts | None:
    if directory is None:
        return None
    if step_id == "faces" and state.faces is not None and state.faces.faces:
        return save_recompose_artifacts(directory, state.before_faces, state.faces)
    if step_id == "colorize" and state.ab_512 is not None:
        return save_ab_artifact(directory, state.ab_512)
    return None


def _post_result(state: _PostState, pre: PreResult, directory: Path | None) -> PostResult:
    preview = None
    if pre.region is not None and pre.request.preview_crop is not None:
        preview = preview_output(state.image, pre.region, pre.request.preview_crop, state.scale)
    return PostResult(
        image=state.image,
        records=state.records,
        scale=state.scale,
        uncolored=state.uncolored,
        before_faces=state.before_faces,
        faces=state.faces,
        ab_512=state.ab_512,
        artifacts=_recompose_artifacts(state, directory),
        saved_artifacts=state.saved,
        warnings=state.warnings,
        preview=preview,
    )


def _recompose_artifacts(state: _PostState, directory: Path | None) -> RecomposeArtifacts | None:
    # Recomponer necesita los recortes de las caras y, si hubo color, el ab guardado.
    needed = _needed_artifacts(state)
    if directory is None or not needed:
        return None
    if all(kind in state.saved for kind in needed):
        return RecomposeArtifacts(directory)
    return RecomposeArtifacts(None, state.warnings[0] if state.warnings else None)


def _needed_artifacts(state: _PostState) -> tuple[str, ...]:
    if state.faces is None or not state.faces.faces:
        return ()
    return (FACES_ARTIFACT, AB_ARTIFACT) if state.ab_512 is not None else (FACES_ARTIFACT,)


def _upscaled(pre: PreResult, upscale: Upscaler | None, cancel_event: threading.Event | None) -> np.ndarray | None:
    if not needs_upscale(pre.request):
        return None
    if upscale is None:
        raise ValueError(f"Scale {pre.request.scale}x needs an upscaler")
    _raise_if_cancelled(cancel_event)
    return upscale(pre.image, cancel_event)


def _working_request(request: RestoreRequest) -> tuple[RestoreRequest, Region | None, tuple[str, ...]]:
    scaled = replace(request, faces=faces_for_image(request.faces, request.image.shape[:2]))
    if request.preview_crop is None:
        return scaled, None, ()
    region = preview_region(request.image.shape[:2], request.preview_crop)
    cropped, warnings = request_for_region(scaled, region)
    return cropped, region, warnings


def _phase_specs(request: RestoreRequest, phase: str) -> list[RestoreStepSpec]:
    return [spec for spec in steps_from_selection(request.steps) if spec.phase == phase]


def _params_for_region(
    params: Mapping[str, Mapping[str, Any]], region: Region
) -> tuple[Mapping[str, Mapping[str, Any]], tuple[str, ...]]:
    tone = params.get("tone", {})
    point = tone.get("gray_point")
    if point is None:
        return params, ()
    x, y = point[0] - region[0], point[1] - region[1]
    inside = 0 <= x < region[2] - region[0] and 0 <= y < region[3] - region[1]
    moved = {**tone, "gray_point": (x, y) if inside else None}
    return {**params, "tone": moved}, () if inside else (GRAY_POINT_OUTSIDE_PREVIEW,)


def _face_in_image_coords(face: FaceSelection, shape: tuple[int, int]) -> FaceSelection:
    if face.reference_size is None or tuple(face.reference_size) == tuple(shape):
        return replace(face, reference_size=None)
    factor = shape[0] / face.reference_size[0]
    if abs(face.reference_size[1] * factor - shape[1]) > 1:
        raise ValueError(
            f"Face {face.index} was measured on a {face.reference_size} image that is not a rescale of {shape}"
        )
    return _scaled_face(face, factor)


def _scaled_face(face: FaceSelection, factor: float) -> FaceSelection:
    return replace(
        face,
        landmarks=tuple((x * factor, y * factor) for x, y in face.landmarks),
        box=None if face.box is None else tuple(value * factor for value in face.box),
        eye_px=None if face.eye_px is None else face.eye_px * factor,
        reference_size=None,
    )


def _translated_face(face: FaceSelection, dx: float, dy: float) -> FaceSelection:
    box = None if face.box is None else (face.box[0] + dx, face.box[1] + dy, face.box[2] + dx, face.box[3] + dy)
    return replace(face, landmarks=tuple((x + dx, y + dy) for x, y in face.landmarks), box=box)


def _landmarks_inside(landmarks: Sequence[Point], width: int, height: int) -> bool:
    return all(0 <= x < width and 0 <= y < height for x, y in landmarks)


def _step_details(records: Sequence[StepRecord], step_id: str) -> dict[str, object] | None:
    record = next((record for record in records if record.step_id == step_id), None)
    return None if record is None else dict(record.details)


def _face_metadata(face: FaceSelection, restored: bool) -> dict[str, object]:
    return {
        "index": face.index,
        "box": None if face.box is None else [round(value, 2) for value in face.box],
        "eyePx": face.eye_px,
        "sharpness": face.sharpness,
        "confidence": face.score,
        "enabled": face.enabled,
        "restored": restored,
        "blend": face.blend,
        "recomposedAt": None,
    }


def _require_rgb_float(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != RGB_CHANNELS or image.dtype != np.float32:
        raise ValueError(f"Restore needs a float32 RGB image, got {image.dtype} {image.shape}")


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RestoreCancelled("Restoration cancelled")
