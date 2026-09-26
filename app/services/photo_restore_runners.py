from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from app.services.engines.colorize import COLORIZE_MODEL_ID, ColorizeContext, ColorizeOptions, colorize
from app.services.engines.drunet_restore import (
    DrunetContext,
    DrunetResult,
    deblock_photo,
    denoise_photo,
)
from app.services.engines.face_detect import (
    DETECT_DEVICE,
    DETECT_PRECISION,
    RETINAFACE_MODEL_ID,
    FaceDetection,
    landmarked_face_detector,
)
from app.services.engines.face_restore import FACE_MODEL_ID, FaceRestoreContext, FaceTarget, restore_faces
from app.services.engines.photo_restore_engine import PhotoRestoreEngine
from app.services.engines.scratch_detect import (
    DETECTOR_DEVICE,
    DETECTOR_PRECISION,
    SCRATCH_MODEL_ID,
    damage_mask,
    scratch_detector,
)
from app.services.engines.scratch_fill import (
    FAST_ENGINE,
    MIGAN_PRECISION,
    FillRequest,
    FillResult,
    fill_damage,
    migan_infer,
)
from app.services.engines.tiled_restore_runner import CalibrationCache, Clock, TileInfer, TileProgress
from app.services.face_geometry import align_face, align_matrix, default_face_policy, face_sharpness
from app.services.photo_diagnosis import DEBLOCK_DEFAULT_STRENGTH, DESCREEN_DEFAULT_STRENGTH, PORTRAIT_BLEND
from app.services.photo_dsp import ToneSettings, apply_tone_in_bands, descreen_halftone, descreen_notch
from app.services.photo_restore_chain import resolve_tone_flags
from app.services.photo_restore_pipeline import (
    FaceSelection,
    ModelUse,
    RestoreHints,
    StepCall,
    StepOutcome,
    StepRunner,
)
from app.services.photo_restore_presets import KEEP_GRAIN_DEFAULT, REPAIR_SENSITIVITY_MEDIUM
from app.services.restore_models import MIGAN_MODEL_ID

DESCREEN_MODES = ("halftone", "texture")
DENOISE_DEFAULT_STRENGTH = 0.3
TONE_DEFAULT_STRENGTH = 0.7


@dataclass(frozen=True, slots=True)
class RunnerDeps:
    engine: PhotoRestoreEngine
    calibrations: CalibrationCache
    budget_ms: float
    clock: Clock = time.perf_counter


def build_step_runners(deps: RunnerDeps) -> dict[str, StepRunner]:
    return {
        "descreen": run_descreen,
        "repair": partial(run_repair, deps),
        "deblock": partial(run_deblock, deps),
        "denoise": partial(run_denoise, deps),
        "tone": run_tone,
        "faces": partial(run_faces, deps),
        "colorize": partial(run_colorize, deps),
    }


def run_descreen(image: np.ndarray, call: StepCall) -> StepOutcome:
    hints = call.request.hints
    mode = descreen_mode(call.params.get("mode", "auto"), hints)
    strength = float(call.params.get("strength", DESCREEN_DEFAULT_STRENGTH))
    if mode == "halftone":
        output = descreen_halftone(image, hints.pattern_period, strength)
    else:
        output = descreen_notch(image, hints.peaks, strength)
    return StepOutcome(output, details={"mode": mode})


def descreen_mode(requested: str, hints: RestoreHints) -> str:
    if requested in DESCREEN_MODES:
        return requested
    if requested != "auto":
        raise ValueError(f"Unknown descreen mode {requested!r}")
    return hints.pattern_kind if hints.pattern_kind in DESCREEN_MODES else "texture"


@dataclass(frozen=True, slots=True)
class RepairMask:
    mask: np.ndarray
    detected_coverage: float | None
    user_edited: bool
    detector: tuple[ModelUse, ...] = ()


def run_repair(deps: RunnerDeps, image: np.ndarray, call: StepCall) -> StepOutcome:
    chosen = repair_mask(deps, image, call)
    request = FillRequest(
        engine=str(call.params.get("engine", FAST_ENGINE)),
        leave_large_holes=bool(call.params.get("leave_large_holes", False)),
        bit_depth=call.request.bit_depth,
        face_boxes=face_boxes(call.request.faces),
    )
    fill = fill_damage(
        image,
        chosen.mask,
        request,
        migan=migan_or_none(deps, request.engine, call.request.device),
        cancel_event=call.cancel_event,
        on_progress=_stage(call, "restore_repair_fill"),
    )
    details = {"detectedCoverage": chosen.detected_coverage, "userEdited": chosen.user_edited, **fill.to_metadata()}
    model = _fill_model(fill, call.request.device)
    return StepOutcome(fill.image, model=model, aux_models=chosen.detector, tiles=fill.windows, details=details)


def repair_mask(deps: RunnerDeps, image: np.ndarray, call: StepCall) -> RepairMask:
    call.progress("restore_repair_detect", 0, 1)
    painted = painted_mask(call)
    detected, detector = detected_damage(deps, image, call, needed=painted is None)
    call.progress("restore_repair_detect", 1, 1)
    # La mascara pintada reemplaza a la automatica (§3.4.2).
    mask = painted if painted is not None else detected
    if call.params.get("leave_faces", False):
        mask = mask_without_boxes(mask, face_boxes(call.request.faces))
    coverage = None if detected is None else float(detected.mean())
    # Con mascara pintada el detector no decidio ningun pixel: no se declara.
    return RepairMask(mask, coverage, painted is not None, detector if painted is None else ())


def painted_mask(call: StepCall) -> np.ndarray | None:
    user_mask = call.request.hints.user_mask
    if user_mask is None or not call.params.get("use_user_mask", True):
        return None
    return user_mask.astype(bool, copy=False)


def detected_damage(
    deps: RunnerDeps, image: np.ndarray, call: StepCall, *, needed: bool
) -> tuple[np.ndarray | None, tuple[ModelUse, ...]]:
    probability = call.request.hints.damage_probability
    if probability is None and not needed:
        return None, ()
    detector = (call.request.hints.damage_detector or scratch_detector_use(),)
    if probability is None:
        probability = scratch_detector(deps.engine)(image)
        detector = (scratch_detector_use(),)
    sensitivity = float(call.params.get("sensitivity", REPAIR_SENSITIVITY_MEDIUM))
    return damage_mask(probability, sensitivity, int(call.params.get("grow_px", 0))), detector


def scratch_detector_use() -> ModelUse:
    # El mapa de una sesion vieja sin el dato tambien salio de BOPBTL: es el unico detector de daños.
    return ModelUse(SCRATCH_MODEL_ID, DETECTOR_DEVICE, DETECTOR_PRECISION)


def mask_without_boxes(mask: np.ndarray, boxes: Sequence[tuple[float, float, float, float]]) -> np.ndarray:
    kept = mask.copy()
    height, width = mask.shape[:2]
    for x0, y0, x1, y1 in boxes:
        left, top = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        right, bottom = min(width, int(np.ceil(x1))), min(height, int(np.ceil(y1)))
        kept[top:bottom, left:right] = False
    return kept


def face_boxes(faces: Sequence[FaceSelection]) -> tuple[tuple[float, float, float, float], ...]:
    # Todas las caras detectadas, no solo las elegidas: rellenar sobre cualquiera inventa rasgos.
    return tuple(face.box for face in faces if face.box is not None)


def migan_or_none(deps: RunnerDeps, engine_name: str, device: str) -> TileInfer | None:
    if engine_name != FAST_ENGINE or not deps.engine.settings.migan_available():
        return None
    return migan_infer(deps.engine, device)


def run_deblock(deps: RunnerDeps, image: np.ndarray, call: StepCall) -> StepOutcome:
    strength = float(call.params.get("strength", DEBLOCK_DEFAULT_STRENGTH))
    result = deblock_photo(drunet_context(deps, call, "restore_deblock"), image, strength, call.request.device)
    return drunet_outcome(deps.engine, result)


def run_denoise(deps: RunnerDeps, image: np.ndarray, call: StepCall) -> StepOutcome:
    strength = float(call.params.get("strength", DENOISE_DEFAULT_STRENGTH))
    grain = float(call.params.get("keep_grain", KEEP_GRAIN_DEFAULT))
    context = drunet_context(deps, call, "restore_denoise")
    sigma = call.request.hints.noise_sigma
    result = denoise_photo(context, image, strength, sigma, call.request.device, keep_grain_amount=grain)
    return drunet_outcome(deps.engine, result)


def drunet_context(deps: RunnerDeps, call: StepCall, stage: str) -> DrunetContext:
    return DrunetContext(
        engine=deps.engine,
        calibrations=deps.calibrations,
        budget_ms=deps.budget_ms,
        clock=deps.clock,
        cancel_event=call.cancel_event,
        on_progress=_stage(call, stage),
    )


def drunet_outcome(engine: PhotoRestoreEngine, result: DrunetResult) -> StepOutcome:
    return StepOutcome(
        result.image,
        model=ModelUse(result.model_id, result.device, result.precision, result.tile),
        warnings=fp16_warnings(engine, result.model_id),
        cpu_fallbacks=() if result.cpu_fallback is None else (result.cpu_fallback,),
    )


def fp16_warnings(engine: PhotoRestoreEngine, model_id: str) -> tuple[str, ...]:
    return tuple(f"fp16Rejected:{r.model_id}:{r.reason}" for r in engine.fp16_rejections() if r.model_id == model_id)


def run_tone(image: np.ndarray, call: StepCall) -> StepOutcome:
    return StepOutcome(apply_tone_in_bands(image, tone_settings(call.params, call.request.steps)))


def tone_settings(params: Mapping[str, Any], steps: Sequence[str]) -> ToneSettings:
    flags = resolve_tone_flags(steps, bool(params.get("keep_tone", True)), bool(params.get("neutral_gray", False)))
    point = params.get("gray_point")
    return ToneSettings(
        strength=float(params.get("strength", TONE_DEFAULT_STRENGTH)),
        gray_point=None if point is None else (int(point[0]), int(point[1])),
        keep_tone=flags.keep_tone,
        neutral_gray=flags.neutral_gray,
        fix_faded=bool(params.get("fix_faded", False)),
        local_contrast=bool(params.get("local_contrast", False)),
    )


def run_faces(deps: RunnerDeps, image: np.ndarray, call: StepCall) -> StepOutcome:
    source = call.source if call.source is not None else image
    selections, aux = faces_to_restore(deps, source, call)
    context = FaceRestoreContext(
        engine=deps.engine,
        calibrations=deps.calibrations,
        budget_ms=deps.budget_ms,
        clock=deps.clock,
        cancel_event=call.cancel_event,
        on_progress=_stage(call, "restore_faces"),
    )
    model_id = str(call.params.get("model", FACE_MODEL_ID))
    targets = face_targets(selections, call.params)
    result = restore_faces(
        context, source, image, targets, tone_kind=call.request.tone_kind, device=call.request.device, model_id=model_id
    )
    model = None if result.model_id is None else ModelUse(result.model_id, result.device, result.precision)
    return StepOutcome(
        result.image,
        model=model,
        aux_models=aux,
        warnings=() if model is None else fp16_warnings(deps.engine, model_id),
        cpu_fallbacks=() if result.cpu_fallback is None else (result.cpu_fallback,),
        details={"restored": [face.index for face in result.faces]},
        faces=result,
    )


def faces_to_restore(
    deps: RunnerDeps, source: np.ndarray, call: StepCall
) -> tuple[tuple[FaceSelection, ...], tuple[ModelUse, ...]]:
    # Si el usuario confirmo caras en el analisis se usan esas; si no, se detecta con la politica de §2.4.
    if call.request.faces:
        declared = call.request.hints.face_detector
        return call.request.faces, () if declared is None else (declared,)
    detections = landmarked_face_detector(deps.engine)(source)
    blend = float(call.params.get("blend", PORTRAIT_BLEND))
    return detected_selections(source, detections, blend), (face_detector_use(),)


def face_detector_use() -> ModelUse:
    return ModelUse(RETINAFACE_MODEL_ID, DETECT_DEVICE, DETECT_PRECISION)


def detected_selections(
    source: np.ndarray, detections: Sequence[FaceDetection], blend: float
) -> tuple[FaceSelection, ...]:
    by_size = sorted(detections, key=lambda face: face.eye_px, reverse=True)
    return tuple(_detected_selection(source, index, face, blend) for index, face in enumerate(by_size))


def _detected_selection(source: np.ndarray, index: int, face: FaceDetection, blend: float) -> FaceSelection:
    sharpness = face_sharpness(align_face(source, align_matrix(np.asarray(face.landmarks))))
    policy = default_face_policy(face.eye_px, sharpness)
    return FaceSelection(
        index=index,
        landmarks=face.landmarks,
        blend=_default_blend(policy.enabled, policy.blend, blend),
        enabled=policy.enabled,
        box=face.box,
        score=face.score,
        eye_px=face.eye_px,
        sharpness=sharpness,
    )


def _default_blend(enabled: bool, policy_blend: float | None, requested: float) -> float:
    # La mezcla pedida vale para las caras que la politica enciende; las apagadas guardan la suya.
    return requested if enabled or policy_blend is None else policy_blend


def face_targets(selections: Sequence[FaceSelection], params: Mapping[str, Any]) -> tuple[FaceTarget, ...]:
    selected = params.get("selected")
    chosen = None if selected is None else {int(index) for index in selected}
    per_face = {int(index): float(value) for index, value in dict(params.get("per_face") or {}).items()}
    return tuple(
        FaceTarget(
            index=face.index,
            landmarks=face.landmarks,
            blend=per_face.get(face.index, face.blend),
            enabled=face.enabled if chosen is None else face.index in chosen,
        )
        for face in selections
    )


def run_colorize(deps: RunnerDeps, image: np.ndarray, call: StepCall) -> StepOutcome:
    options = ColorizeOptions(
        strength=float(call.params.get("strength", 1.0)),
        saturation=float(call.params.get("saturation", 1.0)),
        from_luminance=bool(call.params.get("from_luminance", False)),
    )
    context = ColorizeContext(deps.engine, deps.calibrations, deps.budget_ms, deps.clock, call.cancel_event)
    model_id = str(call.params.get("model", COLORIZE_MODEL_ID))
    call.progress("restore_colorize", 0, 1)
    result = colorize(
        context, image, tone_kind=call.request.tone_kind, device=call.request.device, options=options, model_id=model_id
    )
    return StepOutcome(
        result.image,
        model=ModelUse(result.model_id, result.device, result.precision),
        warnings=fp16_warnings(deps.engine, model_id),
        cpu_fallbacks=() if result.cpu_fallback is None else (result.cpu_fallback,),
        details={
            "model": result.model_id,
            "strength": options.strength,
            "saturation": options.saturation,
            "fromLuminance": options.from_luminance,
        },
        ab_512=result.ab_512,
    )


def _fill_model(fill: FillResult, device: str) -> ModelUse | None:
    return ModelUse(MIGAN_MODEL_ID, device, MIGAN_PRECISION) if fill.engine == FAST_ENGINE else None


def _stage(call: StepCall, stage: str) -> TileProgress:
    return lambda done, total: call.progress(stage, done, total)
