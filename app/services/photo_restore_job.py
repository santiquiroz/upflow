from __future__ import annotations

import asyncio
import contextlib
import platform
import shutil
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Protocol, TypeVar

import cv2
import numpy as np
from PIL import Image

from app.config import Settings
from app.core.version import get_app_version
from app.models import UpscaleJob
from app.services.engines.photo_restore_engine import PhotoRestoreEngine
from app.services.engines.scratch_fill import CLASSIC_ENGINE, FAST_ENGINE
from app.services.engines.tiled_restore_runner import CalibrationCache, RestoreCancelled
from app.services.image_io import LoadedImage, load_image_for_restore
from app.services.photo_diagnosis import analyze_pattern, classify_tone, estimate_noise_sigma
from app.services.photo_geometry import Geometry
from app.services.photo_restore_chain import RestoreStepSpec, steps_from_selection
from app.services.photo_restore_pipeline import (
    FaceSelection,
    PhotoRestorePipeline,
    PixelLimits,
    PostResult,
    PreResult,
    RestoreHints,
    RestoreRequest,
    StepRunner,
    check_upscale,
    restore_metadata,
)
from app.services.photo_restore_presets import ToneKind
from app.services.photo_restore_runners import RunnerDeps, build_step_runners
from app.services.photo_restorer_registry import validate_step_ready
from app.services.progress import SAVING_STAGE, apply_image_tile_progress, enter_image_stage
from app.services.restore_outputs import (
    OutputContext,
    RestoreOutputPaths,
    discard_restore_outputs,
    restore_output_paths,
    save_full_outputs,
    save_preview_output,
)
from app.services.restore_provenance import ModelCatalog, UpscaleInfo, default_model_catalog
from app.services.restore_session import SessionInputs, SessionNotFound
from app.services.xmp_packet import normalize_photo_date

T = TypeVar("T")

UPSCALE_NONE = "none"
UPSCALE_CLASSIC = "classic"
UPSCALE_AI = "ai"
SR_INPUT_NAME = "pre.png"
SR_OUTPUT_NAME = "sr.png"

StepReadiness = Callable[..., None]
SessionCheck = Callable[[str], Mapping[str, Any]]
Analyzer = Callable[[np.ndarray, Sequence[str]], "RestoreAnalysis"]
ImageLoader = Callable[[Path], LoadedImage]


@dataclass(frozen=True, slots=True)
class RestoreSelection:
    steps: tuple[str, ...]
    options: Mapping[str, Any]
    upscale_mode: str
    scale: int
    session: str | None = None


@dataclass(frozen=True, slots=True)
class RestoreAnalysis:
    tone_kind: ToneKind
    hints: RestoreHints
    faces: tuple[FaceSelection, ...] = ()


class SessionSource(Protocol):
    def geometry_of(self, token: str) -> Mapping[str, Any]: ...

    def job_inputs(self, token: str) -> SessionInputs: ...


@dataclass(frozen=True, slots=True)
class PreStage:
    loaded: LoadedImage
    pre: PreResult
    pipeline: PhotoRestorePipeline


def step_uses_model(spec: RestoreStepSpec, options: Mapping[str, Any]) -> bool:
    if spec.strategy != "model_or_dsp":
        return spec.strategy == "model"
    engine = (options.get(spec.id) or {}).get("engine", FAST_ENGINE)
    return engine != CLASSIC_ENGINE


def restore_uses_model(steps: Sequence[str], options: Mapping[str, Any]) -> bool:
    return any(step_uses_model(spec, options) for spec in steps_from_selection(steps))


def restore_upscale_mode(options: Mapping[str, Any], scale: int) -> str:
    mode = options.get("upscale_mode") or (UPSCALE_NONE if scale == 1 else UPSCALE_AI)
    check_upscale(float(scale), str(mode))
    return str(mode)


def validate_restore_selection(
    settings: Settings,
    steps: Sequence[str],
    options: Mapping[str, Any],
    scale: int,
    *,
    session: str | None = None,
    check_ready: StepReadiness = validate_step_ready,
    check_session: SessionCheck | None = None,
) -> RestoreSelection:
    specs = steps_from_selection(list(steps))
    if not specs:
        raise ValueError("A restore job needs at least one restore step")
    for spec in specs:
        check_ready(settings, spec.id, uses_model=step_uses_model(spec, options))
    _validate_photo_date(options.get("photo_date"))
    mode = restore_upscale_mode(options, scale)
    steps_in_order = tuple(spec.id for spec in specs)
    with_geometry = options_with_session_geometry(options, session, check_session)
    return RestoreSelection(steps_in_order, with_geometry, mode, scale, session)


def options_with_session_geometry(
    options: Mapping[str, Any], session: str | None, check_session: SessionCheck | None
) -> dict[str, Any]:
    # La mascara y las caras de la sesion se midieron sobre su copia de trabajo: la geometria
    # del job es la de la sesion, nunca otra que mande el cliente.
    if session is None:
        Geometry.from_mapping(options.get("geometry"))
        return dict(options)
    if check_session is None:
        raise ValueError("Restore analysis sessions are not available on this server")
    return {**options, "geometry": dict(check_session(session))}


def _validate_photo_date(value: Any) -> None:
    if value is not None:
        normalize_photo_date(str(value))


def analyze_for_restore(rgb: np.ndarray, steps: Sequence[str]) -> RestoreAnalysis:
    pattern = analyze_pattern(rgb) if "descreen" in steps else None
    hints = RestoreHints(
        noise_sigma=estimate_noise_sigma(rgb) if "denoise" in steps else 0.0,
        pattern_kind=None if pattern is None else pattern.kind,
        pattern_period=None if pattern is None else pattern.period,
        peaks=None if pattern is None else pattern.peaks,
    )
    return RestoreAnalysis(tone_kind=classify_tone(rgb).kind, hints=hints)


def with_session_hints(
    analysis: RestoreAnalysis, inputs: SessionInputs | None, shape: tuple[int, int]
) -> RestoreAnalysis:
    if inputs is None:
        return analysis
    for layer in (inputs.damage_probability, inputs.user_mask):
        if layer is not None and layer.shape[:2] != shape:
            raise ValueError("The analysis session no longer matches the photo; analyze it again")
    hints = replace(analysis.hints, damage_probability=inputs.damage_probability, user_mask=inputs.user_mask)
    return replace(analysis, hints=hints, faces=inputs.faces)


def with_geometry(loaded: LoadedImage, options: Mapping[str, Any]) -> LoadedImage:
    geometry = Geometry.from_mapping(options.get("geometry"))
    return loaded if geometry == Geometry() else replace(loaded, rgb=geometry.apply(loaded.rgb))


def request_from_job(job: UpscaleJob, loaded: LoadedImage, analysis: RestoreAnalysis) -> RestoreRequest:
    options = job.restore_options
    crop = options.get("preview_crop")
    return RestoreRequest(
        image=loaded.rgb,
        steps=tuple(job.restore_steps),
        tone_kind=analysis.tone_kind,
        device=job.device or "cpu",
        params={step: dict(options.get(step) or {}) for step in job.restore_steps},
        bit_depth=loaded.bit_depth,
        scale=float(job.scale),
        upscale_mode=restore_upscale_mode(options, job.scale),
        faces=analysis.faces,
        hints=analysis.hints,
        preview_crop=None if crop is None else tuple(int(value) for value in crop),
    )


def report_restore_progress(job: UpscaleJob, stage_key: str, done: int, total: int) -> None:
    if job.metadata.get("stage") != stage_key:
        enter_image_stage(job, stage_key)
    if total > 1:
        apply_image_tile_progress(job, done, total, stage_key)


def restore_work_dir(settings: Settings, job_id: str) -> Path:
    return settings.video_work_path / job_id


def remove_work_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def write_sr_input(image: np.ndarray, path: Path) -> Path:
    # El motor SR trabaja en 8 bits: el intermedio es PNG sin perdida, nunca el formato final.
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.round(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path, format="PNG")
    return path


def load_sr_output(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    return pixels / np.float32(255.0)


def move_into(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(target))
    return target


def classic_upscale(image: np.ndarray, scale: float, cancel_event: threading.Event | None = None) -> np.ndarray:
    if cancel_event is not None and cancel_event.is_set():
        raise RestoreCancelled("Restoration cancelled")
    height, width = image.shape[:2]
    size = (round(width * scale), round(height * scale))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_LANCZOS4)
    np.clip(resized, 0.0, 1.0, out=resized)
    return resized.astype(np.float32, copy=False)


async def run_shielded(start: Callable[[threading.Event], Awaitable[T]]) -> T:
    # Como run_cancellable pero para una corrutina que corre en un hilo ajeno (el SR ONNX): al
    # cancelar se avisa por el evento y se espera, asi el permiso se suelta con la GPU ya libre.
    cancel_event = threading.Event()
    worker = asyncio.ensure_future(start(cancel_event))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        cancel_event.set()
        with contextlib.suppress(BaseException):
            await worker
        raise


class PhotoRestoreJobRunner:
    def __init__(
        self,
        settings: Settings,
        engine: PhotoRestoreEngine,
        *,
        step_runners: Mapping[str, StepRunner] | None = None,
        analyze: Analyzer = analyze_for_restore,
        load_image: ImageLoader = load_image_for_restore,
        catalog: ModelCatalog | None = None,
        app_version: str | None = None,
        check_ready: StepReadiness = validate_step_ready,
        sessions: SessionSource | None = None,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.check_ready = check_ready
        self.sessions = sessions
        self._step_runners = step_runners if step_runners is not None else _default_runners(settings, engine)
        self._analyze = analyze
        self._load_image = load_image
        self._catalog = catalog
        self._app_version = app_version

    def release_before_ncnn(self, device: str) -> bool:
        return self.engine.release_before_ncnn(device)

    def check_session(self, token: str) -> Mapping[str, Any]:
        if self.sessions is None:
            raise ValueError("Restore analysis sessions are not available on this server")
        try:
            return self.sessions.geometry_of(token)
        except SessionNotFound as exc:
            raise ValueError("Unknown restore session; analyze the photo again") from exc

    def run_pre(self, job: UpscaleJob, cancel_event: threading.Event | None = None) -> PreStage:
        inputs = self._session_inputs(job)
        loaded = with_geometry(self._load_image(job.source_path), job.restore_options)
        analysis = with_session_hints(self._analyze(loaded.rgb, job.restore_steps), inputs, loaded.rgb.shape[:2])
        pipeline = self._pipeline(job)
        self.engine.begin_phase(_device_of(job))
        pre = pipeline.run_pre(request_from_job(job, loaded, analysis), cancel_event)
        return PreStage(loaded, pre, pipeline)

    def finish(
        self,
        job: UpscaleJob,
        stage: PreStage,
        upscaled: np.ndarray | None,
        upscale: UpscaleInfo,
        cancel_event: threading.Event | None = None,
    ) -> Path:
        paths = restore_output_paths(self.settings.outputs_path, job.id, job.output_format)
        paths.final.parent.mkdir(parents=True, exist_ok=True)
        try:
            return self._post_and_save(job, stage, upscaled, upscale, paths, cancel_event)
        except BaseException:
            discard_restore_outputs(paths)
            raise

    def _post_and_save(
        self,
        job: UpscaleJob,
        stage: PreStage,
        upscaled: np.ndarray | None,
        upscale: UpscaleInfo,
        paths: RestoreOutputPaths,
        cancel_event: threading.Event | None,
    ) -> Path:
        self.engine.begin_phase(_device_of(job))
        post = stage.pipeline.run_post(stage.pre, upscaled, cancel_event, paths.artifact_dir)
        _raise_if_cancelled(cancel_event)
        enter_image_stage(job, SAVING_STAGE)
        if stage.pre.region is not None:
            job.metadata["restore"] = _preview_summary(stage.pre, post)
            return save_preview_output(post, paths, stage.loaded.icc)
        job.metadata["restore"] = save_full_outputs(stage.loaded, stage.pre, post, paths, self._context(job, upscale))
        return paths.final

    def _session_inputs(self, job: UpscaleJob) -> SessionInputs | None:
        if job.restore_session is None:
            return None
        if self.sessions is None:
            raise RuntimeError("Restore analysis sessions are not available on this server")
        return self.sessions.job_inputs(job.restore_session)

    def _pipeline(self, job: UpscaleJob) -> PhotoRestorePipeline:
        return PhotoRestorePipeline(
            self._step_runners,
            limits=PixelLimits.from_settings(self.settings),
            on_progress=partial(report_restore_progress, job),
        )

    def _context(self, job: UpscaleJob, upscale: UpscaleInfo) -> OutputContext:
        return OutputContext(
            original_name=job.original_filename,
            source_path=job.source_path,
            fmt=job.output_format,
            upscale=upscale,
            app_version=self._app_version or get_app_version(self.settings.update_package_name),
            catalog=self._catalog or default_model_catalog(),
            options=job.restore_options,
            environment={"os": platform.platform(), "device": _device_of(job)},
        )


def _preview_summary(pre: PreResult, post: PostResult) -> dict[str, Any]:
    return {**restore_metadata(pre, post), "artifacts": ["preview"]}


def _default_runners(settings: Settings, engine: PhotoRestoreEngine) -> dict[str, StepRunner]:
    deps = RunnerDeps(engine=engine, calibrations=CalibrationCache(), budget_ms=float(settings.restore_call_budget_ms))
    return build_step_runners(deps)


def _device_of(job: UpscaleJob) -> str:
    return job.device or "cpu"


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RestoreCancelled("Restoration cancelled")
