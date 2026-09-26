from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError
from starlette.datastructures import UploadFile as ReceivedFile

from app.api.auth_deps import current_user_from_request, require
from app.api.routes import (
    _can_cancel_job,
    _can_view_job,
    _parse_chain_steps,
    get_devices_service,
    get_job_manager,
    get_storage,
    job_to_response,
    resolve_request_device,
    sanitize_filename,
)
from app.config import Settings, get_settings
from app.exceptions import QueueFullError, QuotaExceededError
from app.models import JobStatus, UpscaleJob
from app.schemas import JobResponse
from app.schemas_restore import (
    RecomposeRequest,
    RecomposeResponse,
    RestoreAnalysisResponse,
    RestoreCapabilitiesResponse,
    RestoreCaptureResponse,
    RestoreDamageResponse,
    RestoreDiagnosisResponse,
    RestoreEtaResponse,
    RestoreFaceResponse,
    RestoreFindingResponse,
    RestoreGeometry,
    RestoreMaskResponse,
    RestoreOptions,
    RestorePresetResponse,
    RestorePresetSelectionResponse,
    RestoreStepCapabilityResponse,
    RestoreStepProposalResponse,
    StepEtaResponse,
)
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.permissions import Permission
from app.services.devices_service import DevicesService
from app.services.job_artifacts import UnknownArtifact, restore_artifact
from app.services.job_manager import JobManager
from app.services.photo_capture import CaptureSuggestions
from app.services.photo_diagnosis import Finding, PhotoDiagnosis
from app.services.photo_geometry import Geometry
from app.services.photo_restore_chain import HALFTONE_DENOISE_LIMIT, RESTORE_CHAIN
from app.services.photo_restore_pipeline import FaceSelection
from app.services.photo_restore_presets import PHOTO_PRESETS, PhotoFacts, resolve_preset
from app.services.release_gates import FeatureDisabledError, ensure_restore_enabled
from app.services.restore_models import PACK_PREFIX
from app.services.restore_recompose import FaceChoice, RecomposeUnavailable
from app.services.restore_recompose_job import (
    RecomposeOutcome,
    RecomposeTarget,
    recompose_outputs,
    recomposed_summary,
)
from app.services.restore_session import (
    DAMAGE_PROB_NAME,
    PREVIEW_NAME,
    InvalidMask,
    RestoreSessionStore,
    SessionAnalysis,
    SessionNotFound,
    SessionRecord,
)
from app.services.storage import StorageService

def require_restore_enabled(settings: Settings = Depends(get_settings)) -> None:
    try:
        ensure_restore_enabled(settings)
    except FeatureDisabledError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


router = APIRouter(prefix="/api/v1", tags=["restore"], dependencies=[Depends(require_restore_enabled)])

logger = logging.getLogger(__name__)

DEFAULT_PHOTO_NAME = "photo.png"
DEFAULT_SR_MODEL = "realesrgan-x4plus"
# Recomponer reescribe las salidas del job: dos pedidos a la vez sobre el mismo resultado
# mezclarian archivos. Es CPU y dura segundos, asi que van de a uno.
RECOMPOSE_LOCK = threading.Lock()


def get_restore_sessions(request: Request) -> RestoreSessionStore:
    return request.app.state.restore_sessions


def session_url(token: str, name: str) -> str:
    return f"/api/v1/restore/analysis/{token}/{name}"


def form_text(raw: object) -> str | None:
    # Llamadas directas a la corrutina reciben el centinela Form(...) en lo no enviado.
    return raw if isinstance(raw, str) and raw.strip() else None


def parse_restore_options(raw: object) -> dict[str, Any]:
    text = form_text(raw)
    if text is None:
        return {}
    try:
        parsed = RestoreOptions.model_validate_json(text)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=_validation_detail(exc)) from exc
    return parsed.model_dump(exclude_none=True)


def _validation_detail(exc: ValidationError) -> list[dict[str, Any]]:
    return [{"loc": list(error["loc"]), "msg": error["msg"]} for error in exc.errors()]


def analysis_response(analysis: SessionAnalysis) -> RestoreAnalysisResponse:
    record, diagnosis = analysis.record, analysis.diagnosis
    return RestoreAnalysisResponse(
        token=record.token,
        original_name=record.original_name,
        sha256=record.sha256,
        width=record.width,
        height=record.height,
        bit_depth=record.bit_depth,
        has_icc=record.has_icc,
        geometry=geometry_response(record.geometry),
        preview_url=session_url(record.token, PREVIEW_NAME),
        diagnosis=RestoreDiagnosisResponse(
            findings=[finding_response(finding) for finding in diagnosis.findings],
            suggested_presets=list(diagnosis.suggested_presets),
            tone_kind=diagnosis.facts.tone_kind,
        ),
        proposed_preset=diagnosis.proposed_preset,
        proposed_steps=list(diagnosis.proposed_steps),
        proposed_options=resolve_preset(diagnosis.proposed_preset, diagnosis.facts).options,
        preset_selections=preset_selections(diagnosis.facts),
        faces=[face_response(record.token, face) for face in analysis.faces],
        damage=damage_response(analysis),
        damage_over_faces=analysis.damage_over_faces,
        eta=eta_response(diagnosis),
        capture=capture_response(analysis.capture),
    )


def geometry_response(geometry: Geometry) -> RestoreGeometry:
    return RestoreGeometry(**geometry.to_dict())


def capture_response(capture: CaptureSuggestions) -> RestoreCaptureResponse:
    return RestoreCaptureResponse(
        auto_crop=None if capture.auto_crop is None else geometry_response(capture.auto_crop),
        photos=[geometry_response(geometry) for geometry in capture.photos],
        perspective=None if capture.perspective is None else geometry_response(capture.perspective),
        frame_width=capture.frame[1],
        frame_height=capture.frame[0],
    )


def preset_selections(facts: PhotoFacts) -> dict[str, RestorePresetSelectionResponse]:
    resolved = (resolve_preset(preset.id, facts) for preset in PHOTO_PRESETS)
    return {
        selection.preset: RestorePresetSelectionResponse(steps=list(selection.steps), options=selection.options)
        for selection in resolved
    }


def eta_response(diagnosis: PhotoDiagnosis) -> RestoreEtaResponse:
    per_step = {
        step: StepEtaResponse(gpu_seconds=estimate.gpu_seconds, cpu_seconds=estimate.cpu_seconds)
        for step, estimate in diagnosis.step_estimates.items()
    }
    total = diagnosis.estimate
    return RestoreEtaResponse(gpu_seconds=total.gpu_seconds, cpu_seconds=total.cpu_seconds, per_step=per_step)


def finding_response(finding: Finding) -> RestoreFindingResponse:
    return RestoreFindingResponse(
        key=finding.key,
        value=finding.value,
        reason_key=finding.reason_key,
        params=dict(finding.params),
        proposes=[
            RestoreStepProposalResponse(step_id=item.step_id, options=dict(item.options), enabled=item.enabled)
            for item in finding.proposes
        ],
        missing_pack=finding.missing_pack,
        message=finding.message,
    )


def face_response(token: str, face: FaceSelection) -> RestoreFaceResponse:
    return RestoreFaceResponse(
        index=face.index,
        box=None if face.box is None else [round(value, 2) for value in face.box],
        eye_px=face.eye_px,
        sharpness=face.sharpness,
        confidence=face.score,
        enabled=face.enabled,
        blend=face.blend,
        thumbnail_url=None if face.box is None else session_url(token, f"face-{face.index}.jpg"),
    )


def damage_response(analysis: SessionAnalysis) -> RestoreDamageResponse:
    damage = analysis.diagnosis.damage
    if damage is None:
        return RestoreDamageResponse()
    return RestoreDamageResponse(
        coverage=damage.coverage,
        prob_url=session_url(analysis.record.token, DAMAGE_PROB_NAME) if analysis.has_damage_map else None,
        large_holes=damage.large_areas,
    )


def step_installed(settings: Settings, pack: str | None) -> bool:
    return pack is None or bool(settings.restore_bundle_manifest(pack.removeprefix(PACK_PREFIX)))


def can_use_session(record: SessionRecord, user: AuthenticatedUser | None) -> bool:
    if user is None or record.owner_id is None:
        return True
    return record.owner_id == user.id or Permission.jobs_read_all in user.permissions


def require_session(sessions: RestoreSessionStore, token: str, request: Request | None) -> SessionRecord:
    try:
        record = sessions.record(token)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail="Unknown restore session") from exc
    if not can_use_session(record, current_user_from_request(request)):
        raise HTTPException(status_code=404, detail="Unknown restore session")
    return record


def require_visible_job(jobs: JobManager, job_id: str, request: Request | None, *, to_change: bool) -> UpscaleJob:
    job = jobs.get_job(job_id)
    if job is None or not _may_access(job, current_user_from_request(request), to_change):
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _may_access(job: UpscaleJob, user: AuthenticatedUser | None, to_change: bool) -> bool:
    if user is None:
        return True
    return _can_cancel_job(job, user) if to_change else _can_view_job(job, user)


def require_restored(job: UpscaleJob) -> None:
    if not job.restore_steps:
        raise HTTPException(status_code=404, detail="This job is not a photo restoration")
    if job.status != JobStatus.completed:
        raise HTTPException(status_code=409, detail="Job is not completed yet")


def face_choices(payload: RecomposeRequest) -> dict[int, FaceChoice]:
    return {index: FaceChoice(face.enabled, face.blend) for index, face in payload.faces.items()}


def recompose_serialized(target: RecomposeTarget, choices: Mapping[int, FaceChoice]) -> RecomposeOutcome:
    with RECOMPOSE_LOCK:
        return recompose_outputs(target, choices)


async def save_photo_upload(file: UploadFile, storage: StorageService, settings: Settings) -> tuple[Path, str]:
    original_name = Path(file.filename or DEFAULT_PHOTO_NAME).name
    destination = settings.uploads_path / f"{uuid4().hex}-{sanitize_filename(original_name, DEFAULT_PHOTO_NAME)}"
    try:
        await storage.save_upload(file, destination)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return destination, original_name


@router.get("/restore/capabilities", response_model=RestoreCapabilitiesResponse)
async def restore_capabilities(settings: Settings = Depends(get_settings)) -> RestoreCapabilitiesResponse:
    steps = [
        RestoreStepCapabilityResponse(
            id=spec.id,
            phase=spec.phase,
            strategy=spec.strategy,
            label_key=spec.label_key,
            description_key=spec.description_key,
            invents_detail=spec.invents_detail,
            warning_key=spec.warning_key,
            pack=spec.pack,
            installed=step_installed(settings, spec.pack),
        )
        for spec in RESTORE_CHAIN
    ]
    presets = [
        RestorePresetResponse(
            id=preset.id,
            label_key=preset.label_key,
            description_key=preset.description_key,
            steps=list(preset.step_ids()),
        )
        for preset in PHOTO_PRESETS
    ]
    return RestoreCapabilitiesResponse(steps=steps, presets=presets, halftone_denoise_limit=HALFTONE_DENOISE_LIMIT)


@router.post(
    "/restore/analyze",
    response_model=RestoreAnalysisResponse,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def analyze_photo(
    file: UploadFile = File(...),
    sessions: RestoreSessionStore = Depends(get_restore_sessions),
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
    request: Request = None,
) -> RestoreAnalysisResponse:
    user = current_user_from_request(request)
    try:
        upload, original_name = await save_photo_upload(file, storage, settings)
        analysis = await sessions.open(upload, original_name, owner_id=None if user is None else user.id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return analysis_response(analysis)


@router.post(
    "/restore/analysis/{token}/geometry",
    response_model=RestoreAnalysisResponse,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def set_photo_geometry(
    token: str,
    payload: RestoreGeometry,
    sessions: RestoreSessionStore = Depends(get_restore_sessions),
    request: Request = None,
) -> RestoreAnalysisResponse:
    require_session(sessions, token, request)
    try:
        geometry = Geometry(rotate90=payload.rotate90, crop=payload.crop, angle=payload.angle, corners=payload.corners)
        analysis = await sessions.set_geometry(token, geometry)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return analysis_response(analysis)


@router.get(
    "/restore/analysis/{token}/{name}",
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def get_session_file(
    token: str,
    name: str,
    sessions: RestoreSessionStore = Depends(get_restore_sessions),
    request: Request = None,
) -> FileResponse:
    require_session(sessions, token, request)
    try:
        path = sessions.file(token, name)
    except SessionNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    media_type = "image/png" if path.suffix == ".png" else "image/jpeg"
    return FileResponse(path=path, media_type=media_type)


@router.post(
    "/restore/analysis/{token}/mask",
    response_model=RestoreMaskResponse,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def upload_damage_mask(
    token: str,
    file: UploadFile = File(...),
    sessions: RestoreSessionStore = Depends(get_restore_sessions),
    settings: Settings = Depends(get_settings),
    request: Request = None,
) -> RestoreMaskResponse:
    record = require_session(sessions, token, request)
    data = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"Upload exceeds limit of {settings.max_upload_mb} MB")
    try:
        coverage = await asyncio.to_thread(sessions.save_mask, token, data)
    except InvalidMask as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RestoreMaskResponse(coverage=coverage, width=record.width, height=record.height)


@router.post(
    "/restore/jobs",
    response_model=JobResponse,
    status_code=202,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def create_restore_job(
    file: UploadFile | None = File(default=None),
    token: str | None = Form(default=None),
    restore_steps: str | None = Form(default=None),
    restore_options: str | None = Form(default=None),
    scale: int = Form(default=1),
    model_name: str = Form(default=DEFAULT_SR_MODEL),
    model_id: str | None = Form(default=None),
    device: str | None = Form(default=None),
    output_format: str = Form(default="png"),
    jobs: JobManager = Depends(get_job_manager),
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
    devices: DevicesService = Depends(get_devices_service),
    sessions: RestoreSessionStore = Depends(get_restore_sessions),
    request: Request = None,
) -> JobResponse:
    steps = _parse_chain_steps(restore_steps)
    if not steps:
        raise HTTPException(status_code=400, detail="Choose at least one restore step")
    options = parse_restore_options(restore_options)
    session = form_text(token)
    # FastAPI entrega el UploadFile de Starlette; en llamadas directas llega el centinela File(...).
    upload = file if isinstance(file, ReceivedFile) else None
    if (session is None) == (upload is None):
        raise HTTPException(status_code=400, detail="Send either a photo or an analysis token")
    resolved_device = await resolve_request_device(form_text(device), devices, settings)
    source, original_name, owned = await _job_source(session, upload, sessions, storage, settings, request)
    job: UpscaleJob | None = None
    try:
        job = await jobs.create_job(
            source_path=source,
            original_filename=original_name,
            model_name=model_name,
            model_id=form_text(model_id),
            device=resolved_device,
            scale=scale,
            output_format=output_format,
            job_id=uuid4().hex,
            owner=current_user_from_request(request),
            restore_steps=steps,
            restore_options=options,
            restore_session=session,
        )
    except (QueueFullError, QuotaExceededError) as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while creating a photo restoration job")
        raise HTTPException(status_code=500, detail="Failed to start the photo restoration") from exc
    finally:
        if job is None and owned:
            source.unlink(missing_ok=True)
    return job_to_response(job)


async def _job_source(
    session: str | None,
    upload: UploadFile | None,
    sessions: RestoreSessionStore,
    storage: StorageService,
    settings: Settings,
    request: Request | None,
) -> tuple[Path, str, bool]:
    if session is not None:
        record = require_session(sessions, session, request)
        return sessions.original_path(session), record.original_name, False
    try:
        path, original_name = await save_photo_upload(upload, storage, settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return path, original_name, True


@router.post(
    "/restore/jobs/{job_id}/recompose",
    response_model=RecomposeResponse,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def recompose_restored_faces(
    job_id: str,
    payload: RecomposeRequest,
    jobs: JobManager = Depends(get_job_manager),
    settings: Settings = Depends(get_settings),
    request: Request = None,
) -> RecomposeResponse:
    job = require_visible_job(jobs, job_id, request, to_change=True)
    require_restored(job)
    target = RecomposeTarget(settings.outputs_path, job.id, job.output_format, job.restore_options)
    try:
        outcome = await asyncio.to_thread(recompose_serialized, target, face_choices(payload))
    except RecomposeUnavailable as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job.metadata["restore"] = recomposed_summary(job.metadata.get("restore") or {}, outcome)
    return RecomposeResponse(sidecar=outcome.sidecar)


@router.get(
    "/jobs/{job_id}/artifacts/{name}",
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def download_job_artifact(
    job_id: str,
    name: str,
    jobs: JobManager = Depends(get_job_manager),
    settings: Settings = Depends(get_settings),
    request: Request = None,
) -> FileResponse:
    job = require_visible_job(jobs, job_id, request, to_change=False)
    require_restored(job)
    try:
        artifact = restore_artifact(
            settings.outputs_path,
            job.id,
            job.output_format,
            name,
            job.original_filename,
            job.metadata.get("restore") or {},
        )
    except UnknownArtifact as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(path=artifact.path, filename=artifact.download_name, media_type=artifact.media_type)

