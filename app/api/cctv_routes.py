"""Rutas del modo CCTV (spec §5.5). La validacion vive en los servicios y en
`VideoJobManager.create_cctv_job`; aca solo se traducen sus errores con clave
(`cctv.error.*`) a respuestas HTTP con `detail = {key, reason}`."""

from __future__ import annotations

import asyncio
import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.datastructures import UploadFile as StarletteUpload

from app.api.auth_deps import current_user_from_request, require
from app.api.routes import (
    _can_view_job,
    get_devices_service,
    get_storage,
    get_video_job_manager,
    resolve_request_device,
    video_job_to_response,
)
from app.config import MODEL_CATALOG, Settings, get_settings
from app.core.version import get_app_version
from app.exceptions import QueueFullError, QuotaExceededError
from app.models import JobStatus, VideoUpscaleJob
from app.schemas import VideoJobResponse
from app.schemas_cctv import (
    ENHANCE_ONLY,
    TARGET_HEIGHT_UNSUPPORTED,
    CctvAnalysisJobResponse,
    CctvAnalysisResponse,
    CctvJobRequest,
    CctvPresetsResponse,
    CctvReproduceRequest,
    CctvReproduceResultResponse,
    CctvReproduceStartResponse,
    OsdBoxCheckResponse,
    OsdCheckRequest,
    OsdCheckResponse,
    RoiReferenceRequest,
    RoiReferenceResponse,
    VerifyFilesResponse,
    ai_upscale_choice,
    cctv_options,
    enhance_only_fields,
)
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.permissions import Permission
from app.services.cctv_analysis import (
    FFMPEG_UNAVAILABLE,
    AnalysisRequest,
    AnalysisTools,
    CctvAnalysisError,
    adopt_staged_upload,
    analysis_tools,
    checked_frame_rate,
    discard_session,
    new_session,
    run_session_analysis,
    staged_upload,
    upload_destination,
)
from app.services.cctv_ai_models import export_on_disk, stream_upscale_models
from app.services.cctv_analysis_jobs import AnalysisSnapshot, CctvAnalysisJobs
from app.services.cctv_artifacts import ArtifactNotFound, cctv_outputs, is_inline, media_type_for, resolve_artifact
from app.services.cctv_chain import CctvChainError
from app.services.cctv_ingest import MediaTools
from app.services.cctv_presets_view import presets_payload
from app.services.cctv_preview import (
    PreviewSource,
    check_osd_on_session,
    frame_window,
    load_preview_source,
    parse_steps_param,
    preview_steps,
    render_frame,
    render_processed_frame,
)
from app.services.cctv_reproduce import (
    METADATA_KEY as REPRODUCE_KEY,
    NOT_A_REPRODUCTION,
    compare_reproduction,
    expected_facts,
    preflight_warnings,
    produced_report,
    reproduce_request,
)
from app.services.cctv_session import SESSION_NOT_FOUND, cctv_job_dir
from app.services.devices_service import DevicesService
from app.services.ffmpeg_capabilities import FfmpegCapabilities, FfmpegProbeError, cached_capabilities
from app.services.frame_export import StillFrameError
from app.services.handover_package import check_files_unchanged
from app.services.osd_check import OSD_NOT_TEXT, OsdBoxCheck, osd_warnings
from app.services.roi_frames import ROI_DECODE_FAILED, RoiDecodeError
from app.services.roi_reference import (
    ReferenceRequest,
    check_reference_request,
    resolved_roi_steps,
    suggest_session_reference,
)
from app.services.roi_registration import RoiRegistrationError
from app.services.storage import StorageService
from app.services.video_analysis import VideoAnalysisError
from app.services.video_job_manager import VideoJobManager

router = APIRouter(prefix="/api/v1/video", tags=["cctv"])

logger = logging.getLogger(__name__)

PREVIEW_CONCURRENCY = 1
DEFAULT_PREVIEW_WINDOW = 4
JOB_NOT_FOUND = "cctv.error.jobNotFound"
NOT_A_CCTV_JOB = "cctv.error.notACctvJob"
JOB_NOT_COMPLETED = "cctv.error.jobNotCompleted"
ANALYSIS_NOT_FOUND = "cctv.error.analysisNotFound"
UPLOAD_CHOICE = "cctv.error.uploadChoice"
OSD_CHECK_FAILED = "cctv.error.osdCheckFailed"
NOT_FOUND_KEYS = frozenset({SESSION_NOT_FOUND})
REPORT_CSP = "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'"
NO_STORE = {"Cache-Control": "no-store"}


@dataclass
class CctvApiState:
    analyses: CctvAnalysisJobs = field(default_factory=CctvAnalysisJobs)
    preview_slots: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(PREVIEW_CONCURRENCY))


def get_cctv_state(request: Request) -> CctvApiState:
    return request.app.state.cctv


# --- Errores con clave ---


def keyed_error(status: int, key: str, reason: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"key": key, "reason": reason})


def chain_error(exc: CctvChainError) -> HTTPException:
    return keyed_error(404 if exc.code in NOT_FOUND_KEYS else 400, exc.code, str(exc))


def analysis_http_error(exc: CctvAnalysisError) -> HTTPException:
    return keyed_error(exc.status, exc.key, str(exc))


def ffmpeg_unavailable() -> HTTPException:
    return keyed_error(503, FFMPEG_UNAVAILABLE, "ffmpeg is not available for CCTV mode.")


# --- Dependencias de servicio ---


def session_media(settings: Settings) -> MediaTools:
    return MediaTools(settings.ffmpeg_binary_path, settings.ffprobe_binary_path)


def capabilities_of(settings: Settings) -> Callable[[], FfmpegCapabilities]:
    binary = settings.ffmpeg_binary_path
    return lambda: cached_capabilities(binary)


def session_analysis_tools(settings: Settings) -> AnalysisTools:
    return analysis_tools(settings.ffmpeg_binary_path, settings.ffprobe_binary_path, capabilities_of(settings))


async def current_capabilities(settings: Settings) -> FfmpegCapabilities:
    try:
        return await asyncio.to_thread(capabilities_of(settings))
    except FfmpegProbeError as exc:
        raise ffmpeg_unavailable() from exc


def owner_id_of(user: AuthenticatedUser | None) -> str | None:
    return None if user is None else user.id


# --- Analisis ---


def text_or_none(value: object) -> str | None:
    # Llamadas directas (tests) reciben el centinela Form(...) y no None.
    return value if isinstance(value, str) and value else None


def upload_or_none(file: object) -> UploadFile | None:
    # FastAPI entrega el UploadFile de Starlette, no la subclase de fastapi.
    return file if isinstance(file, StarletteUpload) and file.filename else None


def check_upload_choice(file: UploadFile | None, upload_token: str | None) -> None:
    if (file is None) == (upload_token is None):
        raise keyed_error(400, UPLOAD_CHOICE, "Provide exactly one of file or upload_token.")


async def store_upload(
    directory: Path, file: UploadFile | None, upload_token: str | None, storage: StorageService, settings: Settings
) -> Path:
    if file is None:
        staged = staged_upload(settings.uploads_path, upload_token)
        return await asyncio.to_thread(adopt_staged_upload, staged, directory, upload_token)
    destination = upload_destination(directory, file.filename)
    await storage.save_upload(file, destination, max_mb=settings.max_video_upload_mb)
    return destination


async def open_analysis(
    file: UploadFile | None,
    upload_token: str | None,
    frame_rate: str | None,
    storage: StorageService,
    settings: Settings,
) -> AnalysisRequest:
    rate = checked_frame_rate(frame_rate)
    token, directory = await asyncio.to_thread(new_session, settings.video_work_path)
    try:
        upload = await store_upload(directory, file, upload_token, storage, settings)
    except BaseException:
        await asyncio.to_thread(discard_session, directory)
        raise
    return AnalysisRequest(token, directory, upload, rate)


def analysis_status_url(analysis_id: str) -> str:
    return f"/api/v1/video/cctv/analysis/{analysis_id}"


def analysis_job_response(snapshot: AnalysisSnapshot) -> CctvAnalysisJobResponse:
    error = snapshot.error
    return CctvAnalysisJobResponse(
        analysis_job_id=snapshot.id,
        status=snapshot.status,
        status_url=analysis_status_url(snapshot.id),
        result=None if snapshot.result is None else CctvAnalysisResponse.model_validate(snapshot.result),
        error=None if error is None else str(error),
        error_key=None if error is None else error.key,
    )


def analysis_reply(snapshot: AnalysisSnapshot) -> CctvAnalysisResponse | JSONResponse:
    if snapshot.error is not None:
        raise analysis_http_error(snapshot.error)
    if snapshot.result is not None:
        return CctvAnalysisResponse.model_validate(snapshot.result)
    body = analysis_job_response(snapshot).model_dump(mode="json", by_alias=True)
    return JSONResponse(status_code=202, content=body)


@router.post(
    "/cctv/analyze",
    response_model=CctvAnalysisResponse,
    responses={202: {"model": CctvAnalysisJobResponse}},
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def analyze_cctv(
    request: Request,
    file: UploadFile | None = File(default=None),
    upload_token: str | None = Form(default=None),
    frame_rate: str | None = Form(default=None),
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
    state: CctvApiState = Depends(get_cctv_state),
) -> CctvAnalysisResponse | JSONResponse:
    upload, token = upload_or_none(file), text_or_none(upload_token)
    check_upload_choice(upload, token)
    try:
        analysis = await open_analysis(upload, token, text_or_none(frame_rate), storage, settings)
    except CctvAnalysisError as exc:
        raise analysis_http_error(exc) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    tools = session_analysis_tools(settings)
    owner = owner_id_of(current_user_from_request(request))
    snapshot = await state.analyses.submit(lambda: run_session_analysis(analysis, tools), owner)
    return analysis_reply(snapshot)


def visible_analysis(snapshot: AnalysisSnapshot | None, user: AuthenticatedUser | None) -> AnalysisSnapshot:
    if snapshot is None or not can_see_owned(snapshot.owner_id, user):
        raise keyed_error(404, ANALYSIS_NOT_FOUND, "The CCTV analysis was not found.")
    return snapshot


def can_see_owned(owner_id: str | None, user: AuthenticatedUser | None) -> bool:
    return user is None or Permission.jobs_read_all in user.permissions or owner_id == user.id


@router.get(
    "/cctv/analysis/{analysis_id}",
    response_model=CctvAnalysisJobResponse,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def get_cctv_analysis(
    analysis_id: str, request: Request, state: CctvApiState = Depends(get_cctv_state)
) -> CctvAnalysisJobResponse:
    snapshot = visible_analysis(state.analyses.get(analysis_id), current_user_from_request(request))
    return analysis_job_response(snapshot)


# --- Presets ---


@router.get("/cctv/presets", response_model=CctvPresetsResponse)
async def cctv_presets(settings: Settings = Depends(get_settings)) -> CctvPresetsResponse:
    caps = await current_capabilities(settings)
    models = await asyncio.to_thread(ai_upscale_models_payload, settings)
    return CctvPresetsResponse.model_validate({**presets_payload(caps), "aiUpscaleModels": models})


def ai_upscale_models_payload(settings: Settings) -> list[dict]:
    installed = functools.partial(export_on_disk, settings.builtin_onnx_path)
    return [model.to_json() for model in stream_upscale_models(MODEL_CATALOG, installed)]


# --- Vista previa y chequeo del OSD ---


async def preview_source(settings: Settings, token: str) -> PreviewSource:
    return await load_preview_source(settings.video_work_path, token, session_media(settings))


async def processed_preview(
    settings: Settings, state: CctvApiState, source: PreviewSource, frame: int, steps: str, window: int
) -> bytes:
    caps = await current_capabilities(settings)
    resolved = preview_steps(parse_steps_param(steps), caps, source.geometry)
    bounds = frame_window(frame, window, source.frame_count)
    async with state.preview_slots:
        return await render_processed_frame(session_media(settings), source, bounds, resolved)


@router.get(
    "/cctv/{token}/preview",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def preview_cctv_frame(
    token: str,
    frame: int = Query(ge=0),
    steps: str | None = Query(default=None),
    window: int = Query(default=DEFAULT_PREVIEW_WINDOW),
    settings: Settings = Depends(get_settings),
    state: CctvApiState = Depends(get_cctv_state),
) -> Response:
    try:
        source = await preview_source(settings, token)
        if steps is None:
            png = await render_frame(session_media(settings), source, frame)
        else:
            png = await processed_preview(settings, state, source, frame, steps, window)
    except CctvChainError as exc:
        raise chain_error(exc) from exc
    return Response(content=png, media_type="image/png", headers=NO_STORE)


def osd_check_response(checks: tuple[OsdBoxCheck, ...]) -> OsdCheckResponse:
    return OsdCheckResponse(
        checks=[
            OsdBoxCheckResponse(
                box=list(check.box),
                frames=check.frames,
                contrast=check.contrast,
                static_fraction=check.static_fraction,
                looks_like_text=check.looks_like_text,
                warning_key=None if check.looks_like_text else OSD_NOT_TEXT,
            )
            for check in checks
        ],
        warnings=list(osd_warnings(checks)),
    )


@router.post(
    "/cctv/{token}/osd-check",
    response_model=OsdCheckResponse,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def check_cctv_osd(
    token: str, body: OsdCheckRequest = Body(...), settings: Settings = Depends(get_settings)
) -> OsdCheckResponse:
    try:
        source = await preview_source(settings, token)
        boxes = tuple(tuple(box) for box in body.boxes)
        checks = await check_osd_on_session(session_media(settings), source, boxes, body.frame)
    except CctvChainError as exc:
        raise chain_error(exc) from exc
    except VideoAnalysisError as exc:
        raise keyed_error(500, OSD_CHECK_FAILED, "The on-screen text check could not decode the video.") from exc
    return osd_check_response(checks)


def reference_request_of(body: RoiReferenceRequest) -> ReferenceRequest:
    steps = tuple(step.model_dump() for step in body.steps)
    return ReferenceRequest(body.first_frame, body.last_frame, tuple(body.box), steps)


@router.post(
    "/cctv/{token}/roi/reference",
    response_model=RoiReferenceResponse,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def suggest_roi_reference(
    token: str,
    body: RoiReferenceRequest = Body(...),
    settings: Settings = Depends(get_settings),
    state: CctvApiState = Depends(get_cctv_state),
) -> RoiReferenceResponse:
    request = reference_request_of(body)
    try:
        source = await preview_source(settings, token)
        check_reference_request(request, source, settings.cctv_roi_max_frames)
        steps = resolved_roi_steps(request.steps, await current_capabilities(settings))
        async with state.preview_slots:
            frame = await suggest_session_reference(settings.ffmpeg_binary_path, source, request, steps)
    except CctvChainError as exc:
        raise chain_error(exc) from exc
    except RoiRegistrationError as exc:
        raise keyed_error(400, exc.key, str(exc)) from exc
    except (RoiDecodeError, VideoAnalysisError) as exc:
        raise keyed_error(500, ROI_DECODE_FAILED, "The frames of the range could not be decoded.") from exc
    return RoiReferenceResponse(reference_frame=frame)


# --- Jobs ---


def reject_enhance_only(body: CctvJobRequest) -> None:
    fields = enhance_only_fields(body)
    if fields and body.task != "enhance":
        raise keyed_error(400, ENHANCE_ONLY, f"{', '.join(fields)} only apply to the 'enhance' task.")
    # La banda del rotulo va despues de todo redimensionado: el ajuste a un alto va en el paso "Resize".
    if body.target_height is not None:
        raise keyed_error(
            400, TARGET_HEIGHT_UNSUPPORTED, "targetHeight is not supported in CCTV; use the Resize step instead."
        )


def job_creation_error(exc: Exception) -> HTTPException:
    if isinstance(exc, CctvChainError):
        return chain_error(exc)
    if isinstance(exc, StillFrameError):
        return keyed_error(400, exc.key, str(exc))
    if isinstance(exc, (QueueFullError, QuotaExceededError)):
        return HTTPException(status_code=429, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    logger.exception("Unexpected error while creating a CCTV job", exc_info=exc)
    return HTTPException(status_code=500, detail="Failed to create the CCTV job")


@router.post(
    "/cctv/jobs",
    response_model=VideoJobResponse,
    status_code=202,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def create_cctv_job(
    request: Request,
    body: CctvJobRequest = Body(...),
    video_jobs: VideoJobManager = Depends(get_video_job_manager),
    settings: Settings = Depends(get_settings),
    devices: DevicesService = Depends(get_devices_service),
) -> VideoJobResponse:
    reject_enhance_only(body)
    device = await resolve_request_device(body.device, devices, settings)
    try:
        job = await video_jobs.create_cctv_job(
            cctv=cctv_options(body),
            device=device,
            owner=current_user_from_request(request),
            upscale=ai_upscale_choice(body),
        )
    except Exception as exc:
        raise job_creation_error(exc) from exc
    return video_job_to_response(job)


# --- Artefactos y "Check files are unchanged" ---


def completed_cctv_job(video_jobs: VideoJobManager, job_id: str, user: AuthenticatedUser | None) -> VideoUpscaleJob:
    job = video_jobs.get_job(job_id)
    if job is None or (user is not None and not _can_view_job(job, user)):
        raise keyed_error(404, JOB_NOT_FOUND, "Video job not found.")
    if job.cctv is None:
        raise keyed_error(404, NOT_A_CCTV_JOB, "This video job has no CCTV artifacts.")
    if job.status != JobStatus.completed:
        raise keyed_error(409, JOB_NOT_COMPLETED, "The CCTV job has not finished yet.")
    return job


def artifact_headers(name: str) -> dict[str, str]:
    # El informe se abre en la pestaña: sin scripts ni recursos externos.
    return {**NO_STORE, "Content-Security-Policy": REPORT_CSP} if is_inline(name) else dict(NO_STORE)


@router.get(
    "/jobs/{job_id}/artifacts/{name}",
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def get_cctv_artifact(
    job_id: str,
    name: str,
    request: Request,
    video_jobs: VideoJobManager = Depends(get_video_job_manager),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    job = completed_cctv_job(video_jobs, job_id, current_user_from_request(request))
    job_dir = cctv_job_dir(settings.outputs_path, job.id)
    try:
        path = await asyncio.to_thread(resolve_artifact, job_dir, name, cctv_outputs(job))
    except ArtifactNotFound as exc:
        raise keyed_error(404, exc.key, str(exc)) from exc
    return FileResponse(
        path,
        media_type=media_type_for(path),
        filename=path.name,
        content_disposition_type="inline" if is_inline(name) else "attachment",
        headers=artifact_headers(name),
    )


@router.post(
    "/jobs/{job_id}/verify",
    response_model=VerifyFilesResponse,
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def verify_cctv_job(
    job_id: str,
    request: Request,
    video_jobs: VideoJobManager = Depends(get_video_job_manager),
    settings: Settings = Depends(get_settings),
) -> VerifyFilesResponse:
    job = completed_cctv_job(video_jobs, job_id, current_user_from_request(request))
    result = await asyncio.to_thread(check_files_unchanged, cctv_job_dir(settings.outputs_path, job.id))
    return VerifyFilesResponse.model_validate(result.to_json())


# --- Reproduce (P4-REPRODUCE) ---


def reproduce_start_response(job: VideoUpscaleJob, warnings: list[str]) -> CctvReproduceStartResponse:
    return CctvReproduceStartResponse(
        job_id=job.id,
        status_url=f"/api/v1/video/jobs/{job.id}",
        result_url=f"/api/v1/video/jobs/{job.id}/reproduce",
        warnings=warnings,
    )


@router.post(
    "/cctv/reproduce",
    response_model=CctvReproduceStartResponse,
    status_code=202,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def reproduce_cctv_job(
    request: Request,
    body: CctvReproduceRequest = Body(...),
    video_jobs: VideoJobManager = Depends(get_video_job_manager),
    settings: Settings = Depends(get_settings),
) -> CctvReproduceStartResponse:
    try:
        wanted = await asyncio.to_thread(reproduce_request, body, settings.video_work_path)
        job = await video_jobs.create_cctv_job(cctv=cctv_options(wanted.job), owner=current_user_from_request(request))
    except Exception as exc:
        raise job_creation_error(exc) from exc
    expected = expected_facts(wanted.report)
    job.metadata[REPRODUCE_KEY] = expected
    caps = await asyncio.to_thread(video_jobs.cctv_capabilities)
    return reproduce_start_response(job, preflight_warnings(expected, caps, get_app_version(settings.update_package_name)))


def reproduce_expectation(job: VideoUpscaleJob) -> dict:
    expected = job.metadata.get(REPRODUCE_KEY)
    if not isinstance(expected, dict):
        raise keyed_error(404, NOT_A_REPRODUCTION, "This job is not a reproduction of a report.")
    return expected


@router.get(
    "/jobs/{job_id}/reproduce",
    response_model=CctvReproduceResultResponse,
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def get_reproduce_result(
    job_id: str,
    request: Request,
    video_jobs: VideoJobManager = Depends(get_video_job_manager),
    settings: Settings = Depends(get_settings),
) -> CctvReproduceResultResponse:
    job = completed_cctv_job(video_jobs, job_id, current_user_from_request(request))
    expected = reproduce_expectation(job)
    job_dir = cctv_job_dir(settings.outputs_path, job.id)
    try:
        produced = await asyncio.to_thread(produced_report, job_dir, cctv_outputs(job))
    except ArtifactNotFound as exc:
        raise keyed_error(404, exc.key, str(exc)) from exc
    comparison = compare_reproduction(expected, produced)
    return CctvReproduceResultResponse.model_validate({"jobId": job.id, **comparison.to_json()})
