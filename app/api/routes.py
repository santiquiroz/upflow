from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.config import Settings, VideoProfile, get_settings
from app.exceptions import QueueFullError
from app.models import JobStatus, UpscaleJob, VideoUpscaleJob
from app.schemas import (
    CreateJobResponse,
    EngineInfoResponse,
    HealthResponse,
    JobResponse,
    SupportedModelResponse,
    VideoJobResponse,
    VideoProfileResponse,
)
from app.services.job_manager import JobManager
from app.services.storage import StorageService
from app.services.video_job_manager import VideoJobManager

router = APIRouter(prefix="/api/v1", tags=["api"])

logger = logging.getLogger(__name__)

FORBIDDEN_FILENAME_CHARS = frozenset(':<>"|?*')
WINDOWS_RESERVED_STEMS = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(10)}
    | {f"LPT{i}" for i in range(10)}
)


def _strip_forbidden_chars(name: str) -> str:
    return "".join(char for char in name if char not in FORBIDDEN_FILENAME_CHARS)


def _escape_reserved_stem(name: str) -> str:
    stem = Path(name).stem.upper()
    if stem in WINDOWS_RESERVED_STEMS:
        return f"_{name}"
    return name


def sanitize_filename(filename: str | None, default: str) -> str:
    """Produces a filesystem-safe name for the on-disk upload path.

    Strips characters invalid on Windows (`: < > " | ? *`) and escapes
    reserved device stems (CON, NUL, COM1...) that would otherwise collide
    with OS device names regardless of extension.
    """
    candidate = Path(filename or default).name
    stripped = _strip_forbidden_chars(candidate).strip()
    if not stripped:
        stripped = default
    return _escape_reserved_stem(stripped)


class ResolvedVideoJobFields(NamedTuple):
    model_name: str
    scale: int
    output_container: str
    video_codec: str
    video_preset: str
    crf: int
    keep_audio: bool
    fps_multiplier: int
    target_fps: str | None
    audio_enhance: str | None


def resolve_video_job_fields(
    profile: VideoProfile,
    model_name: str | None,
    scale: int | None,
    output_container: str | None,
    video_codec: str | None,
    video_preset: str | None,
    crf: int | None,
    keep_audio: bool | None,
    fps_multiplier: int | None = None,
    target_fps: str | None = None,
    audio_enhance: str | None = None,
) -> ResolvedVideoJobFields:
    """Resolves per-request overrides against the profile default.

    Uses `is not None` (not `or`) for numeric fields so an explicit 0 from the
    caller is preserved instead of being silently replaced by the profile default.
    target_fps and audio_enhance have no profile default (explicit per-job
    only) — passed through as-is.
    """
    return ResolvedVideoJobFields(
        model_name=model_name or profile["model_key"],
        scale=scale if scale is not None else profile["scale"],
        output_container=output_container or "mp4",
        video_codec=video_codec or profile["video_codec"],
        video_preset=video_preset or profile["video_preset"],
        crf=crf if crf is not None else profile["crf"],
        keep_audio=keep_audio if keep_audio is not None else profile["keep_audio"],
        fps_multiplier=fps_multiplier if fps_multiplier is not None else profile.get("fps_multiplier", 1),
        target_fps=target_fps,
        audio_enhance=audio_enhance,
    )


def get_job_manager(request: Request) -> JobManager:
    return request.app.state.job_manager


def get_video_job_manager(request: Request) -> VideoJobManager:
    return request.app.state.video_job_manager


def get_storage(request: Request) -> StorageService:
    return request.app.state.storage


def job_to_response(job: UpscaleJob) -> JobResponse:
    download_url = f"/api/v1/jobs/{job.id}/download" if job.status == JobStatus.completed else None
    return JobResponse(
        job_id=job.id,
        status=job.status,
        original_filename=job.original_filename,
        model_name=job.model_name,
        scale=job.scale,
        output_format=job.output_format,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        download_url=download_url,
    )


def video_job_to_response(job: VideoUpscaleJob) -> VideoJobResponse:
    download_url = f"/api/v1/video/jobs/{job.id}/download" if job.status == JobStatus.completed else None
    return VideoJobResponse(
        job_id=job.id,
        status=job.status,
        original_filename=job.original_filename,
        model_name=job.model_name,
        scale=job.scale,
        output_container=job.output_container,
        video_codec=job.video_codec,
        video_preset=job.video_preset,
        crf=job.crf,
        keep_audio=job.keep_audio,
        fps_multiplier=job.fps_multiplier,
        target_fps=job.target_fps,
        audio_enhance=job.audio_enhance,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        metadata=job.metadata,
        download_url=download_url,
    )


@router.get("/health", response_model=HealthResponse)
async def health(
    request: Request,
    settings: Settings = Depends(get_settings),
    jobs: JobManager = Depends(get_job_manager),
    video_jobs: VideoJobManager = Depends(get_video_job_manager),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        engine=settings.engine,
        gpu_concurrency=settings.gpu_concurrency,
        queue_depth=jobs.queue_depth(),
        video_queue_depth=video_jobs.queue_depth(),
    )


@router.get("/engine", response_model=EngineInfoResponse)
async def engine_info(request: Request, settings: Settings = Depends(get_settings)) -> EngineInfoResponse:
    engine = request.app.state.engine
    media_tools = request.app.state.media_tools
    return EngineInfoResponse(
        engine=settings.engine,
        configured_binary=settings.engine_binary,
        configured_models_dir=settings.engine_models_dir,
        available=engine.available(),
        default_model=settings.default_model,
        allowed_scales=settings.allowed_scale_values,
        supported_models=[SupportedModelResponse(**item) for item in settings.model_catalog],
        video_profiles=[VideoProfileResponse(**item) for item in settings.video_profile_catalog],
        ffmpeg_available=media_tools.available(),
    )


@router.post("/jobs", response_model=CreateJobResponse, status_code=202)
async def create_job(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form(default="realesrgan-x4plus"),
    scale: int = Form(default=4),
    output_format: str = Form(default="png"),
    jobs: JobManager = Depends(get_job_manager),
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> CreateJobResponse:
    original_name = Path(file.filename or "upload.png").name
    safe_name = sanitize_filename(original_name, default="upload.png")
    token = uuid4().hex
    destination = settings.uploads_path / f"{token}-{safe_name}"

    job: UpscaleJob | None = None
    try:
        await storage.save_upload(file, destination)
        job = await jobs.create_job(
            source_path=destination,
            original_filename=original_name,
            model_name=model_name,
            scale=scale,
            output_format=output_format,
            job_id=token,
        )
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while creating image job")
        raise HTTPException(status_code=500, detail="Failed to process the uploaded image") from exc
    finally:
        if job is None and destination.exists():
            destination.unlink(missing_ok=True)

    return CreateJobResponse(
        job_id=job.id,
        status=job.status,
        status_url=f"/api/v1/jobs/{job.id}",
        download_url=None,
    )


@router.post("/video/jobs", response_model=CreateJobResponse, status_code=202)
async def create_video_job(
    request: Request,
    file: UploadFile = File(...),
    profile_key: str = Form(default="anime-balanced-2x"),
    model_name: str | None = Form(default=None),
    scale: int | None = Form(default=None),
    output_container: str | None = Form(default=None),
    video_codec: str | None = Form(default=None),
    video_preset: str | None = Form(default=None),
    crf: int | None = Form(default=None),
    keep_audio: bool | None = Form(default=None),
    fps_multiplier: int | None = Form(default=None),
    target_fps: str | None = Form(default=None),
    audio_enhance: str | None = Form(default=None),
    video_jobs: VideoJobManager = Depends(get_video_job_manager),
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> CreateJobResponse:
    profile = settings.get_video_profile(profile_key)
    if not profile:
        raise HTTPException(status_code=400, detail=f"Unknown profile: {profile_key}")

    original_name = Path(file.filename or "upload.mp4").name
    safe_name = sanitize_filename(original_name, default="upload.mp4")
    token = uuid4().hex
    destination = settings.uploads_path / f"{token}-{safe_name}"

    resolved = resolve_video_job_fields(
        profile,
        model_name,
        scale,
        output_container,
        video_codec,
        video_preset,
        crf,
        keep_audio,
        fps_multiplier,
        target_fps,
        audio_enhance,
    )

    job: VideoUpscaleJob | None = None
    try:
        await storage.save_upload(file, destination, max_mb=settings.max_video_upload_mb)
        job = await video_jobs.create_job(
            source_path=destination,
            original_filename=original_name,
            model_name=resolved.model_name,
            scale=resolved.scale,
            output_container=resolved.output_container,
            video_codec=resolved.video_codec,
            video_preset=resolved.video_preset,
            crf=resolved.crf,
            keep_audio=resolved.keep_audio,
            fps_multiplier=resolved.fps_multiplier,
            target_fps=resolved.target_fps,
            audio_enhance=resolved.audio_enhance,
            job_id=token,
        )
        job.metadata["profileKey"] = profile_key
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while creating video job")
        raise HTTPException(status_code=500, detail="Failed to process the uploaded video") from exc
    finally:
        if job is None and destination.exists():
            destination.unlink(missing_ok=True)

    return CreateJobResponse(
        job_id=job.id,
        status=job.status,
        status_url=f"/api/v1/video/jobs/{job.id}",
        download_url=None,
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, jobs: JobManager = Depends(get_job_manager)) -> JobResponse:
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_to_response(job)


@router.get("/video/jobs/{job_id}", response_model=VideoJobResponse)
async def get_video_job(job_id: str, video_jobs: VideoJobManager = Depends(get_video_job_manager)) -> VideoJobResponse:
    job = video_jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video job not found")
    return video_job_to_response(job)


@router.get("/jobs/{job_id}/download")
async def download_job(job_id: str, jobs: JobManager = Depends(get_job_manager)) -> FileResponse:
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.completed or not job.output_path:
        raise HTTPException(status_code=409, detail="Job is not completed yet")
    return FileResponse(path=job.output_path, filename=job.output_path.name, media_type="application/octet-stream")


@router.get("/video/jobs/{job_id}/download")
async def download_video_job(job_id: str, video_jobs: VideoJobManager = Depends(get_video_job_manager)) -> FileResponse:
    job = video_jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video job not found")
    if job.status != JobStatus.completed or not job.output_path:
        raise HTTPException(status_code=409, detail="Video job is not completed yet")
    return FileResponse(path=job.output_path, filename=job.output_path.name, media_type="application/octet-stream")
