from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse

from app.config import (
    AUDIO_ENHANCE_MODES,
    AUDIO_RESTORE_MODES,
    INTERP_ENGINES,
    RIFE_ENGINE,
    Settings,
    VideoProfile,
    get_settings,
)
from app.exceptions import ModelNotFoundError, ModelProtectedError, QueueFullError
from app.models import AudioJob, JobStatus, UpdateStatus, UpscaleJob, VideoUpscaleJob
from app.schemas import (
    AnalyzeVideoResponse,
    AudioCapabilitiesResponse,
    AudioJobResponse,
    AudioTrackResponse,
    CreateInstallResponse,
    CreateJobResponse,
    DeviceInfoResponse,
    DevicesResponse,
    EngineInfoResponse,
    HealthResponse,
    HfModelSearchResultResponse,
    InstallModelRequest,
    InstallStatusResponse,
    JobResponse,
    ModelResponse,
    ModelSearchResponse,
    ModelsResponse,
    SubtitleTrackResponse,
    SupportedModelResponse,
    UpdateCheckResponse,
    VideoCapabilitiesResponse,
    VideoJobResponse,
    VideoProfileResponse,
)
from app.services.audio_job_manager import AudioJobManager
from app.services.devices_service import AUTO_DEVICE_ID, DevicesService
from app.services.hf_client import HfClient
from app.services.job_manager import JobManager
from app.services.media_tools import MediaTools
from app.services.model_installer import ModelInstaller
from app.services.model_registry import ModelEntry, ModelRegistry
from app.services.storage import StorageService
from app.services.stream_analysis import parse_audio_tracks, parse_subtitle_tracks
from app.services.update_service import UpdateService
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


def get_audio_job_manager(request: Request) -> AudioJobManager:
    return request.app.state.audio_job_manager


def get_storage(request: Request) -> StorageService:
    return request.app.state.storage


def get_media_tools(request: Request) -> MediaTools:
    return request.app.state.media_tools


def get_devices_service(request: Request) -> DevicesService:
    return request.app.state.devices_service


def get_model_registry(request: Request) -> ModelRegistry:
    return request.app.state.model_registry


def get_hf_client(request: Request) -> HfClient:
    return request.app.state.hf_client


def get_model_installer(request: Request) -> ModelInstaller:
    return request.app.state.model_installer


def get_update_service(request: Request) -> UpdateService:
    return request.app.state.update_service


async def resolve_request_device(device: str | None, devices: DevicesService, settings: Settings) -> str:
    """Resolves the `device` Form param to a concrete device id.

    An explicit device (including the "auto" sentinel) is passed through
    untouched -- "auto" is validated/resolved downstream by the job manager
    and its device_router. `None` means the caller didn't pin a device: if
    ENABLE_AUTO_ROUTE is on, that implicitly means "auto" too; otherwise it
    defaults to settings.DEFAULT_DEVICE via DevicesService.resolve_default --
    real hardware enumeration, so it is dispatched through asyncio.to_thread
    rather than blocking the event loop.
    """
    if device is not None:
        return device
    if settings.enable_auto_route:
        return AUTO_DEVICE_ID
    device_list = await asyncio.to_thread(devices.list_devices)
    return devices.resolve_default(device_list)["id"]


def _progress_pct_from_metadata(metadata: dict[str, Any]) -> float | None:
    progress = metadata.get("progress")
    return progress * 100 if isinstance(progress, (int, float)) else None


def job_to_response(job: UpscaleJob) -> JobResponse:
    download_url = f"/api/v1/jobs/{job.id}/download" if job.status == JobStatus.completed else None
    return JobResponse(
        job_id=job.id,
        status=job.status,
        original_filename=job.original_filename,
        model_name=job.model_name,
        scale=job.scale,
        output_format=job.output_format,
        model_id=job.model_id,
        device=job.device,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        metadata=job.metadata,
        progress_pct=_progress_pct_from_metadata(job.metadata),
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
        audio_restore=job.audio_restore,
        interp_engine=job.interp_engine,
        model_id=job.model_id,
        device=job.device,
        backend=job.backend,
        video_encoder=job.video_encoder,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        error=job.error,
        metadata=job.metadata,
        progress_pct=_progress_pct_from_metadata(job.metadata),
        download_url=download_url,
    )


def audio_job_to_response(job: AudioJob) -> AudioJobResponse:
    download_url = f"/api/v1/audio/jobs/{job.id}/download" if job.status == JobStatus.completed else None
    return AudioJobResponse(
        id=job.id,
        status=job.status,
        original_filename=job.original_filename,
        denoise=job.denoise,
        restore=job.restore,
        device=job.device,
        progress_pct=_progress_pct_from_metadata(job.metadata),
        stages=job.metadata.get("stages"),
        error=job.error,
        download_url=download_url,
    )


def model_entry_to_response(entry: ModelEntry) -> ModelResponse:
    return ModelResponse(
        id=entry.id,
        name=entry.name,
        kind=entry.kind.value,
        source=entry.source,
        scale=entry.scale,
        arch=entry.arch,
        size_bytes=entry.size_bytes,
        status=entry.status.value,
        error=entry.error,
    )


def update_status_to_response(status: UpdateStatus) -> UpdateCheckResponse:
    return UpdateCheckResponse(
        current_version=status.current_version,
        latest_version=status.latest_version,
        update_available=status.update_available,
        release_url=status.release_url,
        published_at=status.published_at,
        checked_at=status.checked_at,
        error=status.error,
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
        # gpu_concurrency now reports the per-device GPU concurrency (the
        # role GPU_CONCURRENCY used to play before per-device semaphores) --
        # the JSON field name is unchanged so existing API consumers keep
        # working.
        gpu_concurrency=settings.per_device_gpu_concurrency,
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


@router.get("/update-check", response_model=UpdateCheckResponse)
async def update_check(
    force: bool = Query(default=False),
    updates: UpdateService = Depends(get_update_service),
) -> UpdateCheckResponse:
    status = await updates.check(force=force)
    return update_status_to_response(status)


@router.get("/devices", response_model=DevicesResponse)
async def list_devices(devices: DevicesService = Depends(get_devices_service)) -> DevicesResponse:
    device_list = await asyncio.to_thread(devices.list_devices)
    default_device = devices.resolve_default(device_list)
    return DevicesResponse(
        devices=[DeviceInfoResponse(**item) for item in device_list],
        default_device_id=default_device["id"],
    )


@router.post("/jobs", response_model=CreateJobResponse, status_code=202)
async def create_job(
    request: Request,
    file: UploadFile = File(...),
    model_name: str = Form(default="realesrgan-x4plus"),
    model_id: str | None = Form(default=None),
    device: str | None = Form(default=None),
    scale: int = Form(default=4),
    output_format: str = Form(default="png"),
    jobs: JobManager = Depends(get_job_manager),
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
    devices: DevicesService = Depends(get_devices_service),
) -> CreateJobResponse:
    original_name = Path(file.filename or "upload.png").name
    safe_name = sanitize_filename(original_name, default="upload.png")
    token = uuid4().hex
    destination = settings.uploads_path / f"{token}-{safe_name}"
    resolved_device = await resolve_request_device(device, devices, settings)

    job: UpscaleJob | None = None
    try:
        await storage.save_upload(file, destination)
        job = await jobs.create_job(
            source_path=destination,
            original_filename=original_name,
            model_name=model_name,
            model_id=model_id,
            device=resolved_device,
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
    audio_restore: str | None = Form(default=None),
    interp_engine: str | None = Form(default=None),
    model_id: str | None = Form(default=None),
    device: str | None = Form(default=None),
    backend: str | None = Form(default=None),
    video_encoder: str | None = Form(default=None),
    video_jobs: VideoJobManager = Depends(get_video_job_manager),
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
    devices: DevicesService = Depends(get_devices_service),
) -> CreateJobResponse:
    profile = settings.get_video_profile(profile_key)
    if not profile:
        raise HTTPException(status_code=400, detail=f"Unknown profile: {profile_key}")

    # FastAPI passes `backend` as str|None; a direct unit-test call that omits
    # it receives the Form() sentinel instead, so normalize non-strings to None.
    backend_value = backend if isinstance(backend, str) else None
    video_encoder_value = video_encoder if isinstance(video_encoder, str) else "auto"
    interp_engine_value = interp_engine if isinstance(interp_engine, str) else RIFE_ENGINE

    original_name = Path(file.filename or "upload.mp4").name
    safe_name = sanitize_filename(original_name, default="upload.mp4")
    token = uuid4().hex
    destination = settings.uploads_path / f"{token}-{safe_name}"
    resolved_device = await resolve_request_device(device, devices, settings)

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
            audio_restore=audio_restore,
            interp_engine=interp_engine_value,
            model_id=model_id,
            device=resolved_device,
            backend=backend_value,
            video_encoder=video_encoder_value,
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


@router.post("/video/analyze", response_model=AnalyzeVideoResponse)
async def analyze_video(
    file: UploadFile = File(...),
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
    media_tools: MediaTools = Depends(get_media_tools),
) -> AnalyzeVideoResponse:
    original_name = Path(file.filename or "upload.mp4").name
    safe_name = sanitize_filename(original_name, default="upload.mp4")
    token = uuid4().hex
    destination = settings.uploads_path / f"{token}-{safe_name}"

    try:
        await storage.save_upload(file, destination, max_mb=settings.max_video_upload_mb)
    except ValueError as exc:
        if destination.exists():
            destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    probe: dict[str, Any] | None = None
    try:
        probe = await media_tools.ffprobe_json(destination)
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid video") from exc
    except RuntimeError as exc:
        logger.exception("ffprobe unavailable while analyzing uploaded video")
        raise HTTPException(status_code=500, detail="Video analysis is unavailable") from exc
    except Exception as exc:
        logger.exception("Unexpected error while analyzing uploaded video")
        raise HTTPException(status_code=500, detail="Failed to analyze the uploaded video") from exc
    finally:
        if probe is None and destination.exists():
            destination.unlink(missing_ok=True)

    audio_tracks = parse_audio_tracks(probe)
    subtitle_tracks = parse_subtitle_tracks(probe)
    return AnalyzeVideoResponse(
        upload_token=token,
        audio_tracks=[
            AudioTrackResponse(
                index=t.index, codec=t.codec, channels=t.channels, is_default=t.is_default, language=t.language
            )
            for t in audio_tracks
        ],
        subtitle_tracks=[
            SubtitleTrackResponse(index=t.index, codec=t.codec, language=t.language) for t in subtitle_tracks
        ],
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, jobs: JobManager = Depends(get_job_manager)) -> JobResponse:
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_to_response(job)


@router.post("/jobs/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(job_id: str, jobs: JobManager = Depends(get_job_manager)) -> JobResponse:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not jobs.cancel_job(job_id):
        raise HTTPException(status_code=409, detail="Job already finished")
    return job_to_response(job)


@router.get("/video/jobs/{job_id}", response_model=VideoJobResponse)
async def get_video_job(job_id: str, video_jobs: VideoJobManager = Depends(get_video_job_manager)) -> VideoJobResponse:
    job = video_jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video job not found")
    return video_job_to_response(job)


@router.post("/video/jobs/{job_id}/cancel", response_model=VideoJobResponse)
async def cancel_video_job(
    job_id: str, video_jobs: VideoJobManager = Depends(get_video_job_manager)
) -> VideoJobResponse:
    job = video_jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Video job not found")
    if not video_jobs.cancel_job(job_id):
        raise HTTPException(status_code=409, detail="Job already finished")
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


@router.post("/audio/jobs", response_model=CreateJobResponse, status_code=202)
async def create_audio_job(
    request: Request,
    file: UploadFile = File(...),
    denoise: str | None = Form(default=None),
    restore: str | None = Form(default=None),
    device: str | None = Form(default=None),
    audio_jobs: AudioJobManager = Depends(get_audio_job_manager),
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> CreateJobResponse:
    original_name = Path(file.filename or "upload.wav").name
    safe_name = sanitize_filename(original_name, default="upload.wav")
    token = uuid4().hex
    destination = settings.uploads_path / f"{token}-{safe_name}"

    job: AudioJob | None = None
    try:
        await storage.save_upload(file, destination, max_mb=settings.max_audio_upload_mb)
        job = await audio_jobs.create_job(
            source_path=destination,
            original_filename=original_name,
            denoise=denoise,
            restore=restore,
            device=device,
            job_id=token,
        )
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while creating audio job")
        raise HTTPException(status_code=500, detail="Failed to process the uploaded audio") from exc
    finally:
        if job is None and destination.exists():
            destination.unlink(missing_ok=True)

    return CreateJobResponse(
        job_id=job.id,
        status=job.status,
        status_url=f"/api/v1/audio/jobs/{job.id}",
        download_url=None,
    )


@router.get("/video/capabilities", response_model=VideoCapabilitiesResponse)
async def video_capabilities(settings: Settings = Depends(get_settings)) -> VideoCapabilitiesResponse:
    interp_engines = [
        engine for engine in sorted(INTERP_ENGINES) if settings.interp_engine_available(engine)
    ]
    return VideoCapabilitiesResponse(interp_engines=interp_engines)


@router.get("/audio/capabilities", response_model=AudioCapabilitiesResponse)
async def audio_capabilities(settings: Settings = Depends(get_settings)) -> AudioCapabilitiesResponse:
    denoise_modes = [mode for mode in sorted(AUDIO_ENHANCE_MODES) if settings.audio_enhance_available(mode)]
    restore_modes = [
        mode for mode in sorted(AUDIO_RESTORE_MODES) if settings.audio_restore_mode_available(mode)
    ]
    return AudioCapabilitiesResponse(
        denoise_modes=denoise_modes,
        restore_available=bool(restore_modes),
        restore_modes=restore_modes,
    )


@router.get("/audio/jobs/{job_id}", response_model=AudioJobResponse)
async def get_audio_job(job_id: str, audio_jobs: AudioJobManager = Depends(get_audio_job_manager)) -> AudioJobResponse:
    job = audio_jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Audio job not found")
    return audio_job_to_response(job)


@router.post("/audio/jobs/{job_id}/cancel", response_model=AudioJobResponse)
async def cancel_audio_job(
    job_id: str, audio_jobs: AudioJobManager = Depends(get_audio_job_manager)
) -> AudioJobResponse:
    job = audio_jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Audio job not found")
    if not audio_jobs.cancel_job(job_id):
        raise HTTPException(status_code=409, detail="Job already finished")
    return audio_job_to_response(job)


@router.get("/audio/jobs/{job_id}/download")
async def download_audio_job(
    job_id: str, audio_jobs: AudioJobManager = Depends(get_audio_job_manager)
) -> FileResponse:
    job = audio_jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Audio job not found")
    if job.status != JobStatus.completed or not job.output_path:
        raise HTTPException(status_code=409, detail="Audio job is not completed yet")
    return FileResponse(path=job.output_path, filename=job.output_path.name, media_type="application/octet-stream")


@router.get("/models", response_model=ModelsResponse)
async def list_models(registry: ModelRegistry = Depends(get_model_registry)) -> ModelsResponse:
    return ModelsResponse(models=[model_entry_to_response(entry) for entry in registry.list()])


@router.get("/models/search", response_model=ModelSearchResponse)
async def search_models(
    q: str = Query(..., min_length=1),
    hf_client: HfClient = Depends(get_hf_client),
) -> ModelSearchResponse:
    try:
        results = await hf_client.search(q)
    except Exception as exc:
        logger.exception("Hugging Face search failed for query %r", q)
        raise HTTPException(status_code=502, detail="Hugging Face search failed") from exc
    return ModelSearchResponse(
        results=[
            HfModelSearchResultResponse(
                id=item.id,
                author=item.author,
                pipeline_tag=item.pipeline_tag,
                downloads=item.downloads,
                likes=item.likes,
                tags=list(item.tags),
            )
            for item in results
        ]
    )


@router.post("/models/install", response_model=CreateInstallResponse, status_code=202)
async def install_model(
    payload: InstallModelRequest,
    installer: ModelInstaller = Depends(get_model_installer),
) -> CreateInstallResponse:
    try:
        install_id = await installer.install_from_hf(payload.repo_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CreateInstallResponse(
        install_id=install_id, status_url=f"/api/v1/models/install/{install_id}"
    )


@router.get("/models/install/{install_id}", response_model=InstallStatusResponse)
async def get_install_status(
    install_id: str, installer: ModelInstaller = Depends(get_model_installer)
) -> InstallStatusResponse:
    job = installer.status(install_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Install job not found")
    return InstallStatusResponse(
        install_id=job.id,
        repo_id=job.repo_id,
        status=job.status.value,
        progress_pct=job.progress_pct,
        model_id=job.model_id,
        error=job.error,
    )


@router.delete("/models/{model_id}", status_code=204)
async def delete_model(
    model_id: str, installer: ModelInstaller = Depends(get_model_installer)
) -> Response:
    try:
        await installer.delete(model_id)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelProtectedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return Response(status_code=204)
