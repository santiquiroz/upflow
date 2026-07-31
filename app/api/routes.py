from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse

from app.api.auth_deps import current_user_from_request, require
from app.config import (
    AUDIO_ENHANCE_MODES,
    AUDIO_RESTORE_MODES,
    INTERP_ENGINES,
    RIFE_ENGINE,
    Settings,
    VideoProfile,
    get_settings,
)
from app.exceptions import ModelNotFoundError, ModelProtectedError, QueueFullError, QuotaExceededError
from app.models import (
    DownloadJob,
    AudioJob,
    GenerationJob,
    JobStatus,
    TranscribeJob,
    UpdateStatus,
    UpscaleJob,
    VideoUpscaleJob,
)
from app.schemas import (
    CreateDownloadJobRequest,
    DownloadJobResponse,
    MediaProbeResponse,
    AnalyzeVideoResponse,
    AudioCapabilitiesResponse,
    AudioJobResponse,
    AudioJobsListResponse,
    AudioTrackResponse,
    CapabilityDomainResponse,
    CapabilityResponse,
    CapabilityTreeResponse,
    ConversionStatusResponse,
    CreateConversionResponse,
    CreateGenerationJobRequest,
    CreateInstallResponse,
    CreateJobResponse,
    DeviceInfoResponse,
    DevicesResponse,
    EditableSettingStatusResponse,
    EditableSettingsResponse,
    EngineInfoResponse,
    GenerationCapabilitiesResponse,
    GenerationJobResponse,
    GenerationJobsListResponse,
    GenerationModelSummary,
    HealthResponse,
    HfModelSearchResultResponse,
    InstallModelRequest,
    InitImageResponse,
    InstallStatusResponse,
    JobResponse,
    JobsListResponse,
    ModelResponse,
    ModelSearchResponse,
    ModelsResponse,
    PreflightResponse,
    ProvisionJobResponse,
    SubtitleTrackResponse,
    SupportedModelResponse,
    TranscribeJobResponse,
    UpdateCheckResponse,
    UpdateSettingRequest,
    UpdateSettingResponse,
    UpscalerPreflightResponse,
    VideoCapabilitiesResponse,
    VideoJobResponse,
    VideoJobsListResponse,
    VideoProfileResponse,
    VoiceCatalogResponse,
)
from app.services.asr_installer import AsrModelInstaller
from app.services.audio_job_manager import AudioJobManager
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.permissions import Permission
from app.services.capabilities import (
    ResolvedCapability,
    group_by_domain,
    resolve_capabilities,
)
from app.services.compat_strategy import CompatStrategy, InstallOptions, strategy_for
from app.services.devices_service import AUTO_DEVICE_ID, DevicesService
from app.services.engines.generation_onnx import generation_dependencies_available
from app.services.generation_converter import GenerationModelConverter
from app.services.generation_installer import (
    CheckpointNotFoundError,
    GenerationModelInstaller,
)
from app.services.generation_compat import classify
from app.services.generation_job_manager import GenerationJobManager
from app.services.generation_preflight import preflight
from app.services.generation_variants import available_precisions_from_names
from app.services.hf_client import (
    ASR_SEARCH_TASK_TAGS,
    GENERATION_SEARCH_TASK_TAGS,
    HfClient,
)
from app.services.job_manager import JobManager
from app.services.media_tools import MediaTools
from app.services.model_installer import ModelInstaller
from app.services.model_preflight import preflight_upscaler
from app.services.model_registry import ModelEntry, ModelKind, ModelRegistry, ModelStatus
from app.services.pack_provisioner import PackProvisioner, ProvisionJob, UnknownPackError
from app.services.settings_service import editable_settings_status, update_setting
from app.services.storage import StorageService
from app.services.download_job_manager import (
    DownloadJobManager,
    describe_failure,
    validate_url,
)
from fetchflow import engine as fetch_engine
from app.services.transcribe_job_manager import TranscribeJobManager
from app.services.stream_analysis import parse_audio_tracks, parse_subtitle_tracks
from app.services.update_service import UpdateService
from app.services.video_job_manager import VideoJobManager
from app.services.voice_chain import delivery_choices, step_catalog

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


def _original_filename_from_staged_upload(uploads_path: Path, upload_token: str, default: str) -> str:
    # Best-effort only: create_job's own _resolve_source_path raises the real
    # error for a missing token, so this never needs to fail the request.
    matches = sorted(uploads_path.glob(f"{upload_token}-*"))
    if not matches:
        return default
    return matches[0].name[len(upload_token) + 1 :]


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


def _parse_voice_steps(raw: object) -> list[str]:
    """Los pasos llegan como lista separada por comas en un campo de formulario.

    El ORDEN de lo que llega no importa: steps_from_selection lo reordena segun
    el catalogo, porque la cadena tiene causalidad y un request no deberia poder
    invertirla.

    Acepta `object` y no `str | None` a proposito: los tests de ruta de este
    repo llaman las corrutinas DIRECTO, sin pasar por FastAPI, asi que un
    parametro no provisto llega como el sentinel `Form(...)` en vez de None.
    Tratar cualquier cosa que no sea str como "no vino" cubre los dos caminos.
    """
    if not isinstance(raw, str) or not raw.strip():
        return []
    return [entry.strip() for entry in raw.split(",") if entry.strip()]


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


def get_generation_job_manager(request: Request) -> GenerationJobManager:
    return request.app.state.generation_job_manager


def get_generation_installer(request: Request) -> GenerationModelInstaller:
    return request.app.state.generation_installer


def get_generation_converter(request: Request) -> GenerationModelConverter:
    return request.app.state.generation_converter


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
        owner_id=job.owner_id,
    )


def _can_view_job(job: Any, user: AuthenticatedUser) -> bool:
    return Permission.jobs_read_all in user.permissions or job.owner_id == user.id


def _can_cancel_job(job: Any, user: AuthenticatedUser) -> bool:
    return Permission.jobs_cancel_any in user.permissions or job.owner_id == user.id


def _require_read_all_if_requested(all_users: bool, current_user: AuthenticatedUser) -> None:
    if all_users and Permission.jobs_read_all not in current_user.permissions:
        raise HTTPException(status_code=403, detail="No tenés permiso para ver los jobs de todos los usuarios")


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
        target_height=job.target_height,
        fps_multiplier=job.fps_multiplier,
        target_fps=job.target_fps,
        audio_enhance=job.audio_enhance,
        audio_restore=job.audio_restore,
        audio_track_indices=job.audio_track_indices,
        keep_subtitles=job.keep_subtitles,
        audio_output_format=job.audio_output_format,
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
        owner_id=job.owner_id,
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
        output_format=job.output_format,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        progress_pct=_progress_pct_from_metadata(job.metadata),
        stages=job.metadata.get("stages"),
        error=job.error,
        download_url=download_url,
        owner_id=job.owner_id,
    )


def generation_job_to_response(job: GenerationJob) -> GenerationJobResponse:
    download_url = (
        f"/api/v1/generation/jobs/{job.id}/download" if job.status == JobStatus.completed else None
    )
    return GenerationJobResponse(
        id=job.id, status=job.status, prompt=job.prompt, negative_prompt=job.negative_prompt,
        model_id=job.model_id, steps=job.steps, guidance=job.guidance, width=job.width,
        height=job.height, seed=job.seed, device=job.device, auto_upscale=job.auto_upscale,
        created_at=job.created_at, started_at=job.started_at, finished_at=job.finished_at,
        progress_pct=_progress_pct_from_metadata(job.metadata), stages=job.metadata.get("stages"),
        error=job.error, download_url=download_url, owner_id=job.owner_id,
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
    from app.services import ep_registry

    def snapshot() -> tuple[list[dict], list[DeviceInfoResponse]]:
        # active_ep_for_device puede disparar la inicialización del registry
        # (enumeración DXGI + registro de plugins): fuera del event loop.
        raw = devices.list_devices()
        enriched = []
        for item in raw:
            ep_status = ep_registry.active_ep_for_device(item["id"], devices.settings)
            enriched.append(
                DeviceInfoResponse(
                    **item,
                    active_ep=ep_status.ep_name,
                    ep_label=ep_status.label,
                    ep_state=ep_status.state,
                    ep_detail=ep_status.detail,
                )
            )
        return raw, enriched

    raw_list, device_list = await asyncio.to_thread(snapshot)
    default_device = devices.resolve_default(raw_list)
    return DevicesResponse(devices=device_list, default_device_id=default_device["id"])


@router.post(
    "/jobs", response_model=CreateJobResponse, status_code=202,
    dependencies=[Depends(require(Permission.jobs_create))],
)
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
    current_user = current_user_from_request(request)

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
            owner=current_user,
        )
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except QuotaExceededError as exc:
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


@router.post(
    "/video/jobs", response_model=CreateJobResponse, status_code=202,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def create_video_job(
    request: Request,
    file: UploadFile | None = File(default=None),
    upload_token: str | None = Form(default=None),
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
    # Alto de salida pedido. Presente = el usuario pidio una RESOLUCION y no un
    # multiplicador ciego, que es lo que llevaba a pedir 15360x8640 sin querer.
    target_height: int | None = Form(default=None),
    audio_enhance: str | None = Form(default=None),
    audio_restore: str | None = Form(default=None),
    audio_track_indices: str | None = Form(default=None),
    keep_subtitles: bool = Form(default=False),
    audio_output_format: str | None = Form(default=None),
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

    # FastAPI passes these as their declared type (str|None, bool); a direct
    # unit-test call that omits one receives the Form() sentinel instead
    # (always truthy), so normalize anything of the wrong type to its real
    # default -- same pattern already used below for backend/video_encoder/
    # interp_engine.
    upload_token_value = upload_token if isinstance(upload_token, str) else None
    audio_track_indices_value = audio_track_indices if isinstance(audio_track_indices, str) else None
    keep_subtitles_value = keep_subtitles if isinstance(keep_subtitles, bool) else False

    has_file = bool(file and file.filename)
    if has_file == bool(upload_token_value):
        raise HTTPException(status_code=400, detail="Provide exactly one of file or upload_token")

    backend_value = backend if isinstance(backend, str) else None
    video_encoder_value = video_encoder if isinstance(video_encoder, str) else "auto"
    audio_output_format_value = audio_output_format if isinstance(audio_output_format, str) else "auto"
    interp_engine_value = interp_engine if isinstance(interp_engine, str) else RIFE_ENGINE

    resolved_device = await resolve_request_device(device, devices, settings)
    current_user = current_user_from_request(request)

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

    destination: Path | None = None
    job: VideoUpscaleJob | None = None
    try:
        parsed_audio_track_indices = (
            [int(i) for i in audio_track_indices_value.split(",") if i.strip()]
            if audio_track_indices_value
            else None
        )
        if has_file:
            original_name = Path(file.filename or "upload.mp4").name
            safe_name = sanitize_filename(original_name, default="upload.mp4")
            new_upload_token = uuid4().hex
            destination = settings.uploads_path / f"{new_upload_token}-{safe_name}"
            await storage.save_upload(file, destination, max_mb=settings.max_video_upload_mb)
            source_path = destination
            resolved_upload_token = None
            # The freshly-generated upload token doubles as the job id here (as
            # before this task): one request == one upload == one job, so no
            # collision risk. When reusing a staged upload_token below, the job
            # id must instead be independent -- see the else branch.
            new_job_id = new_upload_token
        else:
            original_name = _original_filename_from_staged_upload(
                settings.uploads_path, upload_token_value, default="upload.mp4"
            )
            source_path = None
            resolved_upload_token = upload_token_value
            # A staged upload_token can be reused across multiple job-creation
            # attempts (e.g. retrying with different settings after a
            # validation error), so it must NOT double as the job id or a
            # second successful job would overwrite the first in jobs[].
            new_job_id = uuid4().hex

        job = await video_jobs.create_job(
            source_path=source_path,
            upload_token=resolved_upload_token,
            original_filename=original_name,
            model_name=resolved.model_name,
            scale=resolved.scale,
            output_container=resolved.output_container,
            video_codec=resolved.video_codec,
            video_preset=resolved.video_preset,
            crf=resolved.crf,
            keep_audio=resolved.keep_audio,
            target_height=target_height if isinstance(target_height, int) else None,
            fps_multiplier=resolved.fps_multiplier,
            target_fps=resolved.target_fps,
            audio_enhance=resolved.audio_enhance,
            audio_restore=audio_restore,
            audio_track_indices=parsed_audio_track_indices,
            keep_subtitles=keep_subtitles_value,
            audio_output_format=audio_output_format_value,
            interp_engine=interp_engine_value,
            model_id=model_id,
            device=resolved_device,
            backend=backend_value,
            video_encoder=video_encoder_value,
            job_id=new_job_id,
            owner=current_user,
        )
        job.metadata["profileKey"] = profile_key
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except QuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while creating video job")
        raise HTTPException(status_code=500, detail="Failed to process the uploaded video") from exc
    finally:
        # Only ever clean up a destination THIS request wrote (the `file`
        # path). A staged upload_token pre-dates this request (written by a
        # prior /video/analyze call) and must survive a failed attempt here
        # so the caller can retry with different job parameters without
        # re-uploading the video.
        if job is None and destination is not None and destination.exists():
            destination.unlink(missing_ok=True)

    return CreateJobResponse(
        job_id=job.id,
        status=job.status,
        status_url=f"/api/v1/video/jobs/{job.id}",
        download_url=None,
    )


@router.post(
    "/video/analyze", response_model=AnalyzeVideoResponse,
    dependencies=[Depends(require(Permission.jobs_create))],
)
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


@router.get("/jobs", response_model=JobsListResponse)
async def list_jobs(
    all_users: bool = Query(default=False, alias="all"),
    jobs: JobManager = Depends(get_job_manager),
    current_user: AuthenticatedUser = Depends(require(Permission.jobs_read_own)),
) -> JobsListResponse:
    _require_read_all_if_requested(all_users, current_user)
    visible = [job for job in jobs.jobs.values() if all_users or job.owner_id == current_user.id]
    return JobsListResponse(jobs=[job_to_response(job) for job in visible])


@router.get("/jobs/{job_id}", response_model=JobResponse, dependencies=[Depends(require(Permission.jobs_read_own))])
async def get_job(
    job_id: str,
    jobs: JobManager = Depends(get_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> JobResponse:
    job = jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Job not found")
    return job_to_response(job)


@router.post(
    "/jobs/{job_id}/cancel", response_model=JobResponse,
    dependencies=[Depends(require(Permission.jobs_cancel_own))],
)
async def cancel_job(
    job_id: str,
    jobs: JobManager = Depends(get_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> JobResponse:
    job = jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if job is None or (current_user is not None and not _can_cancel_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Job not found")
    if not jobs.cancel_job(job_id):
        raise HTTPException(status_code=409, detail="Job already finished")
    return job_to_response(job)


@router.get("/video/jobs", response_model=VideoJobsListResponse)
async def list_video_jobs(
    all_users: bool = Query(default=False, alias="all"),
    video_jobs: VideoJobManager = Depends(get_video_job_manager),
    current_user: AuthenticatedUser = Depends(require(Permission.jobs_read_own)),
) -> VideoJobsListResponse:
    _require_read_all_if_requested(all_users, current_user)
    visible = [job for job in video_jobs.jobs.values() if all_users or job.owner_id == current_user.id]
    return VideoJobsListResponse(jobs=[video_job_to_response(job) for job in visible])


@router.get(
    "/video/jobs/{job_id}", response_model=VideoJobResponse,
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def get_video_job(
    job_id: str,
    video_jobs: VideoJobManager = Depends(get_video_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> VideoJobResponse:
    job = video_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Video job not found")
    return video_job_to_response(job)


@router.post(
    "/video/jobs/{job_id}/cancel", response_model=VideoJobResponse,
    dependencies=[Depends(require(Permission.jobs_cancel_own))],
)
async def cancel_video_job(
    job_id: str,
    video_jobs: VideoJobManager = Depends(get_video_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> VideoJobResponse:
    job = video_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if job is None or (current_user is not None and not _can_cancel_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Video job not found")
    if not video_jobs.cancel_job(job_id):
        raise HTTPException(status_code=409, detail="Job already finished")
    return video_job_to_response(job)


@router.get("/jobs/{job_id}/download", dependencies=[Depends(require(Permission.jobs_read_own))])
async def download_job(
    job_id: str,
    jobs: JobManager = Depends(get_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> FileResponse:
    job = jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.completed or not job.output_path:
        raise HTTPException(status_code=409, detail="Job is not completed yet")
    return FileResponse(path=job.output_path, filename=job.output_path.name, media_type="application/octet-stream")


@router.get("/video/jobs/{job_id}/download", dependencies=[Depends(require(Permission.jobs_read_own))])
async def download_video_job(
    job_id: str,
    video_jobs: VideoJobManager = Depends(get_video_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> FileResponse:
    job = video_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Video job not found")
    if job.status != JobStatus.completed or not job.output_path:
        raise HTTPException(status_code=409, detail="Video job is not completed yet")
    return FileResponse(path=job.output_path, filename=job.output_path.name, media_type="application/octet-stream")


@router.post(
    "/audio/jobs", response_model=CreateJobResponse, status_code=202,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def create_audio_job(
    request: Request,
    file: UploadFile = File(...),
    denoise: str | None = Form(default=None),
    restore: str | None = Form(default=None),
    device: str | None = Form(default=None),
    output_format: str = Form(default="flac"),
    voice_steps: str | None = Form(default=None),
    voice_delivery: str | None = Form(default=None),
    voice_presence_db: float | None = Form(default=None),
    audio_jobs: AudioJobManager = Depends(get_audio_job_manager),
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> CreateJobResponse:
    original_name = Path(file.filename or "upload.wav").name
    safe_name = sanitize_filename(original_name, default="upload.wav")
    token = uuid4().hex
    destination = settings.uploads_path / f"{token}-{safe_name}"
    current_user = current_user_from_request(request)

    job: AudioJob | None = None
    try:
        await storage.save_upload(file, destination, max_mb=settings.max_audio_upload_mb)
        job = await audio_jobs.create_job(
            source_path=destination,
            original_filename=original_name,
            denoise=denoise,
            restore=restore,
            device=device,
            output_format=output_format,
            voice_steps=_parse_voice_steps(voice_steps),
            voice_delivery=voice_delivery if isinstance(voice_delivery, str) else None,
            voice_presence_db=(
                voice_presence_db if isinstance(voice_presence_db, (int, float)) else None
            ),
            job_id=token,
            owner=current_user,
        )
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except QuotaExceededError as exc:
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


def get_download_job_manager(request: Request) -> DownloadJobManager:
    return request.app.state.download_jobs


def download_job_to_response(job: DownloadJob) -> DownloadJobResponse:
    return DownloadJobResponse(
        id=job.id,
        status=job.status,
        url=job.url,
        max_height=job.max_height,
        audio_only=job.audio_only,
        audio_format=job.audio_format,
        audio_bitrate_kbps=job.audio_bitrate_kbps,
        video_container=job.video_container,
        media_title=job.media_title,
        media_uploader=job.media_uploader,
        extractor=job.extractor,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        progress_pct=job.progress_pct,
        downloaded_bytes=job.downloaded_bytes,
        total_bytes=job.total_bytes,
        # Solo el NOMBRE: la ruta absoluta expondria la estructura de directorios del
        # servidor sin darle nada util a quien la lee.
        output_files=[path.name for path in job.output_paths],
        # El DIRECTORIO si viaja (a diferencia de las rutas completas por archivo): es
        # la carpeta que el usuario ya ve en Ajustes, y sin ella la UI decia el nombre
        # del archivo sin decir donde buscarlo.
        output_directory=str(job.output_paths[0].parent) if job.output_paths else "",
        error=job.error,
        owner_id=job.owner_id,
    )


@router.post(
    "/download/probe",
    response_model=MediaProbeResponse,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def probe_media(payload: CreateDownloadJobRequest) -> MediaProbeResponse:
    """Que hay en esta URL, sin descargar.

    Existe para que la UI muestre titulo, duracion y calidades ANTES de comprometerse, y
    sobre todo para que se vea que una URL es una playlist de 200 items antes de
    disparar 200 descargas.
    """
    try:
        validate_url(payload.url)
        info = await asyncio.to_thread(fetch_engine.probe, payload.url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except fetch_engine.FetchUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - el motivo del sitio es lo util
        raise HTTPException(status_code=422, detail=describe_failure(exc)) from exc
    return MediaProbeResponse(
        title=info.title,
        duration_seconds=info.duration_seconds,
        uploader=info.uploader,
        extractor=info.extractor,
        is_playlist=info.is_playlist,
        entry_count=info.entry_count,
        available_heights=list(info.available_heights),
        thumbnail_url=info.thumbnail_url,
    )


@router.post(
    "/download/jobs",
    response_model=DownloadJobResponse,
    status_code=202,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def create_download_job(
    payload: CreateDownloadJobRequest,
    download_jobs: DownloadJobManager = Depends(get_download_job_manager),
    request: Request = None,
) -> DownloadJobResponse:
    try:
        job = await download_jobs.create_job(
            url=payload.url,
            max_height=payload.max_height,
            audio_only=payload.audio_only,
            audio_format=payload.audio_format,
            audio_bitrate_kbps=payload.audio_bitrate_kbps,
            video_container=payload.video_container,
            include_playlist=payload.include_playlist,
            playlist_limit=payload.playlist_limit,
            subtitle_languages=payload.subtitle_languages,
            owner=current_user_from_request(request),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    return download_job_to_response(job)


@router.get(
    "/download/jobs/{job_id}",
    response_model=DownloadJobResponse,
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def get_download_job(
    job_id: str,
    download_jobs: DownloadJobManager = Depends(get_download_job_manager),
    request: Request = None,
) -> DownloadJobResponse:
    job = download_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    # 404 y no 403: un 403 confirmaria que el job existe, y con el la URL que otro
    # usuario decidio bajar.
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Download job not found")
    return download_job_to_response(job)


@router.post(
    "/download/jobs/{job_id}/cancel",
    response_model=DownloadJobResponse,
    dependencies=[Depends(require(Permission.jobs_cancel_own))],
)
async def cancel_download_job(
    job_id: str,
    download_jobs: DownloadJobManager = Depends(get_download_job_manager),
    request: Request = None,
) -> DownloadJobResponse:
    job = download_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_cancel_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Download job not found")
    download_jobs.cancel_job(job_id)
    return download_job_to_response(job)


@router.get(
    "/download/jobs/{job_id}/download",
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def download_download_job_file(
    job_id: str,
    index: int = Query(default=0, ge=0),
    download_jobs: DownloadJobManager = Depends(get_download_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> FileResponse:
    """El archivo producido, servido por HTTP.

    Sin esto un usuario remoto ve el NOMBRE del archivo y la ruta de otra maquina:
    la descarga termina y el resultado queda inalcanzable. `index` porque una
    playlist produce varios archivos y cada uno necesita su propio link.
    """
    job = download_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Download job not found")
    if job.status != JobStatus.completed or not job.output_paths:
        raise HTTPException(status_code=409, detail="Download job is not completed yet")
    if index >= len(job.output_paths):
        raise HTTPException(status_code=404, detail="No such file in this download job")
    path = job.output_paths[index]
    # El sweeper de retencion puede borrar el archivo antes de que el job expire.
    if not path.is_file():
        raise HTTPException(status_code=404, detail="El archivo ya no esta en disco")
    return FileResponse(path=path, filename=path.name, media_type="application/octet-stream")


def get_transcribe_job_manager(request: Request) -> TranscribeJobManager:
    return request.app.state.transcribe_jobs


def transcribe_job_to_response(job: TranscribeJob) -> TranscribeJobResponse:
    return TranscribeJobResponse(
        id=job.id,
        status=job.status,
        original_filename=job.original_filename,
        model_id=job.model_id,
        language=job.language,
        device=job.device,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        progress_pct=job.progress_pct,
        text=job.text,
        error=job.error,
        owner_id=job.owner_id,
        download_url=(
            f"/api/v1/transcribe/jobs/{job.id}/download"
            if job.status == JobStatus.completed and job.output_path
            else None
        ),
    )


@router.post(
    "/transcribe/jobs", response_model=CreateJobResponse, status_code=202,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def create_transcribe_job(
    request: Request,
    file: UploadFile = File(...),
    model_id: str = Form(...),
    language: str | None = Form(default=None),
    device: str | None = Form(default=None),
    transcribe_jobs: TranscribeJobManager = Depends(get_transcribe_job_manager),
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> CreateJobResponse:
    original_name = Path(file.filename or "upload.wav").name
    safe_name = sanitize_filename(original_name, default="upload.wav")
    token = uuid4().hex
    destination = settings.uploads_path / f"{token}-{safe_name}"
    current_user = current_user_from_request(request)

    job: TranscribeJob | None = None
    try:
        await storage.save_upload(file, destination, max_mb=settings.max_audio_upload_mb)
        job = await transcribe_jobs.create_job(
            source_path=destination,
            original_filename=original_name,
            model_id=model_id,
            # Los tests de ruta llaman las corrutinas DIRECTO, sin FastAPI, asi que
            # un Form no provisto llega como su sentinel en vez de None.
            language=language if isinstance(language, str) and language else None,
            device=device if isinstance(device, str) and device else None,
            job_id=token,
            owner=current_user,
        )
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except QuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while creating transcribe job")
        raise HTTPException(
            status_code=500, detail="Failed to process the uploaded audio"
        ) from exc
    finally:
        if job is None and destination.exists():
            destination.unlink(missing_ok=True)

    return CreateJobResponse(
        job_id=job.id,
        status=job.status,
        status_url=f"/api/v1/transcribe/jobs/{job.id}",
        download_url=None,
    )


@router.get(
    "/transcribe/jobs/{job_id}",
    response_model=TranscribeJobResponse,
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def get_transcribe_job(
    job_id: str,
    transcribe_jobs: TranscribeJobManager = Depends(get_transcribe_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case injection
    # still recognizes it -- `lenient_issubclass` rejects unions. Direct/unit-test
    # calls that omit this kwarg still get `None`.
    request: Request = None,
) -> TranscribeJobResponse:
    job = transcribe_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    # 404 y no 403 a proposito: un 403 confirmaria que el job existe. Una
    # transcripcion es el contenido de un audio ajeno, asi que ni su existencia se
    # filtra.
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Transcribe job not found")
    return transcribe_job_to_response(job)


@router.post(
    "/transcribe/jobs/{job_id}/cancel",
    response_model=TranscribeJobResponse,
    dependencies=[Depends(require(Permission.jobs_cancel_own))],
)
async def cancel_transcribe_job(
    job_id: str,
    transcribe_jobs: TranscribeJobManager = Depends(get_transcribe_job_manager),
    request: Request = None,
) -> TranscribeJobResponse:
    job = transcribe_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_cancel_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Transcribe job not found")
    transcribe_jobs.cancel_job(job_id)
    return transcribe_job_to_response(job)


@router.get(
    "/transcribe/jobs/{job_id}/download",
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def download_transcribe_job(
    job_id: str,
    transcribe_jobs: TranscribeJobManager = Depends(get_transcribe_job_manager),
    request: Request = None,
) -> FileResponse:
    job = transcribe_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Transcribe job not found")
    if job.status != JobStatus.completed or not job.output_path:
        raise HTTPException(status_code=409, detail="Transcribe job is not completed yet")
    return FileResponse(
        path=job.output_path,
        filename=f"{Path(job.original_filename).stem}.txt",
        media_type="text/plain; charset=utf-8",
    )


def get_asr_installer(request: Request) -> AsrModelInstaller:
    return request.app.state.asr_installer


@router.get("/asr/models/search", response_model=ModelSearchResponse)
async def search_asr_models(
    q: str = Query("", max_length=200),
    hf_client: HfClient = Depends(get_hf_client),
) -> ModelSearchResponse:
    """Modelos de reconocimiento de voz en Hugging Face.

    El filtro por TAG es lo que decide que es un modelo de ASR: la clasificacion por
    nombres de archivo no puede distinguirlo de otro modelo de audio, y decir lo
    contrario seria prometer una deteccion que no existe.
    """
    try:
        results = await hf_client.search(q, task_tags=ASR_SEARCH_TASK_TAGS, sort="downloads")
    except Exception as exc:
        logger.exception("Hugging Face ASR search failed for query %r", q)
        raise HTTPException(status_code=502, detail="Hugging Face search failed") from exc
    return _search_results_to_response(results, strategy_for("audio"))


@router.post(
    "/asr/models/install", response_model=CreateInstallResponse, status_code=202,
    dependencies=[Depends(require(Permission.models_install))],
)
async def install_asr_model(
    payload: InstallModelRequest,
    installer: AsrModelInstaller = Depends(get_asr_installer),
) -> CreateInstallResponse:
    install_id = await installer.install_from_hf(payload.repo_id)
    return CreateInstallResponse(
        install_id=install_id,
        status_url=f"/api/v1/asr/models/install/{install_id}",
    )


@router.get("/asr/models/install/{install_id}", response_model=InstallStatusResponse)
async def get_asr_install_status(
    install_id: str,
    installer: AsrModelInstaller = Depends(get_asr_installer),
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


@router.get("/audio/voice-catalog", response_model=VoiceCatalogResponse)
async def get_voice_catalog() -> VoiceCatalogResponse:
    """Los pasos y destinos de la cadena de voz.

    La ESTRUCTURA viene del backend a proposito: el orden de los pasos tiene
    causalidad (ver build_filter_chain) y los numeros de loudness son
    especificaciones publicadas, asi que hay una sola fuente de verdad. La COPIA
    en cambio viaja como clave de traduccion y la arma el frontend, que es la
    capa que conoce el idioma activo.
    """
    return VoiceCatalogResponse(
        steps=[
            {
                "id": info.id,
                "label_key": info.label_key,
                "description_key": info.description_key,
                "kind": info.kind,
                "default_enabled": info.default_enabled,
            }
            for info in step_catalog()
        ],
        deliveries=[
            {
                "id": choice["id"],
                "label_key": choice["labelKey"],
                "description_key": choice["descriptionKey"],
                "lufs": choice["lufs"],
                "true_peak_db": choice["truePeakDb"],
            }
            for choice in delivery_choices()
        ],
    )


@router.get("/audio/jobs", response_model=AudioJobsListResponse)
async def list_audio_jobs(
    all_users: bool = Query(default=False, alias="all"),
    audio_jobs: AudioJobManager = Depends(get_audio_job_manager),
    current_user: AuthenticatedUser = Depends(require(Permission.jobs_read_own)),
) -> AudioJobsListResponse:
    _require_read_all_if_requested(all_users, current_user)
    visible = [job for job in audio_jobs.jobs.values() if all_users or job.owner_id == current_user.id]
    return AudioJobsListResponse(jobs=[audio_job_to_response(job) for job in visible])


@router.get(
    "/audio/jobs/{job_id}", response_model=AudioJobResponse,
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def get_audio_job(
    job_id: str,
    audio_jobs: AudioJobManager = Depends(get_audio_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> AudioJobResponse:
    job = audio_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Audio job not found")
    return audio_job_to_response(job)


@router.post(
    "/audio/jobs/{job_id}/cancel", response_model=AudioJobResponse,
    dependencies=[Depends(require(Permission.jobs_cancel_own))],
)
async def cancel_audio_job(
    job_id: str,
    audio_jobs: AudioJobManager = Depends(get_audio_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> AudioJobResponse:
    job = audio_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if job is None or (current_user is not None and not _can_cancel_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Audio job not found")
    if not audio_jobs.cancel_job(job_id):
        raise HTTPException(status_code=409, detail="Job already finished")
    return audio_job_to_response(job)


@router.get("/audio/jobs/{job_id}/download", dependencies=[Depends(require(Permission.jobs_read_own))])
async def download_audio_job(
    job_id: str,
    audio_jobs: AudioJobManager = Depends(get_audio_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> FileResponse:
    job = audio_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Audio job not found")
    if job.status != JobStatus.completed or not job.output_path:
        raise HTTPException(status_code=409, detail="Audio job is not completed yet")
    return FileResponse(path=job.output_path, filename=job.output_path.name, media_type="application/octet-stream")


@router.get("/models", response_model=ModelsResponse)
async def list_models(registry: ModelRegistry = Depends(get_model_registry)) -> ModelsResponse:
    return ModelsResponse(models=[model_entry_to_response(entry) for entry in registry.list()])


def _result_to_response(item, strategy: CompatStrategy) -> HfModelSearchResultResponse:
    verdict, reason = strategy.classify(item.filenames, item.gated)
    # Un repo ready_onnx no tiene paso de export, asi que no hay nada que elegir:
    # no se ofrecen opciones (ver alcance de B en el spec de 2026-07-28).
    options = (
        strategy.install_options(item.filenames)
        if verdict == "needs_conversion"
        else InstallOptions()
    )
    return HfModelSearchResultResponse(
        id=item.id,
        author=item.author,
        pipeline_tag=item.pipeline_tag,
        downloads=item.downloads,
        likes=item.likes,
        tags=list(item.tags),
        compat=verdict,
        compat_reason_key=reason.key,
        compat_reason_params=dict(reason.params),
        available_precisions=list(options.precisions),
    )


def _search_results_to_response(
    results: list, strategy: CompatStrategy
) -> ModelSearchResponse:
    return ModelSearchResponse(
        results=[_result_to_response(item, strategy) for item in results]
    )


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
    return _search_results_to_response(results, strategy_for("image"))
@router.get("/models/preflight", response_model=UpscalerPreflightResponse)
async def preflight_upscaler_model(
    request: Request,
    repo_id: str = Query(..., alias="repoId"),
    hf_client: HfClient = Depends(get_hf_client),
    settings: Settings = Depends(get_settings),
) -> UpscalerPreflightResponse:
    """Capacidad y compatibilidad de un repo de upscaler antes de instalarlo.

    Reusa las mediciones genericas del pre-flight de generacion. NO estima pico de
    VRAM: el factor de vram_estimate asume activaciones que crecen con la
    resolucion, y para un upscaler que hace tiling esa premisa es falsa. La VRAM
    libre viaja como dato, sin veredicto de "no entra".
    """
    report = await preflight_upscaler(
        hf_client=hf_client,
        devices_service=request.app.state.devices_service,
        settings=settings,
        probes=request.app.state.resource_probes,
        repo_id=repo_id,
        strategy=strategy_for("image"),
    )
    return UpscalerPreflightResponse(**asdict(report))


@router.post(
    "/models/install", response_model=CreateInstallResponse, status_code=202,
    dependencies=[Depends(require(Permission.models_install))],
)
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
        conversion_id=job.conversion_id,
    )


@router.delete(
    "/models/{model_id}", status_code=204,
    dependencies=[Depends(require(Permission.models_delete))],
)
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


INIT_IMAGE_TOKEN_PATTERN = "%s-*"


def _resolve_init_image(settings: Settings, token: str | None) -> Path | None:
    """Traduce el token a la ruta staged, sin confiar en el token.

    El token viene del cliente, asi que se valida que sea hexadecimal antes de
    usarlo en un glob: sin eso, un token con separadores de ruta podria hacer que
    el glob mire fuera del directorio de uploads.
    """
    if token is None:
        return None
    if not token or len(token) > 64 or any(char not in "0123456789abcdef" for char in token):
        raise HTTPException(status_code=400, detail="init_image_token invalido")
    matches = sorted(settings.uploads_path.glob(INIT_IMAGE_TOKEN_PATTERN % token))
    if not matches:
        raise HTTPException(status_code=400, detail="init_image_token desconocido o expirado")
    return matches[0]


@router.post(
    "/generation/init-image",
    response_model=InitImageResponse,
    status_code=201,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def upload_init_image(
    file: UploadFile = File(...),
    storage: StorageService = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> InitImageResponse:
    """Deja la imagen de partida en staging y devuelve su token.

    Va aparte del job para que POST /generation/jobs siga siendo JSON: volverlo
    multipart cambiaria el contrato de todos sus clientes por una capacidad que no
    todos usan. Es el mismo patron que /video/analyze ya usa.
    """
    original_name = Path(file.filename or "init.png").name
    safe_name = sanitize_filename(original_name, default="init.png")
    token = uuid4().hex
    destination = settings.uploads_path / f"{token}-{safe_name}"
    await storage.save_upload(file, destination)

    try:
        from PIL import Image

        with Image.open(destination) as image:
            width, height = image.size
    except Exception as exc:  # noqa: BLE001 - un archivo que no es imagen se rechaza aca
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400, detail=f"El archivo no es una imagen legible: {exc}"
        ) from exc

    return InitImageResponse(
        init_image_token=token,
        original_filename=original_name,
        width=width,
        height=height,
    )


@router.post(
    "/generation/jobs", response_model=GenerationJobResponse, status_code=201,
    dependencies=[Depends(require(Permission.jobs_create))],
)
async def create_generation_job(
    payload: CreateGenerationJobRequest,
    generation_jobs: GenerationJobManager = Depends(get_generation_job_manager),
    settings: Settings = Depends(get_settings),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> GenerationJobResponse:
    current_user = current_user_from_request(request)
    init_image_path = _resolve_init_image(settings, payload.init_image_token)
    try:
        job = await generation_jobs.create_job(
            prompt=payload.prompt, negative_prompt=payload.negative_prompt, model_id=payload.model_id,
            steps=payload.steps, guidance=payload.guidance, width=payload.width, height=payload.height,
            seed=payload.seed, device=payload.device,
            init_image_path=init_image_path, strength=payload.strength,
            auto_upscale=payload.auto_upscale,
            upscale_model_name=payload.upscale_model_name, upscale_scale=payload.upscale_scale,
            upscale_model_id=payload.upscale_model_id, owner=current_user,
        )
    except QueueFullError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except QuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while creating generation job")
        raise HTTPException(status_code=500, detail="Failed to create the generation job") from exc
    return generation_job_to_response(job)


@router.get("/generation/jobs", response_model=GenerationJobsListResponse)
async def list_generation_jobs(
    all_users: bool = Query(default=False, alias="all"),
    generation_jobs: GenerationJobManager = Depends(get_generation_job_manager),
    current_user: AuthenticatedUser = Depends(require(Permission.jobs_read_own)),
) -> GenerationJobsListResponse:
    _require_read_all_if_requested(all_users, current_user)
    visible = [job for job in generation_jobs.jobs.values() if all_users or job.owner_id == current_user.id]
    return GenerationJobsListResponse(jobs=[generation_job_to_response(job) for job in visible])


@router.get(
    "/generation/jobs/{job_id}", response_model=GenerationJobResponse,
    dependencies=[Depends(require(Permission.jobs_read_own))],
)
async def get_generation_job(
    job_id: str, generation_jobs: GenerationJobManager = Depends(get_generation_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> GenerationJobResponse:
    job = generation_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Generation job not found")
    return generation_job_to_response(job)


@router.post(
    "/generation/jobs/{job_id}/cancel", response_model=GenerationJobResponse,
    dependencies=[Depends(require(Permission.jobs_cancel_own))],
)
async def cancel_generation_job(
    job_id: str, generation_jobs: GenerationJobManager = Depends(get_generation_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> GenerationJobResponse:
    job = generation_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if job is None or (current_user is not None and not _can_cancel_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Generation job not found")
    if not generation_jobs.cancel_job(job_id):
        raise HTTPException(status_code=409, detail="Job already finished")
    return generation_job_to_response(job)


@router.get("/generation/jobs/{job_id}/download", dependencies=[Depends(require(Permission.jobs_read_own))])
async def download_generation_job(
    job_id: str, generation_jobs: GenerationJobManager = Depends(get_generation_job_manager),
    # Bare `Request` (not `Request | None`) so FastAPI's special-case
    # injection still recognizes it -- `lenient_issubclass` rejects unions.
    # Direct/unit-test calls that omit this kwarg still get `None`.
    request: Request = None,
) -> FileResponse:
    job = generation_jobs.get_job(job_id)
    current_user = current_user_from_request(request)
    if not job or (current_user is not None and not _can_view_job(job, current_user)):
        raise HTTPException(status_code=404, detail="Generation job not found")
    if job.status != JobStatus.completed or not job.output_path:
        raise HTTPException(status_code=409, detail="Generation job is not completed yet")
    return FileResponse(path=job.output_path, filename=job.output_path.name, media_type="image/png")


@router.get("/generation/capabilities", response_model=GenerationCapabilitiesResponse)
async def generation_capabilities(
    registry: ModelRegistry = Depends(get_model_registry),
    devices_service: DevicesService = Depends(get_devices_service),
) -> GenerationCapabilitiesResponse:
    available, reason = generation_dependencies_available()
    if not available:
        return GenerationCapabilitiesResponse(available=False, reason=reason, cpu_only=True)
    # error queda afuera: una conversion fallida se ve en Models con su motivo;
    # el dropdown de Generate no es lugar para un modelo que no existe en disco.
    models = [
        GenerationModelSummary(id=entry.id, name=entry.name, status=entry.status.value)
        for entry in registry.list()
        if entry.kind == ModelKind.diffusion_onnx and entry.status != ModelStatus.error
    ]
    device_infos = devices_service.list_devices()
    return GenerationCapabilitiesResponse(
        available=True,
        models=models,
        devices=[info["id"] for info in device_infos],
        cpu_only=all(info["kind"] != "gpu" for info in device_infos),
    )


@router.post(
    "/generation/models", response_model=CreateInstallResponse, status_code=202,
    dependencies=[Depends(require(Permission.models_install))],
)
async def install_generation_model(
    payload: InstallModelRequest,
    installer: GenerationModelInstaller = Depends(get_generation_installer),
) -> CreateInstallResponse:
    try:
        install_id = await installer.install_from_hf(
            payload.repo_id,
            precision=payload.precision or "fp16",
            checkpoint_path=payload.checkpoint_path,
        )
    except CheckpointNotFoundError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CreateInstallResponse(
        install_id=install_id, status_url=f"/api/v1/generation/models/install/{install_id}"
    )


@router.get("/generation/models/install/{install_id}", response_model=InstallStatusResponse)
async def get_generation_install_status(
    install_id: str, installer: GenerationModelInstaller = Depends(get_generation_installer)
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
        conversion_id=job.conversion_id,
    )


@router.post(
    "/generation/models/convert", response_model=CreateConversionResponse, status_code=202,
    dependencies=[Depends(require(Permission.models_install))],
)
async def convert_generation_model(
    payload: InstallModelRequest,
    converter: GenerationModelConverter = Depends(get_generation_converter),
) -> CreateConversionResponse:
    try:
        conversion_id = await converter.convert_from_hf(
            payload.repo_id,
            precision=payload.precision or "fp16",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CreateConversionResponse(
        conversion_id=conversion_id,
        status_url=f"/api/v1/generation/models/convert/{conversion_id}",
    )


@router.get("/generation/models/convert/{conversion_id}", response_model=ConversionStatusResponse)
async def get_conversion_status(
    conversion_id: str,
    converter: GenerationModelConverter = Depends(get_generation_converter),
) -> ConversionStatusResponse:
    job = converter.status(conversion_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Conversion job not found")
    progress = job.metadata.get("progress")
    return ConversionStatusResponse(
        conversion_id=job.id,
        repo_id=job.repo_id,
        status=job.status,
        progress_pct=round(progress * 100, 1) if progress is not None else None,
        stage=job.metadata.get("stage"),
        stages=job.metadata.get("stages"),
        model_id=job.model_id,
        error=job.error,
    )


@router.get("/generation/models/search", response_model=ModelSearchResponse)
async def search_generation_models(
    q: str = Query("", max_length=200),
    hf_client: HfClient = Depends(get_hf_client),
) -> ModelSearchResponse:
    # Query vacia = browse por descargas: el usuario ve modelos sin tener que
    # saber el repo_id exacto de antemano.
    try:
        results = await hf_client.search(
            q, task_tags=GENERATION_SEARCH_TASK_TAGS, sort=None if q else "downloads"
        )
    except Exception as exc:
        logger.exception("Hugging Face generation search failed for query %r", q)
        raise HTTPException(status_code=502, detail="Hugging Face search failed") from exc
    return _search_results_to_response(results, strategy_for("generate"))


@router.get("/generation/models/preflight", response_model=PreflightResponse)
async def preflight_generation_model(
    request: Request,
    repo_id: str = Query(..., alias="repoId"),
    width: int = Query(512, ge=64, le=4096),
    height: int = Query(512, ge=64, le=4096),
    hf_client: HfClient = Depends(get_hf_client),
    settings: Settings = Depends(get_settings),
) -> PreflightResponse:
    report = await preflight(
        hf_client=hf_client,
        devices_service=request.app.state.devices_service,
        settings=settings,
        probes=request.app.state.resource_probes,
        repo_id=repo_id,
        width=width,
        height=height,
    )
    return PreflightResponse(**asdict(report))

@router.get("/capabilities/tree", response_model=CapabilityTreeResponse)
async def capability_tree(
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_model_registry),
) -> CapabilityTreeResponse:
    """El arbol de lo que la app puede hacer, resuelto contra esta maquina.

    El frontend no puede mentir sobre lo que hay porque no decide: el status sale
    de mirar el disco y el registro, no de un flag persistido.
    """
    grouped = group_by_domain(resolve_capabilities(settings, registry))
    return CapabilityTreeResponse(
        domains=[
            CapabilityDomainResponse(
                domain=group.domain,
                label_key=group.label_key,
                capabilities=[_capability_to_response(item) for item in group.capabilities],
                roadmap=[_capability_to_response(item) for item in group.roadmap],
            )
            for group in grouped
        ]
    )


def _capability_to_response(item: ResolvedCapability) -> CapabilityResponse:
    return CapabilityResponse(
        id=item.id,
        domain=item.domain,
        label_key=item.label_key,
        status=item.status,
        provisioning=item.provisioning,
        job_kind=item.job_kind,
        strategies=list(item.strategies),
        missing_packs=list(item.missing_packs),
        unavailable_reason_key=item.unavailable_reason_key,
        setup_reason_key=item.setup_reason_key,
    )

def _resolved_by_id(settings: Settings, registry: ModelRegistry, capability_id: str) -> ResolvedCapability:
    for item in resolve_capabilities(settings, registry):
        if item.id == capability_id:
            return item
    raise HTTPException(status_code=404, detail=f"Capacidad desconocida: {capability_id!r}")


def _pack_to_provision(item: ResolvedCapability) -> str:
    # No se mira `provisioning`: video.upscale es de registro y aun asi necesita
    # el binario del motor. Lo que decide es si falta un paquete concreto.
    if not item.missing_packs:
        raise HTTPException(
            status_code=400,
            detail=f"La capacidad {item.id!r} no tiene ningun paquete pendiente de descarga.",
        )
    return item.missing_packs[0]


def _provision_job_to_response(job: ProvisionJob) -> ProvisionJobResponse:
    return ProvisionJobResponse(
        job_id=job.id,
        pack=job.pack,
        status=job.status.value,
        error=job.error,
        status_url=f"/api/v1/capabilities/provision/{job.id}",
    )


@router.post(
    "/capabilities/{capability_id}/provision",
    response_model=ProvisionJobResponse,
    status_code=202,
)
async def provision_capability(
    capability_id: str,
    request: Request,
    settings: Settings = Depends(get_settings),
    registry: ModelRegistry = Depends(get_model_registry),
) -> ProvisionJobResponse:
    item = _resolved_by_id(settings, registry, capability_id)
    pack = _pack_to_provision(item)
    provisioner: PackProvisioner = request.app.state.pack_provisioner
    try:
        job_id = await provisioner.provision(pack)
    except UnknownPackError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    job = provisioner.status(job_id)
    assert job is not None
    return _provision_job_to_response(job)


@router.get("/capabilities/provision/{job_id}", response_model=ProvisionJobResponse)
async def provision_status(job_id: str, request: Request) -> ProvisionJobResponse:
    provisioner: PackProvisioner = request.app.state.pack_provisioner
    job = provisioner.status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job de descarga desconocido")
    return _provision_job_to_response(job)



# Gates con los permisos que C ya define (mismo patrón que capability_routes):
# settings_read para leer, settings_write para escribir. Con AUTH_MODE=off el
# usuario off-mode tiene todos los permisos y esto es transparente.
@router.get(
    "/settings",
    response_model=EditableSettingsResponse,
    dependencies=[Depends(require(Permission.settings_read))],
)
async def get_editable_settings(
    settings: Settings = Depends(get_settings),
) -> EditableSettingsResponse:
    return EditableSettingsResponse(
        settings=[
            EditableSettingStatusResponse(**item)
            for item in editable_settings_status(settings)
        ]
    )


@router.patch(
    "/settings",
    response_model=UpdateSettingResponse,
    dependencies=[Depends(require(Permission.settings_write))],
)
async def patch_setting(payload: UpdateSettingRequest) -> UpdateSettingResponse:
    try:
        await asyncio.to_thread(update_setting, payload.key, payload.value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return UpdateSettingResponse(key=payload.key)
