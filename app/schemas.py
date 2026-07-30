from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models import JobStatus
from app.services.capability_probe import LeverStatus
from app.services.generation_variants import Precision


class CreateJobResponse(BaseModel):
    job_id: str = Field(serialization_alias="jobId")
    status: JobStatus
    status_url: str = Field(serialization_alias="statusUrl")
    download_url: str | None = Field(default=None, serialization_alias="downloadUrl")


class JobResponse(BaseModel):
    job_id: str = Field(serialization_alias="jobId")
    status: JobStatus
    original_filename: str = Field(serialization_alias="originalFilename")
    model_name: str = Field(serialization_alias="modelName")
    scale: int
    output_format: str = Field(serialization_alias="outputFormat")
    model_id: str | None = Field(default=None, serialization_alias="modelId")
    device: str | None = None
    created_at: datetime = Field(serialization_alias="createdAt")
    started_at: datetime | None = Field(default=None, serialization_alias="startedAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")
    error: str | None = None
    owner_id: str | None = Field(default=None, serialization_alias="ownerId")
    metadata: dict[str, Any] = Field(default_factory=dict)
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    download_url: str | None = Field(default=None, serialization_alias="downloadUrl")


class VideoJobResponse(BaseModel):
    job_id: str = Field(serialization_alias="jobId")
    status: JobStatus
    original_filename: str = Field(serialization_alias="originalFilename")
    model_name: str = Field(serialization_alias="modelName")
    scale: int
    output_container: str = Field(serialization_alias="outputContainer")
    video_codec: str = Field(serialization_alias="videoCodec")
    video_preset: str = Field(serialization_alias="videoPreset")
    crf: int
    keep_audio: bool = Field(serialization_alias="keepAudio")
    fps_multiplier: int = Field(serialization_alias="fpsMultiplier")
    target_fps: str | None = Field(default=None, serialization_alias="targetFps")
    audio_enhance: str | None = Field(default=None, serialization_alias="audioEnhance")
    audio_restore: str | None = Field(default=None, serialization_alias="audioRestore")
    audio_track_indices: list[int] | None = Field(default=None, serialization_alias="audioTrackIndices")
    keep_subtitles: bool = Field(default=False, serialization_alias="keepSubtitles")
    audio_output_format: str = Field(default="auto", serialization_alias="audioOutputFormat")
    interp_engine: str = Field(default="rife", serialization_alias="interpEngine")
    model_id: str | None = Field(default=None, serialization_alias="modelId")
    device: str | None = None
    backend: str | None = None
    video_encoder: str = Field(default="auto", serialization_alias="videoEncoder")
    created_at: datetime = Field(serialization_alias="createdAt")
    started_at: datetime | None = Field(default=None, serialization_alias="startedAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")
    error: str | None = None
    owner_id: str | None = Field(default=None, serialization_alias="ownerId")
    metadata: dict[str, Any] = Field(default_factory=dict)
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    download_url: str | None = Field(default=None, serialization_alias="downloadUrl")


class AudioJobResponse(BaseModel):
    id: str
    status: JobStatus
    original_filename: str = Field(serialization_alias="originalFilename")
    denoise: str | None = None
    restore: str | None = None
    device: str | None = None
    output_format: str = Field(default="flac", serialization_alias="outputFormat")
    created_at: datetime = Field(serialization_alias="createdAt")
    started_at: datetime | None = Field(default=None, serialization_alias="startedAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    stages: list[dict[str, Any]] | None = None
    error: str | None = None
    owner_id: str | None = Field(default=None, serialization_alias="ownerId")
    download_url: str | None = Field(default=None, serialization_alias="downloadUrl")


class AudioCapabilitiesResponse(BaseModel):
    denoise_modes: list[str] = Field(serialization_alias="denoiseModes")
    restore_available: bool = Field(serialization_alias="restoreAvailable")
    restore_modes: list[str] = Field(default_factory=list, serialization_alias="restoreModes")


class VideoCapabilitiesResponse(BaseModel):
    interp_engines: list[str] = Field(default_factory=list, serialization_alias="interpEngines")


class LeverResponse(BaseModel):
    id: str
    label: str
    status: LeverStatus
    detail: str
    fixable: bool


class CapabilitiesResponse(BaseModel):
    levers: list[LeverResponse]


class FixLeverResponse(BaseModel):
    lever: LeverResponse


class AudioTrackResponse(BaseModel):
    index: int
    codec: str
    channels: int
    is_default: bool = Field(serialization_alias="isDefault")
    language: str | None = None


class SubtitleTrackResponse(BaseModel):
    index: int
    codec: str
    language: str | None = None


class AnalyzeVideoResponse(BaseModel):
    upload_token: str = Field(serialization_alias="uploadToken")
    audio_tracks: list[AudioTrackResponse] = Field(serialization_alias="audioTracks")
    subtitle_tracks: list[SubtitleTrackResponse] = Field(serialization_alias="subtitleTracks")


class SupportedModelResponse(BaseModel):
    key: str
    label: str
    category: str
    description: str
    scales: list[int]


class VideoProfileResponse(BaseModel):
    key: str
    label: str
    category: str
    description: str
    model_key: str = Field(serialization_alias="modelKey")
    scale: int
    video_codec: str = Field(serialization_alias="videoCodec")
    video_preset: str = Field(serialization_alias="videoPreset")
    crf: int
    keep_audio: bool = Field(serialization_alias="keepAudio")


class EngineInfoResponse(BaseModel):
    engine: str
    configured_binary: str = Field(serialization_alias="configuredBinary")
    configured_models_dir: str = Field(serialization_alias="configuredModelsDir")
    available: bool
    default_model: str = Field(serialization_alias="defaultModel")
    allowed_scales: list[int] = Field(serialization_alias="allowedScales")
    supported_models: list[SupportedModelResponse] = Field(serialization_alias="supportedModels")
    video_profiles: list[VideoProfileResponse] = Field(serialization_alias="videoProfiles")
    ffmpeg_available: bool = Field(serialization_alias="ffmpegAvailable")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    engine: str
    gpu_concurrency: int = Field(serialization_alias="gpuConcurrency")
    queue_depth: int = Field(serialization_alias="queueDepth")
    video_queue_depth: int = Field(serialization_alias="videoQueueDepth")


class DeviceInfoResponse(BaseModel):
    id: str
    kind: Literal["cpu", "gpu", "npu"]
    name: str
    backend: Literal["cpu", "directml", "winml"]


class DevicesResponse(BaseModel):
    devices: list[DeviceInfoResponse]
    default_device_id: str = Field(serialization_alias="defaultDeviceId")


class ModelResponse(BaseModel):
    id: str
    name: str
    kind: str
    source: str
    scale: int | None = None
    arch: str | None = None
    size_bytes: int = Field(serialization_alias="sizeBytes")
    status: str
    error: str | None = None


class ModelsResponse(BaseModel):
    models: list[ModelResponse]


class HfModelSearchResultResponse(BaseModel):
    id: str
    author: str | None = None
    pipeline_tag: str | None = Field(default=None, serialization_alias="pipelineTag")
    downloads: int
    likes: int
    tags: list[str]
    # Compatibilidad DETECTADA de la metadata en vivo (siblings + gated, que ya
    # vienen en la respuesta de busqueda con full=true): cero requests extra por
    # resultado. Los DOS caminos la calculan, cada uno con su CompatStrategy;
    # available_precisions queda vacio en upscalers porque su instalador elige
    # el archivo de pesos solo.
    compat: str | None = None
    compat_reason_key: str | None = Field(
        default=None, serialization_alias="compatReasonKey"
    )
    compat_reason_params: dict[str, str] = Field(
        default_factory=dict, serialization_alias="compatReasonParams"
    )
    available_precisions: list[str] = Field(
        default_factory=list, serialization_alias="availablePrecisions"
    )


class ModelSearchResponse(BaseModel):
    results: list[HfModelSearchResultResponse]


class PrecisionCostResponse(BaseModel):
    precision: str
    download_bytes: int = Field(serialization_alias="downloadBytes")
    estimated_peak_bytes: int = Field(serialization_alias="estimatedPeakBytes")


class DeviceCapacityResponse(BaseModel):
    id: str
    name: str
    kind: str
    # null = no se pudo medir. Nunca 0: el frontend no avisa de lo que no sabe.
    free_vram_bytes: int | None = Field(default=None, serialization_alias="freeVramBytes")


class DiskCapacityResponse(BaseModel):
    target_path: str = Field(serialization_alias="targetPath")
    free_bytes: int = Field(serialization_alias="freeBytes")


class CheckpointCandidateResponse(BaseModel):
    path: str
    size_bytes: int = Field(serialization_alias="sizeBytes")
    architecture: str | None = None
    installable: bool | None = None
    reason_key: str = Field(serialization_alias="reasonKey")
    reason_params: dict[str, str] = Field(
        default_factory=dict, serialization_alias="reasonParams"
    )


class PreflightResponse(BaseModel):
    repo_id: str = Field(serialization_alias="repoId")
    compat: str | None = None
    compat_reason_key: str | None = Field(
        default=None, serialization_alias="compatReasonKey"
    )
    compat_reason_params: dict[str, str] = Field(
        default_factory=dict, serialization_alias="compatReasonParams"
    )
    degraded: bool
    reference_width: int = Field(serialization_alias="referenceWidth")
    reference_height: int = Field(serialization_alias="referenceHeight")
    precisions: list[PrecisionCostResponse]
    devices: list[DeviceCapacityResponse]
    disk: DiskCapacityResponse | None = None
    checkpoints: list[CheckpointCandidateResponse] = Field(default_factory=list)
    free_ram_bytes: int | None = Field(
        default=None,
        serialization_alias="freeRamBytes",
    )


class VoiceStepResponse(BaseModel):
    id: str
    label_key: str = Field(serialization_alias="labelKey")
    # Descripcion en lenguaje llano. SIEMPRE visible en la UI, no solo en un
    # tooltip: usar hover como unico mecanismo para informacion critica es un
    # fallo de accesibilidad conocido.
    description_key: str = Field(serialization_alias="descriptionKey")
    kind: str
    default_enabled: bool = Field(serialization_alias="defaultEnabled")


class VoiceDeliveryResponse(BaseModel):
    id: str
    label_key: str = Field(serialization_alias="labelKey")
    description_key: str = Field(serialization_alias="descriptionKey")
    lufs: float
    true_peak_db: float = Field(serialization_alias="truePeakDb")


class VoiceCatalogResponse(BaseModel):
    steps: list[VoiceStepResponse]
    deliveries: list[VoiceDeliveryResponse]


class CapabilityResponse(BaseModel):
    id: str
    domain: str
    label_key: str = Field(serialization_alias="labelKey")
    status: str
    provisioning: str
    job_kind: str | None = Field(default=None, serialization_alias="jobKind")
    strategies: list[str] = Field(default_factory=list)
    missing_packs: list[str] = Field(default_factory=list, serialization_alias="missingPacks")
    unavailable_reason_key: str | None = Field(
        default=None, serialization_alias="unavailableReasonKey"
    )
    setup_reason_key: str | None = Field(default=None, serialization_alias="setupReasonKey")


class CapabilityDomainResponse(BaseModel):
    domain: str
    label_key: str = Field(serialization_alias="labelKey")
    capabilities: list[CapabilityResponse] = Field(default_factory=list)
    # Las no implementadas viajan separadas para que el frontend les pueda dar el
    # encabezado de mapa de ruta sin tener que filtrar por su cuenta.
    roadmap: list[CapabilityResponse] = Field(default_factory=list)


class CapabilityTreeResponse(BaseModel):
    domains: list[CapabilityDomainResponse] = Field(default_factory=list)


class UpscalerPreflightResponse(BaseModel):
    repo_id: str = Field(serialization_alias="repoId")
    compat: str | None = None
    compat_reason_key: str | None = Field(
        default=None, serialization_alias="compatReasonKey"
    )
    compat_reason_params: dict[str, str] = Field(
        default_factory=dict, serialization_alias="compatReasonParams"
    )
    degraded: bool
    # Peso del archivo que el instalador va a bajar. None si el repo no tiene
    # ninguno, que la clasificacion ya reporta como incompatible.
    download_bytes: int | None = Field(default=None, serialization_alias="downloadBytes")
    devices: list[DeviceCapacityResponse] = Field(default_factory=list)
    disk: DiskCapacityResponse | None = None
    free_ram_bytes: int | None = Field(default=None, serialization_alias="freeRamBytes")


class TranscribeJobResponse(BaseModel):
    id: str
    status: JobStatus
    original_filename: str = Field(serialization_alias="originalFilename")
    model_id: str = Field(serialization_alias="modelId")
    language: str | None = None
    device: str | None = None
    created_at: datetime = Field(serialization_alias="createdAt")
    started_at: datetime | None = Field(default=None, serialization_alias="startedAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    # El TEXTO es el resultado, a diferencia del resto de los jobs. Viaja en la
    # respuesta para que la UI no tenga que descargar un archivo para mostrarlo.
    text: str | None = None
    error: str | None = None
    owner_id: str | None = Field(default=None, serialization_alias="ownerId")
    download_url: str | None = Field(default=None, serialization_alias="downloadUrl")


class TranscribeJobsListResponse(BaseModel):
    jobs: list[TranscribeJobResponse] = Field(default_factory=list)


class ProvisionJobResponse(BaseModel):
    job_id: str = Field(serialization_alias="jobId")
    pack: str
    status: str
    error: str | None = None
    status_url: str = Field(serialization_alias="statusUrl")


class InstallModelRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    repo_id: str = Field(alias="repoId")
    # Solo la usa el camino de generacion; el de upscalers la ignora.
    precision: Precision | None = None
    checkpoint_path: str | None = Field(
        default=None,
        alias="checkpointPath",
        serialization_alias="checkpointPath",
    )


class CreateInstallResponse(BaseModel):
    install_id: str = Field(serialization_alias="installId")
    status_url: str = Field(serialization_alias="statusUrl")


class CreateConversionResponse(BaseModel):
    conversion_id: str = Field(serialization_alias="conversionId")
    status_url: str = Field(serialization_alias="statusUrl")


class UpdateCheckResponse(BaseModel):
    current_version: str = Field(serialization_alias="currentVersion")
    latest_version: str | None = Field(default=None, serialization_alias="latestVersion")
    update_available: bool = Field(serialization_alias="updateAvailable")
    release_url: str | None = Field(default=None, serialization_alias="releaseUrl")
    published_at: str | None = Field(default=None, serialization_alias="publishedAt")
    checked_at: datetime = Field(serialization_alias="checkedAt")
    error: str | None = None


class InstallStatusResponse(BaseModel):
    install_id: str = Field(serialization_alias="installId")
    repo_id: str = Field(serialization_alias="repoId")
    status: str
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    model_id: str | None = Field(default=None, serialization_alias="modelId")
    error: str | None = None
    conversion_id: str | None = Field(default=None, serialization_alias="conversionId")


class ConversionStatusResponse(BaseModel):
    conversion_id: str = Field(serialization_alias="conversionId")
    repo_id: str = Field(serialization_alias="repoId")
    status: JobStatus
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    stage: str | None = None
    stages: list[dict[str, Any]] | None = None
    model_id: str | None = Field(default=None, serialization_alias="modelId")
    error: str | None = None


class CreateGenerationJobRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    prompt: str = Field(min_length=1, max_length=2000)
    negative_prompt: str | None = Field(default=None, alias="negativePrompt", max_length=2000)
    model_id: str = Field(alias="modelId")
    steps: int = Field(default=25, ge=1, le=100)
    guidance: float = Field(default=7.5, ge=0, le=30)
    width: int = Field(default=512, ge=64, le=1024, multiple_of=64)
    height: int = Field(default=512, ge=64, le=1024, multiple_of=64)
    seed: int | None = Field(default=None, ge=0)
    device: str | None = None
    # Token de una imagen ya subida con POST /generation/init-image. Presente =
    # imagen a imagen. Se sube aparte para no volver multipart el contrato JSON
    # de este endpoint, igual que hace el flujo de video con /video/analyze.
    init_image_token: str | None = Field(default=None, alias="initImageToken")
    strength: float = Field(default=0.6, gt=0, le=1)
    auto_upscale: bool = Field(default=False, alias="autoUpscale")
    upscale_model_name: str | None = Field(default=None, alias="upscaleModelName")
    upscale_scale: int | None = Field(default=None, alias="upscaleScale", ge=2, le=4)
    upscale_model_id: str | None = Field(default=None, alias="upscaleModelId")


class InitImageResponse(BaseModel):
    init_image_token: str = Field(serialization_alias="initImageToken")
    original_filename: str = Field(serialization_alias="originalFilename")
    width: int
    height: int


class GenerationJobResponse(BaseModel):
    id: str
    status: JobStatus
    prompt: str
    negative_prompt: str | None = Field(default=None, serialization_alias="negativePrompt")
    model_id: str = Field(serialization_alias="modelId")
    steps: int
    guidance: float
    width: int
    height: int
    seed: int | None = None
    device: str | None = None
    auto_upscale: bool = Field(default=False, serialization_alias="autoUpscale")
    created_at: datetime = Field(serialization_alias="createdAt")
    started_at: datetime | None = Field(default=None, serialization_alias="startedAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    stages: list[dict[str, Any]] | None = None
    error: str | None = None
    owner_id: str | None = Field(default=None, serialization_alias="ownerId")
    download_url: str | None = Field(default=None, serialization_alias="downloadUrl")


class GenerationModelSummary(BaseModel):
    id: str
    name: str


class GenerationCapabilitiesResponse(BaseModel):
    available: bool
    reason: str | None = None
    models: list[GenerationModelSummary] = Field(default_factory=list)
    devices: list[str] = Field(default_factory=list)
    cpu_only: bool = Field(default=False, serialization_alias="cpuOnly")


class CpuFallbackReportResponse(BaseModel):
    model_id: str = Field(serialization_alias="modelId")
    device_id: str = Field(serialization_alias="deviceId")
    hot_ops: list[str] = Field(serialization_alias="hotOps")
    clean: bool


class OnnxDiagnosticEntryResponse(BaseModel):
    model_id: str = Field(serialization_alias="modelId")
    device_id: str = Field(serialization_alias="deviceId")
    report: CpuFallbackReportResponse | None = None


class OnnxDiagnosticsResponse(BaseModel):
    entries: list[OnnxDiagnosticEntryResponse]


class ScanOnnxDiagnosticResponse(BaseModel):
    report: CpuFallbackReportResponse


class LoginRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    current_password: str = Field(alias="currentPassword")
    new_password: str = Field(alias="newPassword", min_length=8)


class SetupRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    username: str = Field(min_length=3)
    password: str = Field(min_length=8)


class QuotaStatusResponse(BaseModel):
    max_concurrent: int = Field(serialization_alias="maxConcurrent")
    max_queued: int = Field(serialization_alias="maxQueued")
    max_jobs_per_day: int = Field(serialization_alias="maxJobsPerDay")
    max_gpu_seconds_per_day: int = Field(serialization_alias="maxGpuSecondsPerDay")
    used_jobs_today: int = Field(serialization_alias="usedJobsToday")
    used_gpu_seconds_today: float = Field(serialization_alias="usedGpuSecondsToday")


class MeResponse(BaseModel):
    user_id: str | None = Field(serialization_alias="userId")
    username: str
    role: str
    permissions: list[str]
    must_change_password: bool = Field(serialization_alias="mustChangePassword")
    auth_mode: str = Field(serialization_alias="authMode")
    quota: QuotaStatusResponse


class UserSummaryResponse(BaseModel):
    id: str
    username: str
    role: str
    disabled: bool
    must_change_password: bool = Field(serialization_alias="mustChangePassword")
    quota_overrides: dict[str, int] = Field(default_factory=dict, serialization_alias="quotaOverrides")
    created_at: datetime = Field(serialization_alias="createdAt")
    used_jobs_today: int = Field(serialization_alias="usedJobsToday")
    used_gpu_seconds_today: float = Field(serialization_alias="usedGpuSecondsToday")


class UsersListResponse(BaseModel):
    users: list[UserSummaryResponse]


class CreateUserRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    username: str = Field(min_length=3)
    role: str = Field(default="user")


class CreateUserResponse(BaseModel):
    user: UserSummaryResponse
    temporary_password: str = Field(serialization_alias="temporaryPassword")


class UpdateUserRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    role: str | None = None
    disabled: bool | None = None
    quota_overrides: dict[str, int] | None = Field(default=None, alias="quotaOverrides")
    reset_password: bool = Field(default=False, alias="resetPassword")


class UpdateUserResponse(BaseModel):
    user: UserSummaryResponse
    temporary_password: str | None = Field(default=None, serialization_alias="temporaryPassword")


class OwnedJobSummaryResponse(BaseModel):
    id: str
    kind: str
    status: JobStatus
    original_filename: str | None = Field(default=None, serialization_alias="originalFilename")
    created_at: datetime = Field(serialization_alias="createdAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")


class UserJobsResponse(BaseModel):
    jobs: list[OwnedJobSummaryResponse]


class JobsListResponse(BaseModel):
    jobs: list[JobResponse]


class VideoJobsListResponse(BaseModel):
    jobs: list[VideoJobResponse]


class AudioJobsListResponse(BaseModel):
    jobs: list[AudioJobResponse]


class GenerationJobsListResponse(BaseModel):
    jobs: list[GenerationJobResponse]


class EditableSettingStatusResponse(BaseModel):
    key: str
    configured: bool


class EditableSettingsResponse(BaseModel):
    settings: list[EditableSettingStatusResponse]


class UpdateSettingRequest(BaseModel):
    key: str = Field(min_length=1)
    value: str


class UpdateSettingResponse(BaseModel):
    key: str
