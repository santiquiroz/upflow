"""Modelos pydantic de las rutas CCTV (spec §5.5).

Todo viaja en camelCase. Los campos que el backend valida con claves propias
(`cctv.error.*`: tarea, preset, pasos, ROI) llegan como texto libre y los valida
`VideoJobManager.create_cctv_job`, para que el cliente reciba la clave y no un 422.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt
from pydantic.alias_generators import to_camel

from app.models import CctvOptions, CctvStep, JobStatus, RoiFusionRequest, VideoUpscaleJob
from app.services.cctv_artifacts import listed_artifacts
from app.services.cctv_job_validation import AiUpscaleChoice

MAX_TEXT = 2000
MAX_STEPS = 32
MAX_BOXES = 32
MAX_STILLS = 256
ENHANCE_ONLY = "cctv.error.enhanceOnly"
TARGET_HEIGHT_UNSUPPORTED = "cctv.error.targetHeightUnsupported"

BoxIn = tuple[StrictInt, StrictInt, StrictInt, StrictInt]
AcquisitionValue = str | StrictInt | float


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class CamelRequest(CamelModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


# --- Pedidos ---


class CctvStepIn(CamelRequest):
    id: str = Field(max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)


class CctvRoiIn(CamelRequest):
    first_frame: StrictInt
    last_frame: StrictInt
    reference_frame: StrictInt
    box: BoxIn
    kind: str = Field(max_length=32)
    scale: StrictInt = 2
    method: str = Field(default="median", max_length=32)


class CctvJobRequest(CamelRequest):
    token: str = Field(max_length=64)
    task: str = Field(max_length=32)
    preset: str | None = Field(default=None, max_length=32)
    steps: list[CctvStepIn] = Field(default_factory=list, max_length=MAX_STEPS)
    osd_boxes: list[BoxIn] = Field(default_factory=list, max_length=MAX_BOXES)
    osd_boxes_confirmed: StrictBool = False
    no_osd: StrictBool = False
    trim: tuple[StrictInt, StrictInt] | None = None
    still_frames: list[StrictInt] = Field(default_factory=list, max_length=MAX_STILLS)
    roi: CctvRoiIn | None = None
    acquisition: dict[str, AcquisitionValue] = Field(default_factory=dict)
    case_label: str | None = Field(default=None, max_length=MAX_TEXT)
    operator_name: str | None = Field(default=None, max_length=MAX_TEXT)
    model_id: str | None = Field(default=None, max_length=200)
    target_height: StrictInt | None = None
    scale: StrictInt | None = None
    device: str | None = Field(default=None, max_length=64)


class RoiReferenceRequest(CamelRequest):
    first_frame: StrictInt
    last_frame: StrictInt
    box: BoxIn
    steps: list[CctvStepIn] = Field(default_factory=list, max_length=MAX_STEPS)


class RoiReferenceResponse(CamelModel):
    reference_frame: int


class OsdCheckRequest(CamelRequest):
    boxes: list[BoxIn] = Field(min_length=1, max_length=MAX_BOXES)
    frame: StrictInt | None = None


# --- Respuestas ---


class CctvAnalysisResponse(CamelModel):
    token: str
    source_sha256: str
    received_at: dict[str, str]
    original_name: str
    size_bytes: int
    container: dict[str, str]
    working_copy: dict[str, Any]
    video: dict[str, Any]
    audio: list[dict[str, Any]]
    frame_index: dict[str, Any] | None
    gop: dict[str, Any] | None
    quality: dict[str, Any] | None
    decode_failed: bool
    suggested_preset: str | None
    proposed_steps: dict[str, list[dict[str, Any]]] | None
    mode_available: bool
    mode_unavailable_reason: str | None
    unavailable_steps: list[str]
    unavailable_filters: list[dict[str, Any]]
    warnings: list[str]


AnalysisStatus = Literal["running", "completed", "failed"]


class CctvAnalysisJobResponse(CamelModel):
    analysis_job_id: str
    status: AnalysisStatus
    status_url: str
    result: CctvAnalysisResponse | None = None
    error: str | None = None
    error_key: str | None = None


class OsdBoxCheckResponse(CamelModel):
    box: list[int]
    frames: int
    contrast: float
    static_fraction: float | None
    looks_like_text: bool
    warning_key: str | None


class OsdCheckResponse(CamelModel):
    checks: list[OsdBoxCheckResponse]
    warnings: list[str]


class CctvPresetsResponse(CamelModel):
    mode_available: bool
    mode_unavailable_reason: str | None
    ffmpeg: dict[str, Any]
    presets: list[dict[str, Any]]
    steps: dict[str, list[dict[str, Any]]]
    unavailable_steps: list[str]
    unavailable_filters: list[dict[str, Any]]
    ai_upscale_models: list[dict[str, Any]] = Field(default_factory=list)


class VerifyFilesResponse(CamelModel):
    ok: bool
    checked: int
    mismatches: list[str]
    missing: list[str]


class CctvArtifactLink(CamelModel):
    name: str
    url: str


class CctvSummary(CamelModel):
    task: str
    lane: str | None
    preset: str | None
    source_sha256: str | None
    no_osd: bool
    osd_boxes_confirmed: bool
    warnings: list[str]
    artifacts: list[CctvArtifactLink]
    verify_url: str | None
    # Solo en la foto multi-cuadro: muestras efectivas, densidad y avisos (metadata.cctv.roi).
    roi: dict[str, Any] | None = None


# --- Conversion a los modelos del dominio ---


def enhance_only_fields(request: CctvJobRequest) -> list[str]:
    fields = {"modelId": request.model_id, "targetHeight": request.target_height, "scale": request.scale}
    return [name for name, value in fields.items() if value is not None]


def ai_upscale_choice(request: CctvJobRequest) -> AiUpscaleChoice:
    return AiUpscaleChoice(request.model_id, request.scale)


def cctv_step(step: CctvStepIn) -> CctvStep:
    return CctvStep(step.id, MappingProxyType(dict(step.params)))


def roi_request(roi: CctvRoiIn | None) -> RoiFusionRequest | None:
    if roi is None:
        return None
    return RoiFusionRequest(
        first_frame=roi.first_frame,
        last_frame=roi.last_frame,
        reference_frame=roi.reference_frame,
        box=tuple(roi.box),
        kind=roi.kind,
        scale=roi.scale,
        method=roi.method,
    )


def cctv_options(request: CctvJobRequest) -> CctvOptions:
    return CctvOptions(
        task=request.task,
        session_token=request.token,
        preset=request.preset,
        steps=tuple(cctv_step(step) for step in request.steps),
        osd_boxes=tuple(tuple(box) for box in request.osd_boxes),
        osd_boxes_confirmed=request.osd_boxes_confirmed,
        no_osd=request.no_osd,
        trim=None if request.trim is None else tuple(request.trim),
        still_frames=tuple(request.still_frames),
        roi=roi_request(request.roi),
        acquisition=MappingProxyType(dict(request.acquisition)),
        case_label=request.case_label,
        operator_name=request.operator_name,
    )


# --- Resumen del job para VideoJobResponse.cctv ---


def artifact_url(job_id: str, name: str) -> str:
    return f"/api/v1/video/jobs/{job_id}/artifacts/{name}"


def verify_url(job: VideoUpscaleJob) -> str | None:
    return f"/api/v1/video/jobs/{job.id}/verify" if job.status == JobStatus.completed else None


def cctv_metadata(job: VideoUpscaleJob) -> dict[str, Any]:
    meta = job.metadata.get("cctv")
    return meta if isinstance(meta, dict) else {}


def roi_facts(meta: dict[str, Any]) -> dict[str, Any] | None:
    roi = meta.get("roi")
    return roi if isinstance(roi, dict) else None


def cctv_summary(job: VideoUpscaleJob) -> CctvSummary | None:
    if job.cctv is None:
        return None
    meta = cctv_metadata(job)
    return CctvSummary(
        task=job.cctv.task,
        lane=meta.get("lane"),
        preset=job.cctv.preset,
        source_sha256=meta.get("sourceSha256"),
        no_osd=job.cctv.no_osd,
        osd_boxes_confirmed=job.cctv.osd_boxes_confirmed,
        warnings=[str(key) for key in meta.get("warnings", [])],
        artifacts=[CctvArtifactLink(name=name, url=artifact_url(job.id, name)) for name in listed_artifacts(job)],
        verify_url=verify_url(job),
        roi=roi_facts(meta),
    )
