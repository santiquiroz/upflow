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

from app.models import (
    CctvOptions,
    CctvStep,
    JobStatus,
    RedactionKeyframe,
    RedactionRequest,
    RedactionTrack,
    RoiFusionRequest,
    VideoUpscaleJob,
)
from app.services.cctv_artifacts import listed_artifacts
from app.services.cctv_job_validation import AiUpscaleChoice
from app.services.redaction import MAX_KEYFRAMES, MAX_TRACKS

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


class RedactionKeyframeIn(CamelRequest):
    frame: StrictInt
    box: BoxIn


class RedactionTrackIn(CamelRequest):
    first_frame: StrictInt
    last_frame: StrictInt
    keyframes: list[RedactionKeyframeIn] = Field(min_length=1, max_length=MAX_KEYFRAMES)


class CctvRedactionIn(CamelRequest):
    style: str = Field(default="fill", max_length=32)
    tracks: list[RedactionTrackIn] = Field(max_length=MAX_TRACKS)


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
    redaction: CctvRedactionIn | None = None
    acquisition: dict[str, AcquisitionValue] = Field(default_factory=dict)
    case_label: str | None = Field(default=None, max_length=MAX_TEXT)
    operator_name: str | None = Field(default=None, max_length=MAX_TEXT)
    model_id: str | None = Field(default=None, max_length=200)
    target_height: StrictInt | None = None
    scale: StrictInt | None = None
    device: str | None = Field(default=None, max_length=64)


class CctvReproduceRequest(CamelRequest):
    token: str = Field(max_length=64)
    # Input no confiable: lo valida `parse_untrusted_report` para devolver una clave y no un 422.
    report: dict[str, Any]


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


class CctvReproduceStartResponse(CamelModel):
    job_id: str
    status_url: str
    result_url: str
    warnings: list[str]


class ReproduceCheckResponse(CamelModel):
    kind: str
    name: str
    expected: str
    actual: str | None
    match: bool


class CctvReproduceResultResponse(CamelModel):
    job_id: str
    identical: bool
    frames_identical: bool | None
    checks: list[ReproduceCheckResponse]
    notes: list[str]


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
    # Solo en la copia anonimizada: estilo y cantidad de cajas (metadata.cctv.redaction).
    redaction: dict[str, Any] | None = None


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


def redaction_track(track: RedactionTrackIn) -> RedactionTrack:
    keyframes = tuple(RedactionKeyframe(key.frame, tuple(key.box)) for key in track.keyframes)
    return RedactionTrack(track.first_frame, track.last_frame, keyframes)


def redaction_request(redaction: CctvRedactionIn | None) -> RedactionRequest | None:
    if redaction is None:
        return None
    return RedactionRequest(tuple(redaction_track(track) for track in redaction.tracks), redaction.style)


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
        redaction=redaction_request(request.redaction),
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


def dict_fact(meta: dict[str, Any], key: str) -> dict[str, Any] | None:
    value = meta.get(key)
    return value if isinstance(value, dict) else None


def roi_facts(meta: dict[str, Any]) -> dict[str, Any] | None:
    return dict_fact(meta, "roi")


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
        redaction=dict_fact(meta, "redaction"),
    )
