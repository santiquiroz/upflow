"""Modelo del informe CCTV, `report.json` con `schemaVersion: 1` (spec §4.11).

El JSON Schema publicado en `docs/schemas/cctv-report-v1.schema.json` se genera
desde aca (`report_schema_text`) y un test lo compara con el archivo: si el
modelo cambia y el esquema no, el test falla.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, model_validator
from pydantic.alias_generators import to_camel

SCHEMA_VERSION = 1
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "docs" / "schemas" / "cctv-report-v1.schema.json"

MAX_USER_TEXT = 2000
MAX_SHORT_TEXT = 200

ReportMode = Literal["classic", "ai-visual"]
StepCategory = Literal["classic", "ai", "label"]
OutputRole = Literal[
    "analysis-lossless",
    "viewing-copy",
    "comparison",
    "still",
    "roi-fused",
    "roi-reference",
    "roi-agreement",
    "roi-stack",
    "ai-visualization",
]

SHA256_PATTERN = r"^[0-9a-f]{64}$"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


def check_sha256(value: str) -> str:
    if not _SHA256.fullmatch(value):
        raise ValueError("must be 64 lowercase hex characters")
    return value


def check_relative_path(value: str) -> str:
    parts = value.split("/")
    unsafe = (
        "\\" in value
        or ":" in value
        or _CONTROL.search(value) is not None
        or any(part in ("", ".", "..") for part in parts)
    )
    if unsafe:
        raise ValueError("must be a relative path with forward slashes and no '..'")
    return value


def check_https_url(value: str) -> str:
    if not value.startswith("https://") or _CONTROL.search(value) is not None:
        raise ValueError("must be an https:// URL")
    return value


Sha256 = Annotated[str, StringConstraints(pattern=SHA256_PATTERN)]
RelativePath = Annotated[str, AfterValidator(check_relative_path)]
HttpsUrl = Annotated[str, AfterValidator(check_https_url)]
UserText = Annotated[str, Field(max_length=MAX_USER_TEXT)]
ShortText = Annotated[str, Field(max_length=MAX_SHORT_TEXT)]


class ReportModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", alias_generator=to_camel, populate_by_name=True)


class Timestamp(ReportModel):
    utc: str
    local: str


class CaseInfo(ReportModel):
    case_label: ShortText | None = None
    operator_name: ShortText | None = None
    notes: UserText | None = None


class UpflowInfo(ReportModel):
    version: str
    commit: str | None = None


class CpuInfo(ReportModel):
    model: str
    extensions: list[str]


class GpuInfo(ReportModel):
    name: str
    driver_version: str | None = None


class OnnxRuntimeInfo(ReportModel):
    version: str
    providers: list[str]


class FfmpegBuild(ReportModel):
    version: str
    configuration: list[str]
    gpl: bool


class EnvironmentInfo(ReportModel):
    os: str
    cpu: CpuInfo
    gpus: list[GpuInfo]
    onnx_runtime: OnnxRuntimeInfo | None = None
    ffmpeg: FfmpegBuild
    ffmpeg_sha256: Sha256
    python: str


class ContainerInfo(ReportModel):
    kind: str
    label: str


class RemuxAttemptInfo(ReportModel):
    method: str
    input_options: list[str]
    ok: bool
    detail: str


class WorkingCopyInfo(ReportModel):
    sha256: Sha256
    method: str
    copytb: bool
    attempts: list[RemuxAttemptInfo]
    note: str


class FrameIndexInfo(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow", alias_generator=to_camel, populate_by_name=True)

    csv_sha256: Sha256


class InputFile(ReportModel):
    name: str
    size_bytes: int = Field(ge=0)
    mtime: str
    sha256: Sha256
    received_at: Timestamp
    verified_copy_path: RelativePath
    verified_copy_sha256: Sha256
    container: ContainerInfo
    working_copy: WorkingCopyInfo
    ffprobe: dict[str, Any]
    frame_index: FrameIndexInfo | None
    video: dict[str, Any]
    audio: list[dict[str, Any]]
    decode: dict[str, Any]
    lite: dict[str, Any] | None = None
    diagnosis: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _verified_copy_matches(self) -> InputFile:
        if self.verified_copy_sha256 != self.sha256:
            raise ValueError("verifiedCopySha256 must equal the sha256 computed when the file was received")
        return self


class Acquisition(ReportModel):
    recorder_make: ShortText | None = None
    recorder_model: ShortText | None = None
    channel: ShortText | None = None
    clock_offset_seconds: float | None = Field(default=None, allow_inf_nan=False)
    clock_offset_method: UserText | None = None


class OsdCheckInfo(ReportModel):
    box: list[int] = Field(min_length=4, max_length=4)
    looks_like_text: bool
    contrast: float | None
    static_fraction: float | None
    frames_checked: int = Field(ge=0)


class OsdInfo(ReportModel):
    boxes: list[Annotated[list[int], Field(min_length=4, max_length=4)]]
    osd_boxes_confirmed: bool
    no_osd: bool
    checks: list[OsdCheckInfo]
    temporal_filters: list[str]
    warnings: list[str]


class TrimInfo(ReportModel):
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    audio_start_seconds: float | None
    audio_end_seconds: float | None


class EngineInfo(ReportModel):
    name: str
    version: str | None = None
    model_name: str | None = None
    model_sha256: Sha256 | None = None
    model_license: str | None = None


class ReportStep(ReportModel):
    index: int = Field(ge=0)
    id: str
    category: StepCategory
    label: str
    filter: str
    description: str
    doc_url: HttpsUrl | None
    parameters: dict[str, Any]
    engine: EngineInfo
    argv: list[str]
    started_at: Timestamp
    ended_at: Timestamp
    exit_code: int
    frames_in: int | None
    frames_out: int | None
    warnings: list[str] = Field(default_factory=list)


class ProcessInfo(ReportModel):
    label: str
    argv: list[str]
    started_at: Timestamp
    ended_at: Timestamp
    exit_code: int
    frames_in: int | None
    frames_out: int | None


class OutputFile(ReportModel):
    path: RelativePath
    role: OutputRole
    sha256: Sha256
    ffprobe: dict[str, Any] | None = None
    ai_applied: bool
    visible_label: str | None = None


class StillFrameInfo(ReportModel):
    role: Literal["original", "processed"]
    frame: int = Field(ge=0)
    file: RelativePath
    pts_time: float
    timecode: str
    framehash_sha256: Sha256


class StillPairInfo(ReportModel):
    frame: int = Field(ge=0)
    original: StillFrameInfo
    processed: StillFrameInfo
    approximate: bool
    note: str | None = None


class ClippingInfo(ReportModel):
    before_pct: float | None
    after_pct: float | None
    increase_pct: float | None
    increased: bool


class FrameHashesInfo(ReportModel):
    decoder: str
    original: list[Sha256]
    processed: list[Sha256]


class LimitationInfo(ReportModel):
    key: str
    text: str


class RoiSampleInfo(ReportModel):
    frame: int = Field(ge=0)
    pict_type: str
    copy_group: int = Field(ge=0)
    status: Literal["reference", "accepted", "copy", "rejected"]
    ecc: float | None
    shift: list[float] | None = Field(min_length=2, max_length=2)
    matrix: list[Annotated[list[float], Field(min_length=3, max_length=3)]] | None = Field(min_length=3, max_length=3)


class RoiNoticeInfo(ReportModel):
    key: str
    params: dict[str, Any]


class RoiFusionInfo(ReportModel):
    kind: Literal["plate", "face_or_object"]
    box: list[int] = Field(min_length=4, max_length=4)
    first_frame: int = Field(ge=0)
    last_frame: int = Field(ge=0)
    reference_frame: int = Field(ge=0)
    scale: int = Field(ge=2, le=4)
    method: Literal["median", "trimmed_mean"]
    motion: Literal["translation", "affine", "homography"]
    ecc_min: float
    noise_floor: float
    copy_threshold: float
    frames_total: int = Field(ge=0)
    frames_used: int = Field(ge=0)
    effective_samples: int = Field(ge=0)
    rejected_frames: list[int]
    near_copies: bool
    spread: dict[str, Any]
    density: dict[str, Any]
    clipped_frames_pct: float
    trimmed_share: float
    agreement_full_scale: float
    libraries: dict[str, str]
    samples: list[RoiSampleInfo]
    notices: list[RoiNoticeInfo]


class CctvReportV1(ReportModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    generated_at: Timestamp
    case: CaseInfo
    upflow: UpflowInfo
    mode: ReportMode
    ai_used: bool
    environment: EnvironmentInfo
    inputs: list[InputFile] = Field(min_length=1)
    acquisition: Acquisition
    osd: OsdInfo
    trim: TrimInfo | None = None
    steps: list[ReportStep]
    processes: list[ProcessInfo]
    outputs: list[OutputFile]
    stills: list[StillPairInfo] = Field(default_factory=list)
    clipping: ClippingInfo | None = None
    frame_hashes: FrameHashesInfo | None = None
    roi: RoiFusionInfo | None = None
    limitations: list[LimitationInfo]
    warnings: list[str] = Field(default_factory=list)
    guidelines: str
    disclaimer: str

    @model_validator(mode="after")
    def _ai_flag_matches_mode(self) -> CctvReportV1:
        if self.ai_used != (self.mode == "ai-visual"):
            raise ValueError("aiUsed must be true exactly when mode is 'ai-visual'")
        if not self.ai_used and any(step.category == "ai" for step in self.steps):
            raise ValueError("The classic lane can't list AI steps")
        return self


def report_schema() -> dict[str, Any]:
    schema = CctvReportV1.model_json_schema(by_alias=True, mode="serialization")
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}


def report_schema_text() -> str:
    return json.dumps(report_schema(), indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def write_report_schema(path: Path = SCHEMA_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_schema_text(), encoding="utf-8", newline="\n")
    return path
