from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.services.engines.colorize import MAX_SATURATION, RENDER_SIZE
from app.services.engines.scratch_detect import MAX_GROW_PX
from app.services.photo_geometry import MAX_STRAIGHTEN_DEG

Box = tuple[int, int, int, int]


def _unit() -> Any:
    return Field(default=None, ge=0.0, le=1.0)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RestoreGeometry(_StrictModel):
    rotate90: int = Field(default=0, ge=0, le=3)
    crop: Box | None = None
    angle: float = Field(default=0.0, ge=-MAX_STRAIGHTEN_DEG, le=MAX_STRAIGHTEN_DEG)


class DescreenOptions(_StrictModel):
    mode: Literal["auto", "halftone", "texture"] | None = None
    strength: float | None = _unit()
    # Lo propone el diagnostico; el paso usa el periodo medido en la copia de trabajo.
    period: float | None = Field(default=None, gt=0.0)


class RepairOptions(_StrictModel):
    engine: Literal["fast", "classic"] | None = None
    sensitivity: float | None = _unit()
    grow_px: int | None = Field(default=None, ge=-MAX_GROW_PX, le=MAX_GROW_PX)
    use_user_mask: bool | None = None
    leave_large_holes: bool | None = None
    leave_faces: bool | None = None


class DeblockOptions(_StrictModel):
    strength: float | None = _unit()


class DenoiseOptions(_StrictModel):
    strength: float | None = _unit()
    keep_grain: float | None = _unit()


class ToneOptions(_StrictModel):
    strength: float | None = _unit()
    gray_point: tuple[int, int] | None = None
    keep_tone: bool | None = None
    neutral_gray: bool | None = None
    fix_faded: bool | None = None
    local_contrast: bool | None = None


class FacesOptions(_StrictModel):
    model: str | None = None
    blend: float | None = _unit()
    selected: list[int] | None = None
    per_face: dict[int, float] | None = None


class ColorizeOptionsRequest(_StrictModel):
    model: str | None = None
    render_size: int | None = Field(default=None, ge=RENDER_SIZE, le=RENDER_SIZE)
    strength: float | None = _unit()
    saturation: float | None = Field(default=None, ge=0.0, le=MAX_SATURATION)
    from_luminance: bool | None = None


class RestoreOptions(_StrictModel):
    preset: str | None = None
    geometry: RestoreGeometry | None = None
    descreen: DescreenOptions | None = None
    repair: RepairOptions | None = None
    deblock: DeblockOptions | None = None
    denoise: DenoiseOptions | None = None
    tone: ToneOptions | None = None
    upscale_mode: Literal["none", "classic", "ai"] | None = None
    faces: FacesOptions | None = None
    colorize: ColorizeOptionsRequest | None = None
    preview_crop: Box | None = None
    badge: bool | None = None
    keep_gps: bool | None = None
    photo_date: str | None = None
    batch: bool | None = None


class RecomposeFace(_StrictModel):
    enabled: bool = True
    blend: float = Field(ge=0.0, le=1.0)


class RecomposeRequest(_StrictModel):
    faces: dict[int, RecomposeFace]


class RestoreStepProposalResponse(BaseModel):
    step_id: str = Field(serialization_alias="stepId")
    options: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True


class RestoreFindingResponse(BaseModel):
    key: str
    value: float | int | str | None = None
    reason_key: str = Field(serialization_alias="reasonKey")
    params: dict[str, Any] = Field(default_factory=dict)
    proposes: list[RestoreStepProposalResponse] = Field(default_factory=list)
    missing_pack: str | None = Field(default=None, serialization_alias="missingPack")
    message: str | None = None


class RestoreDiagnosisResponse(BaseModel):
    findings: list[RestoreFindingResponse]
    suggested_presets: list[str] = Field(serialization_alias="suggestedPresets")
    tone_kind: str = Field(serialization_alias="toneKind")


class RestoreFaceResponse(BaseModel):
    index: int
    box: list[float] | None = None
    eye_px: float | None = Field(default=None, serialization_alias="eyePx")
    sharpness: float | None = None
    confidence: float | None = None
    enabled: bool
    blend: float
    thumbnail_url: str | None = Field(default=None, serialization_alias="thumbnailUrl")


class RestoreDamageResponse(BaseModel):
    coverage: float | None = None
    prob_url: str | None = Field(default=None, serialization_alias="probUrl")
    large_holes: int = Field(default=0, serialization_alias="largeHoles")


class StepEtaResponse(BaseModel):
    gpu_seconds: float = Field(serialization_alias="gpuSeconds")
    cpu_seconds: float = Field(serialization_alias="cpuSeconds")


class RestoreEtaResponse(BaseModel):
    gpu_seconds: float = Field(serialization_alias="gpuSeconds")
    cpu_seconds: float = Field(serialization_alias="cpuSeconds")
    per_step: dict[str, StepEtaResponse] = Field(default_factory=dict, serialization_alias="perStep")


class RestorePresetSelectionResponse(BaseModel):
    steps: list[str]
    options: dict[str, dict[str, Any]]


class RestoreAnalysisResponse(BaseModel):
    token: str
    original_name: str = Field(serialization_alias="originalName")
    sha256: str
    width: int
    height: int
    bit_depth: int = Field(serialization_alias="bitDepth")
    has_icc: bool = Field(serialization_alias="hasIcc")
    geometry: RestoreGeometry
    preview_url: str = Field(serialization_alias="previewUrl")
    diagnosis: RestoreDiagnosisResponse
    proposed_preset: str = Field(serialization_alias="proposedPreset")
    proposed_steps: list[str] = Field(serialization_alias="proposedSteps")
    proposed_options: dict[str, dict[str, Any]] = Field(serialization_alias="proposedOptions")
    preset_selections: dict[str, RestorePresetSelectionResponse] = Field(serialization_alias="presetSelections")
    faces: list[RestoreFaceResponse]
    damage: RestoreDamageResponse
    damage_over_faces: bool = Field(serialization_alias="damageOverFaces")
    eta: RestoreEtaResponse


class RestoreMaskResponse(BaseModel):
    coverage: float
    width: int
    height: int


class RecomposeResponse(BaseModel):
    sidecar: dict[str, Any]


class RestoreStepCapabilityResponse(BaseModel):
    id: str
    phase: str
    strategy: str
    label_key: str = Field(serialization_alias="labelKey")
    description_key: str = Field(serialization_alias="descriptionKey")
    invents_detail: bool = Field(serialization_alias="inventsDetail")
    warning_key: str | None = Field(default=None, serialization_alias="warningKey")
    pack: str | None = None
    installed: bool


class RestorePresetResponse(BaseModel):
    id: str
    label_key: str = Field(serialization_alias="labelKey")
    description_key: str = Field(serialization_alias="descriptionKey")
    steps: list[str]


class RestoreCapabilitiesResponse(BaseModel):
    steps: list[RestoreStepCapabilityResponse]
    presets: list[RestorePresetResponse]
    halftone_denoise_limit: float = Field(serialization_alias="halftoneDenoiseLimit")


class LicenseFileResponse(BaseModel):
    name: str
    text: str


class LicenseModelResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    license_spdx: str = Field(alias="licenseSpdx")
    license_url: str = Field(alias="licenseUrl")
    copyright: str
    attribution: str
    data_lineage: str = Field(alias="dataLineage")
    commercial_use: str = Field(alias="commercialUse")
    source_url: str = Field(alias="sourceUrl")
    source_revision: str = Field(alias="sourceRevision")
    modifications: list[str]
    files: list[LicenseFileResponse]


class LicensePackResponse(BaseModel):
    pack: str
    models: list[LicenseModelResponse]


class ThirdPartyNoticeResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    title: str
    section: str
    fields: dict[str, list[str]]
    license_text: str | None = Field(default=None, alias="licenseText")


class LicensesResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    packs: list[LicensePackResponse]
    third_party: list[ThirdPartyNoticeResponse] = Field(alias="thirdParty")
