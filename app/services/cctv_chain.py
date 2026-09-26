"""Catalogo de pasos del modo CCTV (spec §4.4).

Datos puros: que pasos hay, en que orden fijo corren, que filtros admite cada
uno con su esquema de parametros, en que carril se permiten y donde cae cada
paso del carril IA. El cliente nunca manda strings de filtro: manda
`{"id": "denoise", "params": {"filter": "hqdn3d", "luma_spatial": 4}}` y
`steps_from_request` lo valida contra el esquema. Armar el string es trabajo
de `ffmpeg_filters.py`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, get_args

Lane = Literal["classic", "ai"]
StepCategory = Literal["classic", "ai", "label"]
AiPlacement = Literal["decode", "composite", "encode", "label"]
EnumValue = str | int

LANES: tuple[Lane, ...] = get_args(Lane)
FFMPEG_FILTERS_DOC = "https://ffmpeg.org/ffmpeg-filters.html"
FILTER_KEY = "filter"
MAX_FRAME_INDEX = 2**31 - 1
MAX_CROP_SIDE = 16384

UNKNOWN_STEP = "cctv.error.unknownStep"
INVALID_STEP = "cctv.error.invalidStep"
DUPLICATE_STEP = "cctv.error.duplicateStep"
STEP_NOT_IN_LANE = "cctv.error.stepNotInLane"
STEP_DEFERRED = "cctv.error.stepDeferred"
STEP_CONFLICT = "cctv.error.stepConflict"
UNKNOWN_FILTER = "cctv.error.unknownFilter"
INVALID_PARAM = "cctv.error.invalidParam"
MISSING_PARAM = "cctv.error.missingParam"
MISSING_AI_LABEL = "cctv.error.missingAiLabel"


class CctvChainError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class IntParam:
    name: str
    minimum: int
    maximum: int
    default: int | None
    even: bool = False
    odd: bool = False
    option: str | None = None


@dataclass(frozen=True, slots=True)
class FloatParam:
    name: str
    minimum: float
    maximum: float
    default: float
    option: str | None = None


@dataclass(frozen=True, slots=True)
class EnumParam:
    name: str
    choices: tuple[EnumValue, ...]
    default: EnumValue
    ai_only: tuple[EnumValue, ...] = ()
    option: str | None = None


ParamSpec = IntParam | FloatParam | EnumParam


@dataclass(frozen=True, slots=True)
class FilterPreset:
    name: str
    label: str
    params: Mapping[str, EnumValue | float]


@dataclass(frozen=True, slots=True)
class FilterSpec:
    name: str
    description: str
    params: tuple[ParamSpec, ...] = ()
    ffmpeg_filters: tuple[str, ...] = ()
    doc_url: str | None = None
    presets: tuple[FilterPreset, ...] = ()

    @property
    def description_key(self) -> str:
        return f"cctv.filter.{self.name}.description"

    def preset_label_key(self, preset: FilterPreset) -> str:
        return f"cctv.filter.{self.name}.preset.{preset.name}"


@dataclass(frozen=True, slots=True)
class StepSpec:
    id: str
    label: str
    category: StepCategory
    filters: tuple[FilterSpec, ...]
    lanes: frozenset[Lane]
    deferred_to: str | None = None
    conflicts_with: tuple[str, ...] = ()
    default_on: bool = False
    gate: str | None = None

    @property
    def label_key(self) -> str:
        return f"cctv.step.{self.id}"


@dataclass(frozen=True, slots=True)
class ResolvedStep:
    id: str
    filter: str
    params: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True, slots=True)
class AiLanePlan:
    decode: tuple[ResolvedStep, ...]
    composite: tuple[ResolvedStep, ...]
    encode: tuple[ResolvedStep, ...]


@dataclass(frozen=True, slots=True)
class Limitation:
    key: str
    text: str


def ffmpeg_doc(anchor: str) -> str:
    return f"{FFMPEG_FILTERS_DOC}#{anchor}"


def _ffmpeg_filter(
    name: str,
    description: str,
    *params: ParamSpec,
    anchor: str | None = None,
    presets: tuple[FilterPreset, ...] = (),
) -> FilterSpec:
    return FilterSpec(name, description, params, (name,), ffmpeg_doc(anchor or name), presets)


def _preset(name: str, label: str, **params: EnumValue | float) -> FilterPreset:
    return FilterPreset(name, label, MappingProxyType(params))


CLASSIC_AND_AI: frozenset[Lane] = frozenset(LANES)
CLASSIC_ONLY: frozenset[Lane] = frozenset({"classic"})
AI_ONLY: frozenset[Lane] = frozenset({"ai"})
NO_LANE: frozenset[Lane] = frozenset()

STABILIZE_GATE = "stabilize"
LENS_GATE = "lens"
# Abiertas mientras sus tests de determinismo (`-k vidstab`, `test_ffmpeg_filters.py -k lens`) pasen.
OPEN_GATES: frozenset[str] = frozenset({STABILIZE_GATE, LENS_GATE})

# Puntos de partida sin calibrar contra exports reales: se ajustan en la vista previa hasta que las rectas lo sean.
_LENSCORRECTION_PRESETS: tuple[FilterPreset, ...] = (
    _preset("mild", "Slight bend (6 mm lens or longer)", k1=-0.05, k2=0.0),
    _preset("wide", "Wide angle (about 4 mm)", k1=-0.12, k2=-0.01),
    _preset("very_wide", "Very wide angle (2.8 mm, common in dome cameras)", k1=-0.22, k2=-0.02),
    _preset("ultra_wide", "Ultra wide (2.0 to 2.2 mm)", k1=-0.3, k2=-0.04),
)


def _fisheye_preset(name: str, label: str, lens_fov: float, view_fov: float) -> FilterPreset:
    return _preset(name, label, ih_fov=lens_fov, iv_fov=lens_fov, d_fov=view_fov, yaw=0.0, pitch=0.0)


_V360_PRESETS: tuple[FilterPreset, ...] = (
    _fisheye_preset("fisheye_180", "180° fisheye, straight ahead", 180.0, 120.0),
    _fisheye_preset("fisheye_190", "190° fisheye, straight ahead", 190.0, 120.0),
    _fisheye_preset("fisheye_180_narrow", "180° fisheye, narrow view", 180.0, 80.0),
)

_DEINTERLACE_PARAMS: tuple[ParamSpec, ...] = (
    EnumParam("mode", ("send_frame", "send_field"), "send_frame", ai_only=("send_field",)),
    EnumParam("parity", ("auto", "tff", "bff"), "auto"),
)
_QP = IntParam("qp", 0, 63, 0)

CCTV_CHAIN: tuple[StepSpec, ...] = (
    StepSpec(
        "trim",
        "Trim",
        "classic",
        (
            FilterSpec(
                "trim",
                "Kept only the selected frames, counted by frame number; no frame was retimed.",
                (
                    IntParam("start_frame", 0, MAX_FRAME_INDEX, None),
                    IntParam("end_frame", 0, MAX_FRAME_INDEX, None),
                ),
                ("trim", "atrim"),
                ffmpeg_doc("trim"),
            ),
        ),
        CLASSIC_AND_AI,
    ),
    StepSpec(
        "aspect",
        "Aspect ratio",
        "classic",
        (
            _ffmpeg_filter(
                "setsar",
                "Marked the real pixel shape so the video shows at its true proportions; no pixel was changed (setsar).",
                IntParam("num", 1, 1000, None),
                IntParam("den", 1, 1000, None),
                anchor="setdar_002c-setsar",
            ),
        ),
        CLASSIC_AND_AI,
    ),
    StepSpec(
        "deinterlace",
        "Deinterlace",
        "classic",
        (
            _ffmpeg_filter(
                "bwdif",
                "Rebuilt full frames from the two interlaced fields, keeping one output frame per input frame (bwdif).",
                *_DEINTERLACE_PARAMS,
            ),
            _ffmpeg_filter(
                "yadif",
                "Rebuilt full frames from the two interlaced fields, keeping one output frame per input frame (yadif).",
                *_DEINTERLACE_PARAMS,
            ),
        ),
        CLASSIC_AND_AI,
    ),
    StepSpec(
        "deblock",
        "Reduce blocking",
        "classic",
        (
            _ffmpeg_filter(
                "deblock",
                "Smoothed the edges of the compression blocks (deblock).",
                EnumParam("filter_type", ("weak", "strong"), "weak", option="filter"),
                IntParam("block", 4, 512, 8),
            ),
            _ffmpeg_filter(
                "fspp",
                "Reduced compression blocks by re-averaging shifted copies of each block (fspp).",
                IntParam("quality", 4, 5, 4),
                IntParam("qp", 0, 64, 0),
                IntParam("strength", -15, 32, 0),
            ),
            _ffmpeg_filter(
                "spp",
                "Reduced compression blocks by re-averaging shifted copies of each block (spp).",
                IntParam("quality", 0, 6, 3),
                _QP,
                EnumParam("mode", ("hard", "soft"), "hard"),
            ),
            _ffmpeg_filter(
                "pp7",
                "Reduced compression blocks with a 7-point block filter (pp7).",
                IntParam("qp", 0, 64, 0),
                EnumParam("mode", ("hard", "soft", "medium"), "medium"),
            ),
            _ffmpeg_filter(
                "uspp",
                "Reduced compression blocks by re-encoding shifted copies and averaging them (uspp).",
                IntParam("quality", 0, 8, 3),
                _QP,
            ),
        ),
        CLASSIC_AND_AI,
    ),
    StepSpec(
        "ai_deblock",
        "AI deblock",
        "ai",
        (
            FilterSpec(
                "drunet_deblock",
                "An AI model (DRUNet) removed compression blocks frame by frame; it can alter fine detail.",
                (IntParam("strength", 0, 100, 40),),
                (),
                "https://github.com/cszn/DPIR",
            ),
        ),
        AI_ONLY,
        conflicts_with=("deblock", "denoise"),
    ),
    StepSpec(
        "stabilize",
        "Stabilize",
        "classic",
        (
            FilterSpec(
                "vidstab",
                "Measured the camera shake and moved each frame to cancel it (vidstabdetect, vidstabtransform).",
                (IntParam("shakiness", 1, 10, 5), IntParam("smoothing", 0, 1000, 10)),
                ("vidstabdetect", "vidstabtransform"),
                ffmpeg_doc("vidstabdetect"),
            ),
        ),
        CLASSIC_ONLY,
        gate=STABILIZE_GATE,
    ),
    StepSpec(
        "denoise",
        "Reduce noise",
        "classic",
        (
            _ffmpeg_filter(
                "hqdn3d",
                "Reduced noise by averaging each pixel with nearby pixels and frames (hqdn3d).",
                FloatParam("luma_spatial", 0.0, 255.0, 4.0),
                FloatParam("chroma_spatial", 0.0, 255.0, 3.0),
                FloatParam("luma_tmp", 0.0, 255.0, 6.0),
                FloatParam("chroma_tmp", 0.0, 255.0, 4.5),
            ),
            _ffmpeg_filter(
                "atadenoise",
                "Reduced noise by averaging each pixel with the same pixel in nearby frames when it barely changes (atadenoise).",
                FloatParam("0a", 0.0, 0.3, 0.02),
                FloatParam("0b", 0.0, 5.0, 0.04),
                FloatParam("1a", 0.0, 0.3, 0.02),
                FloatParam("1b", 0.0, 5.0, 0.04),
                FloatParam("2a", 0.0, 0.3, 0.02),
                FloatParam("2b", 0.0, 5.0, 0.04),
                IntParam("s", 5, 129, 9, odd=True),
            ),
            _ffmpeg_filter(
                "tmedian",
                "Reduced noise by taking the median of each pixel across nearby frames (tmedian).",
                IntParam("radius", 1, 127, 1),
                FloatParam("percentile", 0.0, 1.0, 0.5),
            ),
            _ffmpeg_filter(
                "fftdnoiz",
                "Reduced noise by removing weak frequencies block by block (fftdnoiz).",
                FloatParam("sigma", 0.0, 100.0, 1.0),
                FloatParam("amount", 0.01, 1.0, 1.0),
                IntParam("block", 8, 256, 32),
                FloatParam("overlap", 0.2, 0.8, 0.5),
                IntParam("prev", 0, 1, 0),
                IntParam("next", 0, 1, 0),
            ),
            _ffmpeg_filter(
                "nlmeans",
                "Reduced noise by averaging similar patches within each frame (nlmeans).",
                FloatParam("s", 1.0, 30.0, 1.0),
                IntParam("p", 1, 99, 7, odd=True),
                IntParam("r", 1, 99, 15, odd=True),
            ),
            _ffmpeg_filter(
                "bm3d",
                "Reduced noise by grouping similar blocks and filtering them together (bm3d).",
                FloatParam("sigma", 0.0, 1000.0, 1.0),
                IntParam("block", 8, 64, 16),
                IntParam("bstep", 1, 64, 4),
                IntParam("group", 1, 256, 1),
            ),
        ),
        CLASSIC_AND_AI,
    ),
    StepSpec(
        "lens",
        "Lens correction",
        "classic",
        (
            _ffmpeg_filter(
                "lenscorrection",
                "Straightened lines bent by the lens by moving pixels toward or away from the center; "
                "no new pixel values were computed (lenscorrection).",
                FloatParam("k1", -1.0, 1.0, 0.0),
                FloatParam("k2", -1.0, 1.0, 0.0),
                presets=_LENSCORRECTION_PRESETS,
            ),
            _ffmpeg_filter(
                "v360",
                "Flattened a fisheye image into a regular view of the same size by moving pixels; "
                "no new pixel values were computed (v360).",
                FloatParam("ih_fov", 1.0, 360.0, 180.0),
                FloatParam("iv_fov", 1.0, 360.0, 180.0),
                FloatParam("d_fov", 10.0, 170.0, 120.0),
                FloatParam("yaw", -180.0, 180.0, 0.0),
                FloatParam("pitch", -90.0, 90.0, 0.0),
                presets=_V360_PRESETS,
            ),
        ),
        CLASSIC_ONLY,
        gate=LENS_GATE,
    ),
    StepSpec(
        "crop",
        "Crop",
        "classic",
        (
            _ffmpeg_filter(
                "crop",
                "Cut the frame down to the selected area (crop).",
                IntParam("w", 2, MAX_CROP_SIDE, None, even=True),
                IntParam("h", 2, MAX_CROP_SIDE, None, even=True),
                IntParam("x", 0, MAX_CROP_SIDE, None),
                IntParam("y", 0, MAX_CROP_SIDE, None),
            ),
        ),
        CLASSIC_AND_AI,
    ),
    StepSpec(
        "gray",
        "Grayscale",
        "classic",
        (
            FilterSpec(
                "gray",
                "Removed the color channels, which only carry noise in infrared footage (format=gray).",
                (),
                ("format",),
                ffmpeg_doc("format"),
            ),
        ),
        CLASSIC_AND_AI,
    ),
    StepSpec(
        "levels",
        "Brightness and contrast",
        "classic",
        (
            _ffmpeg_filter(
                "eq",
                "Adjusted brightness, contrast and gamma (eq).",
                FloatParam("brightness", -1.0, 1.0, 0.0),
                FloatParam("contrast", 0.0, 4.0, 1.0),
                FloatParam("gamma", 0.1, 10.0, 1.0),
                FloatParam("saturation", 0.0, 3.0, 1.0),
            ),
            _ffmpeg_filter(
                "colorlevels",
                "Moved the black and white points of each color channel (colorlevels).",
                FloatParam("rimin", -1.0, 1.0, 0.0),
                FloatParam("gimin", -1.0, 1.0, 0.0),
                FloatParam("bimin", -1.0, 1.0, 0.0),
                FloatParam("rimax", -1.0, 1.0, 1.0),
                FloatParam("gimax", -1.0, 1.0, 1.0),
                FloatParam("bimax", -1.0, 1.0, 1.0),
            ),
            _ffmpeg_filter(
                "curves",
                "Applied a standard tone curve (curves).",
                EnumParam(
                    "preset",
                    (
                        "none",
                        "darker",
                        "lighter",
                        "increase_contrast",
                        "linear_contrast",
                        "medium_contrast",
                        "strong_contrast",
                    ),
                    "none",
                ),
            ),
        ),
        CLASSIC_AND_AI,
    ),
    StepSpec(
        "scale",
        "Resize",
        "classic",
        (
            _ffmpeg_filter(
                "scale",
                "Resized the frame; nearest neighbor repeats recorded pixels and creates no new values (scale).",
                IntParam("factor", 1, 8, 2),
                EnumParam("flags", ("neighbor", "bicubic", "lanczos"), "neighbor"),
            ),
        ),
        CLASSIC_AND_AI,
    ),
    StepSpec(
        "ai_upscale",
        "AI upscale",
        "ai",
        (
            FilterSpec(
                "onnx_upscale",
                "An AI model enlarged the frame; generative models invent texture that was never recorded.",
            ),
        ),
        AI_ONLY,
    ),
    StepSpec(
        "sharpen",
        "Sharpen",
        "classic",
        (
            _ffmpeg_filter(
                "cas",
                "Increased local contrast at edges; it can create halos around edges (cas).",
                FloatParam("strength", 0.0, 1.0, 0.5),
            ),
            _ffmpeg_filter(
                "unsharp",
                "Increased contrast at edges with an unsharp mask; it can create halos around edges (unsharp).",
                IntParam("luma_msize_x", 3, 23, 5, odd=True),
                IntParam("luma_msize_y", 3, 23, 5, odd=True),
                FloatParam("luma_amount", -2.0, 5.0, 1.0),
            ),
        ),
        CLASSIC_AND_AI,
    ),
    StepSpec(
        "osd_protect",
        "Protect on-screen text",
        "classic",
        (
            FilterSpec(
                "osd_restore",
                "Put back the on-screen text boxes (date, time, camera) from the original frames so the filters did not blur them.",
                (),
                ("crop", "overlay"),
                ffmpeg_doc("overlay"),
            ),
        ),
        CLASSIC_AND_AI,
    ),
    StepSpec(
        "interpolate",
        "Frame interpolation",
        "ai",
        (
            FilterSpec("rife", "An AI model created in-between frames that were never recorded (RIFE)."),
            FilterSpec("gmfss", "An AI model created in-between frames that were never recorded (GMFSS)."),
        ),
        NO_LANE,
        deferred_to="P4-INTERP",
    ),
    StepSpec(
        "ai_label",
        "AI label",
        "label",
        (
            FilterSpec(
                "label_band",
                "Added a visible 'AI-enhanced visualization' band outside the picture and a small mark inside it.",
                (),
                ("pad", "overlay"),
                ffmpeg_doc("pad"),
            ),
        ),
        AI_ONLY,
    ),
)

AI_LANE_PLACEMENT: Mapping[str, AiPlacement] = MappingProxyType(
    {
        "trim": "decode",
        "aspect": "decode",
        "deinterlace": "decode",
        "deblock": "decode",
        "denoise": "decode",
        "crop": "decode",
        "gray": "decode",
        "ai_deblock": "composite",
        "ai_upscale": "composite",
        "osd_protect": "composite",
        "levels": "encode",
        "scale": "encode",
        "sharpen": "encode",
        "ai_label": "label",
    }
)

AI_LABEL_STEP = "ai_label"
_SPEC_BY_ID: Mapping[str, StepSpec] = MappingProxyType({step.id: step for step in CCTV_CHAIN})
_POSITION_BY_ID: Mapping[str, int] = MappingProxyType({step.id: index for index, step in enumerate(CCTV_CHAIN)})
_LANE_NAMES: Mapping[Lane, str] = MappingProxyType(
    {"classic": "Classic filters (no AI)", "ai": "AI enhancement (visual only)"}
)

GEOMETRY_CHANGED = Limitation(
    "cctv.limitation.geometryChanged",
    "Geometry was changed (crop/scale): sizes and positions differ from the original.",
)
GRAYSCALE = Limitation("cctv.limitation.grayscale", "Color was removed (converted to grayscale).")
NEW_PIXEL_VALUES = Limitation(
    "cctv.limitation.newPixelValues",
    "Bicubic or Lanczos scaling: new pixel values were computed.",
)
SHARPEN_HALOS = Limitation("cctv.limitation.sharpenHalos", "Sharpening can create halos around edges.")
STABILIZED = Limitation(
    "cctv.limitation.stabilized",
    "Stabilization moved and resampled every frame: positions and pixel values differ from the original, "
    "and edges the motion uncovered are black.",
)
LENS_CORRECTED = Limitation(
    "cctv.limitation.lensCorrected",
    "Lens correction moved pixels to straighten lines: sizes and positions differ from the original, "
    "and part of the recorded view can be cut off or turn black.",
)


def step_spec(step_id: str) -> StepSpec:
    spec = _SPEC_BY_ID.get(step_id)
    if spec is None:
        raise CctvChainError(UNKNOWN_STEP, f"Unknown CCTV step {step_id!r}. Valid: {', '.join(_SPEC_BY_ID)}.")
    return spec


def _filter_spec(step: StepSpec, name: Any) -> FilterSpec:
    if name is None:
        return step.filters[0]
    for spec in step.filters:
        if spec.name == name:
            return spec
    valid = ", ".join(spec.name for spec in step.filters)
    raise CctvChainError(UNKNOWN_FILTER, f"Step {step.id!r} has no filter {name!r}. Valid: {valid}.")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _invalid(step_id: str, name: str, reason: str) -> CctvChainError:
    return CctvChainError(INVALID_PARAM, f"Step {step_id!r}, parameter {name!r}: {reason}.")


def _check_range(step_id: str, spec: IntParam | FloatParam, value: float) -> None:
    if not spec.minimum <= value <= spec.maximum:
        raise _invalid(step_id, spec.name, f"must be between {spec.minimum} and {spec.maximum}")


def _check_int(step_id: str, spec: IntParam, value: Any, lane: Lane) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise _invalid(step_id, spec.name, "must be an integer")
    _check_range(step_id, spec, value)
    if spec.even and value % 2:
        raise _invalid(step_id, spec.name, "must be even")
    if spec.odd and not value % 2:
        raise _invalid(step_id, spec.name, "must be odd")
    return value


def _check_float(step_id: str, spec: FloatParam, value: Any, lane: Lane) -> float:
    if not _is_number(value) or not math.isfinite(value):
        raise _invalid(step_id, spec.name, "must be a finite number")
    _check_range(step_id, spec, value)
    return value


def _check_enum(step_id: str, spec: EnumParam, value: Any, lane: Lane) -> EnumValue:
    allowed = lane_choices(spec, lane)
    if isinstance(value, bool) or value not in allowed:
        raise _invalid(step_id, spec.name, f"must be one of {', '.join(map(str, allowed))} in {_LANE_NAMES[lane]}")
    return value


_CHECKERS = {IntParam: _check_int, FloatParam: _check_float, EnumParam: _check_enum}


def lane_choices(spec: EnumParam, lane: Lane) -> tuple[EnumValue, ...]:
    if lane == "ai":
        return spec.choices
    return tuple(choice for choice in spec.choices if choice not in spec.ai_only)


def _param_value(step_id: str, spec: ParamSpec, raw: Mapping[str, Any], lane: Lane) -> Any:
    if spec.name not in raw:
        if spec.default is None:
            raise CctvChainError(MISSING_PARAM, f"Step {step_id!r} needs parameter {spec.name!r}.")
        return spec.default
    return _CHECKERS[type(spec)](step_id, spec, raw[spec.name], lane)


def _reject_unknown_params(step_id: str, spec: FilterSpec, raw: Mapping[str, Any]) -> None:
    known = {param.name for param in spec.params} | {FILTER_KEY}
    unknown = sorted(str(key) for key in raw if key not in known)
    if unknown:
        raise _invalid(step_id, unknown[0], f"is not a parameter of {spec.name}")


def _check_trim_order(step_id: str, params: Mapping[str, Any]) -> None:
    if step_id == "trim" and params["end_frame"] < params["start_frame"]:
        raise _invalid(step_id, "end_frame", "must not come before start_frame")


def _resolve_params(step: StepSpec, raw: Mapping[str, Any], lane: Lane) -> ResolvedStep:
    spec = _filter_spec(step, raw.get(FILTER_KEY))
    _reject_unknown_params(step.id, spec, raw)
    params = {param.name: _param_value(step.id, param, raw, lane) for param in spec.params}
    _check_trim_order(step.id, params)
    return ResolvedStep(step.id, spec.name, MappingProxyType(params))


def is_open(step: StepSpec, gates: frozenset[str]) -> bool:
    return step.gate is None or step.gate in gates


def _check_lane(step: StepSpec, lane: Lane, gates: frozenset[str]) -> None:
    if step.deferred_to is not None:
        raise CctvChainError(STEP_DEFERRED, f"Step {step.id!r} is not available in this version ({step.deferred_to}).")
    if not is_open(step, gates):
        raise CctvChainError(STEP_DEFERRED, f"Step {step.id!r} is not available in this version.")
    if lane not in step.lanes:
        raise CctvChainError(STEP_NOT_IN_LANE, f"Step {step.id!r} isn't allowed in {_LANE_NAMES[lane]}.")


def _raw_step_parts(raw: Any) -> tuple[str, Mapping[str, Any]]:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("id"), str):
        raise CctvChainError(INVALID_STEP, "Each step must be an object with a string 'id'.")
    params = raw.get("params", {})
    if not isinstance(params, Mapping):
        raise CctvChainError(INVALID_STEP, f"Step {raw['id']!r}: 'params' must be an object.")
    return raw["id"], params


def _resolve_step(raw: Any, lane: Lane, gates: frozenset[str]) -> ResolvedStep:
    step_id, params = _raw_step_parts(raw)
    step = step_spec(step_id)
    _check_lane(step, lane, gates)
    return _resolve_params(step, params, lane)


def _reject_duplicates(steps: Sequence[ResolvedStep]) -> None:
    seen: set[str] = set()
    for step in steps:
        if step.id in seen:
            raise CctvChainError(DUPLICATE_STEP, f"Step {step.id!r} appears more than once.")
        seen.add(step.id)


def _reject_conflicts(steps: Sequence[ResolvedStep]) -> None:
    chosen = {step.id for step in steps}
    for step_id in sorted(chosen):
        clashes = sorted(chosen.intersection(_SPEC_BY_ID[step_id].conflicts_with))
        if clashes:
            raise CctvChainError(
                STEP_CONFLICT, f"Step {step_id!r} can't be combined with {', '.join(repr(c) for c in clashes)}."
            )


def _with_ai_label(steps: Sequence[ResolvedStep], lane: Lane) -> tuple[ResolvedStep, ...]:
    kept = tuple(step for step in steps if step.id != AI_LABEL_STEP)
    if lane != "ai":
        return kept
    label = ResolvedStep(AI_LABEL_STEP, step_spec(AI_LABEL_STEP).filters[0].name)
    return (*kept, label)


def in_catalog_order(steps: Sequence[ResolvedStep]) -> tuple[ResolvedStep, ...]:
    return tuple(sorted(steps, key=lambda step: _POSITION_BY_ID[step.id]))


def steps_from_request(
    raw_steps: Sequence[Any], lane: Lane, gates: frozenset[str] = OPEN_GATES
) -> tuple[ResolvedStep, ...]:
    resolved = [_resolve_step(raw, lane, gates) for raw in raw_steps]
    _reject_duplicates(resolved)
    _reject_conflicts(resolved)
    return in_catalog_order(_with_ai_label(resolved, lane))


def _placed(steps: Sequence[ResolvedStep], placement: AiPlacement) -> tuple[ResolvedStep, ...]:
    return tuple(step for step in steps if AI_LANE_PLACEMENT[step.id] == placement)


def ai_lane_plan(steps: Sequence[ResolvedStep]) -> AiLanePlan:
    if not any(step.id == AI_LABEL_STEP for step in steps):
        raise CctvChainError(MISSING_AI_LABEL, "The AI lane always ends with the AI label.")
    ordered = in_catalog_order(steps)
    return AiLanePlan(
        decode=_placed(ordered, "decode"),
        composite=_placed(ordered, "composite"),
        encode=_placed(ordered, "encode") + _placed(ordered, "label"),
    )


def _changes_geometry(step: ResolvedStep) -> bool:
    return step.id in {"crop", "scale"}


def _computes_new_pixels(step: ResolvedStep) -> bool:
    return step.id == "scale" and step.params.get("flags") != "neighbor"


_LIMITATION_RULES: tuple[tuple[Callable[[ResolvedStep], bool], Limitation], ...] = (
    (_changes_geometry, GEOMETRY_CHANGED),
    (lambda step: step.id == "gray", GRAYSCALE),
    (_computes_new_pixels, NEW_PIXEL_VALUES),
    (lambda step: step.id == "sharpen", SHARPEN_HALOS),
    (lambda step: step.id == "stabilize", STABILIZED),
    (lambda step: step.id == "lens", LENS_CORRECTED),
)


def _step_limitations(step: ResolvedStep) -> tuple[Limitation, ...]:
    return tuple(limitation for applies, limitation in _LIMITATION_RULES if applies(step))


def limitations_for(steps: Sequence[ResolvedStep]) -> tuple[Limitation, ...]:
    return tuple(dict.fromkeys(limitation for step in steps for limitation in _step_limitations(step)))


def _param_schema(spec: ParamSpec, lane: Lane) -> dict[str, Any]:
    if isinstance(spec, EnumParam):
        return {"name": spec.name, "type": "enum", "choices": list(lane_choices(spec, lane)), "default": spec.default}
    schema: dict[str, Any] = {
        "name": spec.name,
        "type": "int" if isinstance(spec, IntParam) else "float",
        "min": spec.minimum,
        "max": spec.maximum,
        "default": spec.default,
    }
    if isinstance(spec, IntParam):
        schema.update(even=spec.even, odd=spec.odd)
    return schema


def _preset_schema(spec: FilterSpec, preset: FilterPreset) -> dict[str, Any]:
    return {
        "name": preset.name,
        "labelKey": spec.preset_label_key(preset),
        "label": preset.label,
        "params": dict(preset.params),
    }


def _filter_schema(spec: FilterSpec, lane: Lane) -> dict[str, Any]:
    return {
        "name": spec.name,
        "descriptionKey": spec.description_key,
        "description": spec.description,
        "docUrl": spec.doc_url,
        "ffmpegFilters": list(spec.ffmpeg_filters),
        "params": [_param_schema(param, lane) for param in spec.params],
        "presets": [_preset_schema(spec, preset) for preset in spec.presets],
    }


def _step_schema(step: StepSpec, lane: Lane) -> dict[str, Any]:
    return {
        "id": step.id,
        "labelKey": step.label_key,
        "label": step.label,
        "category": step.category,
        "defaultOn": step.default_on,
        "placement": AI_LANE_PLACEMENT.get(step.id) if lane == "ai" else None,
        "filters": [_filter_schema(spec, lane) for spec in step.filters],
    }


def catalog_schema(lane: Lane, gates: frozenset[str] = OPEN_GATES) -> list[dict[str, Any]]:
    return [_step_schema(step, lane) for step in CCTV_CHAIN if lane in step.lanes and is_open(step, gates)]
