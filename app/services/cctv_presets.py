"""Presets del modo CCTV (spec §4.5), como tabla de datos.

Cada preset da la cadena de cada carril en el formato del request
(`{"id", "params"}`), para que pase por la misma validacion que un pedido del
cliente. Los pasos condicionales solo entran si el diagnostico los detecto.
Ningun preset enciende `sharpen` ni usa Lanczos: son manuales y declarados.
Los parametros son provisionales hasta P2-VAL (exports reales del HiLook).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from app.services.cctv_chain import LANES, CctvChainError, Lane

PresetId = Literal["day", "night_ir", "analog", "low_res"]
StepCondition = Literal["always", "interlaced", "anamorphic"]

UNKNOWN_PRESET = "cctv.error.unknownPreset"
SQUARE_PIXELS = (1, 1)


@dataclass(frozen=True, slots=True)
class PresetContext:
    interlaced: bool = False
    sample_aspect: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class PresetStep:
    id: str
    params: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    when: StepCondition = "always"


@dataclass(frozen=True, slots=True)
class PresetSpec:
    id: PresetId
    label: str
    description: str
    classic: tuple[PresetStep, ...]
    ai: tuple[PresetStep, ...]
    ai_upscale_hint: int | None = None

    @property
    def label_key(self) -> str:
        return f"cctv.preset.{self.id}"

    @property
    def description_key(self) -> str:
        return f"cctv.preset.{self.id}.description"

    def steps_for(self, lane: Lane) -> tuple[PresetStep, ...]:
        return self.classic if lane == "classic" else self.ai


def _params(**values: Any) -> Mapping[str, Any]:
    return MappingProxyType(values)


ASPECT_IF_ANAMORPHIC = PresetStep("aspect", when="anamorphic")
BWDIF_FRAME = _params(filter="bwdif", mode="send_frame")
DEINTERLACE_IF_INTERLACED = PresetStep("deinterlace", BWDIF_FRAME, "interlaced")
DEINTERLACE_ALWAYS = PresetStep("deinterlace", BWDIF_FRAME)
WEAK_DEBLOCK = PresetStep("deblock", _params(filter="deblock", filter_type="weak", block=8))
STRONG_DEBLOCK = PresetStep("deblock", _params(filter="deblock", filter_type="strong", block=8))
DAY_DENOISE = PresetStep(
    "denoise", _params(filter="hqdn3d", luma_spatial=3, chroma_spatial=2, luma_tmp=4, chroma_tmp=3)
)
NIGHT_DENOISE = PresetStep(
    "denoise",
    _params(filter="atadenoise", **{"0a": 0.04, "0b": 0.08, "1a": 0.04, "1b": 0.08, "s": 9}),
)
AI_DEBLOCK_DAY = PresetStep("ai_deblock", _params(strength=40))
AI_DEBLOCK_NIGHT = PresetStep("ai_deblock", _params(strength=60))
GRAY = PresetStep("gray")
NIGHT_GAMMA = PresetStep("levels", _params(filter="eq", gamma=1.2))
NEAREST_X2 = PresetStep("scale", _params(factor=2, flags="neighbor"))
OSD_PROTECT = PresetStep("osd_protect")

DAY_CLASSIC = (ASPECT_IF_ANAMORPHIC, DEINTERLACE_IF_INTERLACED, WEAK_DEBLOCK, DAY_DENOISE, OSD_PROTECT)
DAY_AI = (ASPECT_IF_ANAMORPHIC, DEINTERLACE_IF_INTERLACED, AI_DEBLOCK_DAY, OSD_PROTECT)

CCTV_PRESETS: tuple[PresetSpec, ...] = (
    PresetSpec(
        "day",
        "Day",
        "For color footage with moderate compression blocking.",
        DAY_CLASSIC,
        DAY_AI,
    ),
    PresetSpec(
        "night_ir",
        "Night / IR",
        "For dark or infrared footage with no real color.",
        (ASPECT_IF_ANAMORPHIC, DEINTERLACE_IF_INTERLACED, STRONG_DEBLOCK, NIGHT_DENOISE, GRAY, NIGHT_GAMMA, OSD_PROTECT),
        (ASPECT_IF_ANAMORPHIC, DEINTERLACE_IF_INTERLACED, AI_DEBLOCK_NIGHT, GRAY, NIGHT_GAMMA, OSD_PROTECT),
    ),
    PresetSpec(
        "analog",
        "Analog (interlaced)",
        "For interlaced footage from analog cameras.",
        (ASPECT_IF_ANAMORPHIC, DEINTERLACE_ALWAYS, WEAK_DEBLOCK, DAY_DENOISE, OSD_PROTECT),
        (ASPECT_IF_ANAMORPHIC, DEINTERLACE_ALWAYS, AI_DEBLOCK_DAY, OSD_PROTECT),
    ),
    PresetSpec(
        "low_res",
        "Low-res sub-stream",
        "For small sub-stream recordings (704x576 or less); enlarged without inventing pixels.",
        (*DAY_CLASSIC[:-1], NEAREST_X2, OSD_PROTECT),
        (*DAY_AI[:-1], NEAREST_X2, OSD_PROTECT),
        ai_upscale_hint=2,
    ),
)

_PRESET_BY_ID: Mapping[str, PresetSpec] = MappingProxyType({preset.id: preset for preset in CCTV_PRESETS})


def preset_spec(preset_id: str) -> PresetSpec:
    preset = _PRESET_BY_ID.get(preset_id)
    if preset is None:
        raise CctvChainError(UNKNOWN_PRESET, f"Unknown CCTV preset {preset_id!r}. Valid: {', '.join(_PRESET_BY_ID)}.")
    return preset


def _is_anamorphic(context: PresetContext) -> bool:
    return context.sample_aspect is not None and context.sample_aspect != SQUARE_PIXELS


def _applies(step: PresetStep, context: PresetContext) -> bool:
    conditions = {
        "always": True,
        "interlaced": context.interlaced,
        "anamorphic": _is_anamorphic(context),
    }
    return conditions[step.when]


def _step_params(step: PresetStep, context: PresetContext) -> dict[str, Any]:
    if step.id == "aspect" and context.sample_aspect is not None:
        num, den = context.sample_aspect
        return {"num": num, "den": den}
    return copy.deepcopy(dict(step.params))


def preset_steps(preset_id: str, lane: Lane, context: PresetContext) -> list[dict[str, Any]]:
    steps = preset_spec(preset_id).steps_for(lane)
    return [{"id": step.id, "params": _step_params(step, context)} for step in steps if _applies(step, context)]


def _step_schema(step: PresetStep) -> dict[str, Any]:
    return {"id": step.id, "params": dict(step.params), "when": step.when}


def _preset_schema(preset: PresetSpec) -> dict[str, Any]:
    return {
        "id": preset.id,
        "labelKey": preset.label_key,
        "label": preset.label,
        "descriptionKey": preset.description_key,
        "description": preset.description,
        "aiUpscaleHint": preset.ai_upscale_hint,
        "lanes": {lane: [_step_schema(step) for step in preset.steps_for(lane)] for lane in LANES},
    }


def presets_schema() -> list[dict[str, Any]]:
    return [_preset_schema(preset) for preset in CCTV_PRESETS]
