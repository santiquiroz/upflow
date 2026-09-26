from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from app.services.photo_restore_chain import step_ids, steps_from_selection

ToneKind = Literal["mono", "toned", "hand_tinted", "color"]
PresetCondition = Literal[
    "damage",
    "heavy_damage",
    "jpeg_blocking",
    "noise",
    "halftone",
    "faded_cast",
    "strong_faded_cast",
    "restorable_faces",
]

# Umbrales de §2.3 y §3.2, todos [propuesta]: se recalibran en P4-BENCH.
DAMAGE_MIN_COVERAGE = 0.003
HEAVY_DAMAGE_MIN_COVERAGE = 0.03
NOISE_MIN_SIGMA = 2.5 / 255
FADED_CAST_MIN_DE = 4.0
STRONG_FADED_CAST_MIN_DE = 8.0
RESTORABLE_FACE_MIN_EYE_PX = 32.0

REPAIR_SENSITIVITY_MEDIUM = 0.5
REPAIR_SENSITIVITY_HIGH = 0.75
KEEP_GRAIN_DEFAULT = 0.25
FACE_MODEL_DEFAULT = "gfpgan-v1.4"
DEFAULT_PRESET = "gentle"


@dataclass(frozen=True, slots=True)
class PhotoFacts:
    damage_coverage: float = 0.0
    jpeg_blocking: bool = False
    noise_sigma: float = 0.0
    halftone: bool = False
    tone_kind: ToneKind = "color"
    color_cast_de: float = 0.0
    max_eye_px: float = 0.0


@dataclass(frozen=True, slots=True)
class PresetStep:
    step_id: str
    options: Mapping[str, Any]
    when: PresetCondition | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@dataclass(frozen=True, slots=True)
class PhotoPreset:
    id: str
    steps: tuple[PresetStep, ...]
    suggest_when: PresetCondition | None = None

    @property
    def label_key(self) -> str:
        return f"restore.preset.{self.id}"

    @property
    def description_key(self) -> str:
        return f"restore.preset.{self.id}.description"

    def step_ids(self) -> tuple[str, ...]:
        return tuple(step.step_id for step in self.steps)


@dataclass(frozen=True, slots=True)
class PresetSelection:
    preset: str
    steps: tuple[str, ...]
    options: dict[str, dict[str, Any]]


class UnknownPhotoPreset(ValueError):
    pass


def _is_faded_color_photo(facts: PhotoFacts, min_de: float) -> bool:
    # Un sepia o un virado tambien se alejan del eje neutro: corregirlos borraria su tono.
    return facts.tone_kind == "color" and facts.color_cast_de > min_de


PRESET_CONDITIONS: Mapping[PresetCondition, Callable[[PhotoFacts], bool]] = MappingProxyType(
    {
        "damage": lambda facts: facts.damage_coverage > DAMAGE_MIN_COVERAGE,
        "heavy_damage": lambda facts: facts.damage_coverage > HEAVY_DAMAGE_MIN_COVERAGE,
        "jpeg_blocking": lambda facts: facts.jpeg_blocking,
        "noise": lambda facts: facts.noise_sigma > NOISE_MIN_SIGMA,
        "halftone": lambda facts: facts.halftone,
        "faded_cast": lambda facts: _is_faded_color_photo(facts, FADED_CAST_MIN_DE),
        "strong_faded_cast": lambda facts: _is_faded_color_photo(facts, STRONG_FADED_CAST_MIN_DE),
        "restorable_faces": lambda facts: facts.max_eye_px >= RESTORABLE_FACE_MIN_EYE_PX,
    }
)


def _repair(sensitivity: float, grow_px: int, when: PresetCondition | None = None) -> PresetStep:
    return PresetStep("repair", {"engine": "fast", "sensitivity": sensitivity, "grow_px": grow_px}, when)


def _denoise(strength: float, when: PresetCondition | None = None) -> PresetStep:
    return PresetStep("denoise", {"strength": strength, "keep_grain": KEEP_GRAIN_DEFAULT}, when)


def _fix_faded(strength: float) -> PresetStep:
    options = {"strength": strength, "keep_tone": True, "neutral_gray": False, "fix_faded": True}
    return PresetStep("tone", options, "faded_cast")


_GENTLE_STEPS: tuple[PresetStep, ...] = (
    _repair(REPAIR_SENSITIVITY_MEDIUM, grow_px=1, when="damage"),
    PresetStep("deblock", {"strength": 0.5}, "jpeg_blocking"),
    _denoise(0.3, when="noise"),
    _fix_faded(0.7),
)

# El orden de la tabla es el de la UI; ninguno colorea ni neutraliza (§2.3).
PHOTO_PRESETS: tuple[PhotoPreset, ...] = (
    PhotoPreset("gentle", _GENTLE_STEPS),
    PhotoPreset(
        "heavy_damage",
        (_repair(REPAIR_SENSITIVITY_HIGH, grow_px=2), _denoise(0.5), _fix_faded(0.7)),
        suggest_when="heavy_damage",
    ),
    PhotoPreset(
        "newspaper",
        (PresetStep("descreen", {"mode": "halftone", "strength": 1.0}), _denoise(0.2)),
        suggest_when="halftone",
    ),
    PhotoPreset("faded_color_print", (_denoise(0.3), _fix_faded(0.85)), suggest_when="strong_faded_cast"),
    PhotoPreset(
        "portrait",
        (*_GENTLE_STEPS, PresetStep("faces", {"model": FACE_MODEL_DEFAULT, "blend": 0.6}, "restorable_faces")),
        suggest_when="restorable_faces",
    ),
)

# Una trama sin descreen se vuelve moire en cualquier paso siguiente, por eso gana
# al dano; "portrait" no esta: se sugiere pero nunca se propone solo (§2.3).
PROPOSAL_PRIORITY: tuple[str, ...] = ("newspaper", "heavy_damage", "faded_color_print")


# Sin analisis no se midio la dominante: "Fix faded" le quitaria el tono a una foto virada o iluminada (§2.3).
UNMEASURED_WITHOUT_ANALYSIS: frozenset[PresetCondition] = frozenset({"faded_cast", "strong_faded_cast"})


def options_without_analysis(step: PresetStep) -> dict[str, Any]:
    options = dict(step.options)
    if step.when in UNMEASURED_WITHOUT_ANALYSIS and "fix_faded" in options:
        options["fix_faded"] = False
    return options


def photo_preset(preset_id: str) -> PhotoPreset:
    for preset in PHOTO_PRESETS:
        if preset.id == preset_id:
            return preset
    valid = ", ".join(preset.id for preset in PHOTO_PRESETS)
    raise UnknownPhotoPreset(f"Unknown photo preset: {preset_id!r}. Valid presets: {valid}.")


def holds(condition: PresetCondition | None, facts: PhotoFacts) -> bool:
    return condition is None or PRESET_CONDITIONS[condition](facts)


def resolve_preset(preset_id: str, facts: PhotoFacts) -> PresetSelection:
    active = [step for step in photo_preset(preset_id).steps if holds(step.when, facts)]
    ordered = step_ids(steps_from_selection([step.step_id for step in active]))
    options = {step.step_id: dict(step.options) for step in active}
    return PresetSelection(preset=preset_id, steps=ordered, options=options)


def suggested_presets(facts: PhotoFacts) -> tuple[str, ...]:
    return tuple(
        preset.id
        for preset in PHOTO_PRESETS
        if preset.suggest_when is not None and holds(preset.suggest_when, facts)
    )


def proposed_preset(facts: PhotoFacts) -> str:
    suggested = set(suggested_presets(facts))
    return next((preset_id for preset_id in PROPOSAL_PRIORITY if preset_id in suggested), DEFAULT_PRESET)
