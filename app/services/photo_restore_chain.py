from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

RestorePhase = Literal["native", "output"]
RestoreStrategy = Literal["dsp", "model", "model_or_dsp"]
DescreenMode = Literal["auto", "halftone", "texture"]

# Por encima de esto, un denoise despues del pasabajos del halftone empieza a borronear
# dos veces la misma foto [propuesta, se recalibra en P4-BENCH].
HALFTONE_DENOISE_LIMIT = 0.3

CORE_PACK = "restore-core"
FACES_PACK = "restore-faces"
COLORIZE_PACK = "restore-colorize"


@dataclass(frozen=True, slots=True)
class RestoreStepSpec:
    id: str
    phase: RestorePhase
    family: str
    covers: tuple[str, ...]
    strategy: RestoreStrategy
    label_key: str
    description_key: str
    invents_detail: bool
    warning_key: str | None
    pack: str | None


@dataclass(frozen=True, slots=True)
class ToneFlags:
    keep_tone: bool
    neutral_gray: bool


def _step(
    step_id: str,
    phase: RestorePhase,
    strategy: RestoreStrategy,
    *,
    pack: str | None = None,
    warning_key: str | None = None,
) -> RestoreStepSpec:
    return RestoreStepSpec(
        id=step_id,
        phase=phase,
        family=step_id,
        covers=(step_id,),
        strategy=strategy,
        label_key=f"restore.step.{step_id}",
        description_key=f"restore.step.{step_id}.description",
        invents_detail=warning_key is not None,
        warning_key=warning_key,
        pack=pack,
    )


# El orden de la tupla es el orden de ejecucion (spec §3.3); el reescalado va entre
# tone y faces, pero es el motor SR existente y no un paso de esta cadena.
RESTORE_CHAIN: tuple[RestoreStepSpec, ...] = (
    _step("descreen", "native", "dsp"),
    _step("repair", "native", "model_or_dsp", pack=CORE_PACK, warning_key="restore.warning.inventsDetail"),
    _step("deblock", "native", "model", pack=CORE_PACK),
    _step("denoise", "native", "model", pack=CORE_PACK),
    _step("tone", "native", "dsp"),
    _step("faces", "output", "model", pack=FACES_PACK, warning_key="restore.warning.faces"),
    _step("colorize", "output", "model", pack=COLORIZE_PACK, warning_key="restore.warning.colorize"),
)

_FAMILY_IN_MESSAGE: dict[str, str] = {
    "descreen": "print pattern removal",
    "repair": "damage repair",
    "deblock": "JPEG artifact removal",
    "denoise": "noise reduction",
    "tone": "color and tone",
    "faces": "face restoration",
    "colorize": "colorization",
}


class UnknownRestoreStep(ValueError):
    pass


class RedundantRestoreSelection(ValueError):
    pass


def step_ids(specs: Sequence[RestoreStepSpec]) -> tuple[str, ...]:
    return tuple(spec.id for spec in specs)


def restore_step(step_id: str, chain: Sequence[RestoreStepSpec] = RESTORE_CHAIN) -> RestoreStepSpec:
    for spec in chain:
        if spec.id == step_id:
            return spec
    raise UnknownRestoreStep(_unknown_message([step_id], chain))


def steps_from_selection(
    selected_ids: Sequence[str],
    chain: Sequence[RestoreStepSpec] = RESTORE_CHAIN,
) -> list[RestoreStepSpec]:
    known = set(step_ids(chain))
    unknown = [step_id for step_id in selected_ids if step_id not in known]
    if unknown:
        raise UnknownRestoreStep(_unknown_message(unknown, chain))
    selected = set(selected_ids)
    # El orden sale del catalogo y no del request: la cadena tiene causalidad (§3.3).
    specs = [spec for spec in chain if spec.id in selected]
    conflict = _first_conflict(specs)
    if conflict is not None:
        raise RedundantRestoreSelection(_redundant_message(*conflict))
    return specs


def is_overprocessing(
    selected_ids: Sequence[str],
    descreen_mode: DescreenMode,
    denoise_strength: float,
) -> bool:
    halftone = "descreen" in selected_ids and descreen_mode == "halftone"
    return halftone and "denoise" in selected_ids and denoise_strength > HALFTONE_DENOISE_LIMIT


def resolve_tone_flags(
    selected_ids: Sequence[str],
    keep_tone: bool = True,
    neutral_gray: bool = False,
) -> ToneFlags:
    # Colorizar reemplaza la croma entera, asi que no hay tono original que conservar.
    keeps = keep_tone and not neutral_gray and "colorize" not in selected_ids
    return ToneFlags(keep_tone=keeps, neutral_gray=neutral_gray)


def _first_conflict(
    specs: Sequence[RestoreStepSpec],
) -> tuple[RestoreStepSpec, RestoreStepSpec] | None:
    for index, first in enumerate(specs):
        for second in specs[index + 1 :]:
            if set(first.covers) & set(second.covers):
                return first, second
    return None


def _unknown_message(unknown: Sequence[str], chain: Sequence[RestoreStepSpec]) -> str:
    valid = ", ".join(step_ids(chain))
    return f"Unknown restore steps: {', '.join(sorted(set(unknown)))}. Valid steps: {valid}."


def _redundant_message(first: RestoreStepSpec, second: RestoreStepSpec) -> str:
    shared = [family for family in first.covers if family in second.covers]
    tasks = ", ".join(_FAMILY_IN_MESSAGE.get(family, family) for family in shared)
    return (
        f"Redundant restore steps: {first.id!r} and {second.id!r} do the same job ({tasks}). "
        "Pick one: running both would process the photo twice for the same fix."
    )
