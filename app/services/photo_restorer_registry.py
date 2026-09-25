from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from app.config import Settings
from app.services.capabilities import ResolvedCapability, resolve_one
from app.services.missing_pack import missing_pack_message
from app.services.photo_restore_chain import restore_step

CapabilityResolver = Callable[[str, Settings, object | None], ResolvedCapability]

DSP_CAPABILITY = "image.restore"

# Que capacidad del CATALOG respalda cada paso: la fuente de "¿esta instalado el
# pack?" es la resolucion de capabilities, igual que en restorer_registry.
STEP_CAPABILITY: Mapping[str, str] = MappingProxyType(
    {
        "descreen": DSP_CAPABILITY,
        "repair": "image.restoreModels",
        "deblock": "image.restoreModels",
        "denoise": "image.restoreModels",
        "tone": DSP_CAPABILITY,
        "faces": "image.restoreFaces",
        "colorize": "image.colorize",
    }
)


def capability_for_step(step_id: str, *, uses_model: bool = True) -> str:
    spec = restore_step(step_id)
    if spec.strategy == "model_or_dsp" and not uses_model:
        return DSP_CAPABILITY
    return STEP_CAPABILITY[spec.id]


def validate_step_ready(
    settings: Settings,
    step_id: str,
    *,
    uses_model: bool = True,
    resolve: CapabilityResolver = resolve_one,
) -> None:
    resolved = resolve(capability_for_step(step_id, uses_model=uses_model), settings, None)
    if resolved.missing_packs:
        _raise_missing_step_pack(resolved.missing_packs[0], step_id)
    if resolved.status != "available":
        raise ValueError(f"Restore step {step_id!r} is not available ({resolved.id}: {resolved.status})")


def _raise_missing_step_pack(pack: str, step_id: str) -> None:
    raise ValueError(missing_pack_message(pack, detail=f"Lo pide el paso de restauración {step_id!r}."))
