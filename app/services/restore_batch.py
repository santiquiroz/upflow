from __future__ import annotations

from collections.abc import Mapping
from typing import Any

BATCH_OPTION = "batch"

# Lo que se eligio mirando otra foto: el indice de una cara, un recorte, la geometria o el
# punto gris caerian en cualquier lado. En lote cada foto usa su deteccion automatica.
_PHOTO_BOUND_OPTIONS = ("geometry", "preview_crop")
_PHOTO_BOUND_STEP_OPTIONS = (("faces", "selected"), ("faces", "per_face"), ("tone", "gray_point"))


def is_batch(options: Mapping[str, Any]) -> bool:
    return bool(options.get(BATCH_OPTION))


def batch_conflicts(options: Mapping[str, Any]) -> list[str]:
    top = [name for name in _PHOTO_BOUND_OPTIONS if options.get(name) is not None]
    nested = [
        f"{step}.{name}"
        for step, name in _PHOTO_BOUND_STEP_OPTIONS
        if (options.get(step) or {}).get(name) is not None
    ]
    painted = ["repair.use_user_mask"] if (options.get("repair") or {}).get("use_user_mask") else []
    return [*top, *nested, *painted]


def validate_batch(options: Mapping[str, Any], session: str | None) -> None:
    if not is_batch(options):
        return
    if session is not None:
        raise ValueError("A batch photo is sent as a file, not as an analysis session")
    conflicts = batch_conflicts(options)
    if conflicts:
        raise ValueError(
            "These settings belong to one photo and can't be reused in a batch: " + ", ".join(conflicts)
        )
