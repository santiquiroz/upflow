from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from app.config import resolve_against_project_root
from app.services.generation_installer import _generation_model_id
from app.services.model_registry import ModelStatus

if TYPE_CHECKING:
    from app.services.generation_converter import GenerationModelConverter
    from app.services.model_registry import ModelRegistry


logger = logging.getLogger(__name__)

MODEL_PACKS = {
    "model-anime": ("John6666/hassaku-xl-illustrious-v31-sdxl", "fp16"),
    "model-photo": ("John6666/epicrealism-xl-vxvi-lastfame-realism-sdxl", "fp16"),
}


def read_selected_packs(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    return tuple(
        key
        for line in path.read_text(encoding="utf-8").splitlines()
        if (key := line.strip()) in MODEL_PACKS
    )


def pending_model_packs(
    selected: Iterable[str],
    registry: ModelRegistry,
) -> tuple[str, ...]:
    installed_ids = {
        entry.id
        for entry in registry.list()
        if entry.status == ModelStatus.installed
    }
    return tuple(
        key
        for key in selected
        if key in MODEL_PACKS
        and _generation_model_id(MODEL_PACKS[key][0], None) not in installed_ids
    )


async def enqueue_pending_model_packs(
    registry: ModelRegistry,
    converter: GenerationModelConverter,
) -> list[str]:
    selected = read_selected_packs(
        resolve_against_project_root("optional-packs.txt")
    )
    pending = pending_model_packs(selected, registry)
    conversion_ids: list[str] = []

    for key in pending:
        repo_id, precision = MODEL_PACKS[key]
        try:
            conversion_id = await converter.convert_from_hf(
                repo_id,
                precision=precision,
            )
        except Exception:
            logger.exception(
                "No se pudo encolar el pack de modelo %s (%s)",
                key,
                repo_id,
            )
            continue
        conversion_ids.append(conversion_id)

    return conversion_ids
