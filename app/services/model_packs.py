from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from app.config import resolve_against_project_root
from app.services.generation_installer import _generation_model_id
from app.services.model_registry import ModelStatus

if TYPE_CHECKING:
    from app.services.asr_installer import AsrModelInstaller
    from app.services.generation_converter import GenerationModelConverter
    from app.services.model_registry import ModelRegistry


logger = logging.getLogger(__name__)

MODEL_PACKS = {
    "model-anime": ("John6666/hassaku-xl-illustrious-v31-sdxl", "fp16"),
    "model-photo": ("John6666/epicrealism-xl-vxvi-lastfame-realism-sdxl", "fp16"),
}

# Modelos de voz del instalador: variantes multilingues `_timestamped` (el
# karaoke necesita tiempos por palabra). large-v3-turbo queda AFUERA a
# proposito: su repo no publica el par encoder/decoder fp32 sin fusionar y el
# instalador de ASR lo rechaza — ofrecerlo seria vender una descarga que falla.
ASR_PACKS = {
    "whisper-tiny": "onnx-community/whisper-tiny_timestamped",
    "whisper-base": "onnx-community/whisper-base_timestamped",
    "whisper-small": "onnx-community/whisper-small_timestamped",
}

# El que la app recomienda cuando no hay ninguno: el mejor de los instalables
# para letra cantada. La UI y el instalador señalan al MISMO modelo a proposito.
RECOMMENDED_ASR_REPO = ASR_PACKS["whisper-small"]


def _read_selected_keys(path: Path, catalog: dict) -> tuple[str, ...]:
    if not path.exists():
        return ()
    return tuple(
        key
        for line in path.read_text(encoding="utf-8").splitlines()
        if (key := line.strip()) in catalog
    )


def read_selected_packs(path: Path) -> tuple[str, ...]:
    return _read_selected_keys(path, MODEL_PACKS)


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


def pending_asr_packs(
    selected: Iterable[str],
    registry: ModelRegistry,
) -> tuple[str, ...]:
    from app.services.asr_installer import asr_model_id

    installed_ids = {
        entry.id
        for entry in registry.list()
        if entry.status == ModelStatus.installed
    }
    return tuple(
        key
        for key in selected
        if key in ASR_PACKS and asr_model_id(ASR_PACKS[key]) not in installed_ids
    )


async def enqueue_pending_asr_packs(
    registry: ModelRegistry,
    installer: AsrModelInstaller,
) -> list[str]:
    """Baja los modelos de voz que el usuario tildo en el instalador.

    Mismo contrato que los packs de generacion: la eleccion viaja en
    optional-packs.txt, y ya-instalado se saltea — el primer arranque despues
    de una actualizacion no debe re-bajar gigas que ya estan.
    """
    selected = _read_selected_keys(
        resolve_against_project_root("optional-packs.txt"), ASR_PACKS
    )
    install_ids: list[str] = []
    for key in pending_asr_packs(selected, registry):
        repo_id = ASR_PACKS[key]
        try:
            install_ids.append(await installer.install_from_hf(repo_id))
        except Exception:
            logger.exception(
                "No se pudo encolar el modelo de voz %s (%s)", key, repo_id
            )
    return install_ids


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
