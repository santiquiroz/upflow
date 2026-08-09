from __future__ import annotations

import errno
import json
from pathlib import Path

from app.services.generation_compat import has_component_weights
from app.services.generation_variants import (
    Precision,
    canonical_weight_name,
    real_available_precisions,
    select_for_precision,
)
from app.services.hf_client import HfClient, HfFile

# ---------------------------------------------------------------------------
# Nucleo compartido de staging/descarga de pipelines diffusers (roadmap #8):
# generation_converter importaba 7 privados de generation_installer y ademas
# duplicaba la descarga de repo-dir. Aca vive lo que ambos (y asr_installer,
# para safe_staging_dest) comparten de verdad.
#
# Promote y registro NO viven aca a proposito: necesitan registry, locks por
# model_id y la validacion funcional con GPU del installer --
# GenerationModelInstaller.validate_and_promote es el seam publico que el
# converter ya usa.
#
# generation_installer re-exporta estos nombres con sus alias historicos
# (_generation_model_id, etc.): routes, model_packs, generation_preflight y
# los tests siguen importando desde alla sin cambios.
# ---------------------------------------------------------------------------

MODEL_INDEX_FILENAME = "model_index.json"


def generation_model_id(
    repo_id: str,
    checkpoint_path: str | None = None,
) -> str:
    model_id = "gen--" + repo_id.lower().replace("/", "--")
    if checkpoint_path is None:
        return model_id
    checkpoint_id = checkpoint_path.lower().replace("\\", "--").replace("/", "--")
    return f"{model_id}--{checkpoint_id}"


def is_inside(candidate: Path, root: Path) -> bool:
    return candidate.resolve().is_relative_to(root.resolve())


def safe_staging_dest(staging_root: Path, relative_path: str) -> Path:
    """Resuelve el destino y verifica que no se escape del staging.

    El path viene del listado del repo o de model_index.json, o sea de la red:
    sin este chequeo un `../..` escribiria fuera del directorio.
    """
    resolved_root = staging_root.resolve()
    dest = (staging_root / relative_path.replace("\\", "/")).resolve()
    if not dest.is_relative_to(resolved_root):
        raise ValueError(f"Archivo del repo escapa el directorio de staging: {relative_path!r}")
    return dest


def read_declared_components(staging_root: Path) -> list[str]:
    """Componentes que el pipeline realmente trae.

    Un slot declarado como [null, null] esta declarado A PROPOSITO VACIO: el
    pipeline lo reconoce pero ese repo no lo incluye. Contarlo como presente
    hacia que la validacion estructural exigiera una carpeta que nunca iba a
    existir, y el install fallaba con "Faltan componentes del pipeline en el
    repo: feature_extractor, image_encoder" sobre repos perfectamente validos
    (reportado 2026-07-29 con UnfilteredAI/NSFW-GEN-ANIME, un SDXL normal).
    """
    index = json.loads((staging_root / MODEL_INDEX_FILENAME).read_text(encoding="utf-8"))
    return [
        name
        for name, value in index.items()
        if not name.startswith("_")
        and isinstance(value, list)
        and any(entry is not None for entry in value)
    ]


def root_checkpoint_paths(files: list[HfFile]) -> list[str]:
    return [
        file.path
        for file in files
        if "/" not in file.path and file.path.lower().endswith(".safetensors")
    ]


def ensure_model_index_listed(files: list[HfFile], repo_id: str) -> None:
    if any(f.path == MODEL_INDEX_FILENAME for f in files):
        return
    # Un repo SIN model_index.json pero CON checkpoints sueltos en la raiz no es un
    # repo invalido: es de archivo unico, y se instala eligiendo cual. Decir solo
    # "falta model_index.json" es cierto y no sirve para nada -- paso con tres repos
    # Flux reales, donde el motivo real era que nadie habia elegido archivo.
    candidates = root_checkpoint_paths(files)
    if candidates:
        listed = ", ".join(candidates[:5])
        more = f" (y {len(candidates) - 5} mas)" if len(candidates) > 5 else ""
        raise ValueError(
            f"El repo {repo_id!r} es de archivo unico: hay que elegir que checkpoint "
            f"instalar. Disponibles: {listed}{more}."
        )
    raise ValueError(
        f"El repo {repo_id!r} no parece un pipeline diffusers ONNX: falta "
        f"{MODEL_INDEX_FILENAME}, y tampoco tiene checkpoints en la raiz para elegir."
    )


def ensure_installable_layout(
    files: list[HfFile],
    repo_id: str,
    checkpoint_path: str | None = None,
) -> None:
    """Rechaza los repos de checkpoints sueltos ANTES de descargar.

    Traen model_index.json pero guardan los pesos en la raiz, sin carpetas por
    componente. Sin este guard el ruteo los daba por validos, la seleccion no
    elegia ni un archivo y el fallo aparecia recien en la validacion
    estructural, con un mensaje que no decia el motivo real.
    """
    if checkpoint_path is not None:
        return
    if not has_component_weights(tuple(f.path for f in files)):
        raise ValueError(
            f"El repo {repo_id!r} guarda los pesos sueltos en la raiz, sin carpetas por "
            "componente (unet, vae, text_encoder...). Es un checkpoint single-file: este "
            "instalador necesita el layout de carpetas de diffusers."
        )


def map_disk_full(exc: OSError) -> str | None:
    if exc.errno != errno.ENOSPC:
        return None
    target = getattr(exc, "filename", None)
    where = f" en {target}" if target else ""
    return (
        f"No queda espacio en disco{where}. Liberá espacio y volvé a intentar; "
        "la descarga parcial ya se limpió."
    )


async def download_repo_dir(
    hf_client: HfClient,
    repo_id: str,
    dest_root: Path,
    precision: Precision,
    files: list[HfFile] | None = None,
) -> tuple[list[HfFile], Precision]:
    """Descarga un repo diffusers completo (componentes declarados, con la
    precisión pedida si el repo la ofrece) a un directorio local.

    `files` opcional: el camino normal de conversión ya listó el repo (y sus
    tests cuentan esas llamadas) -- re-listar acá cambiaría el conteo y sumaría
    una llamada de red. Sin `files`, lista y valida el layout acá (camino del
    merge de inpainting, que baja repos base/inpaint ajenos al job).
    """
    if files is None:
        files = await hf_client.repo_files(repo_id)
        ensure_model_index_listed(files, repo_id)
        ensure_installable_layout(files, repo_id)
    await hf_client.download(
        repo_id,
        MODEL_INDEX_FILENAME,
        safe_staging_dest(dest_root, MODEL_INDEX_FILENAME),
        unlimited=True,
    )
    declared = read_declared_components(dest_root)
    offered = await real_available_precisions(hf_client, repo_id, files, declared)
    chosen = precision if precision in offered else (offered[0] if offered else "fp32")
    selected = select_for_precision(files, declared, chosen)
    for hf_file in selected:
        dest = safe_staging_dest(dest_root, canonical_weight_name(hf_file.path))
        dest.parent.mkdir(parents=True, exist_ok=True)
        await hf_client.download(repo_id, hf_file.path, dest, unlimited=True)
    return selected, chosen
