"""Catalogo de modelos MDX-Net de separacion de stems (karaoke y limpieza).

Politica de origen: modelos distribuidos por el canal oficial de descargas de
Ultimate Vocal Remover (el Download Center de la app), con credito por autor —
sea el equipo core (Anjok07 & aufr33, MIT) o contribuidores comunitarios como
FoxJoy (Reverb HQ). Nada de pesos que UVR no distribuya oficialmente.
Los parametros por modelo salen de model_data_new.json del repo
TRvlvr/application_data, indexado por el hash UVR (MD5 de los ultimos
10000 KiB del .onnx); los hashes se verificaron contra los archivos
reales el 2026-08-09.

Modulo de datos puro (sin imports de app.*): lo consumen config, engine,
rutas y el script de descarga sin riesgo de ciclos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

MDX_SAMPLE_RATE = 44100
MDX_HOP = 1024


@dataclass(frozen=True, slots=True)
class MdxStem:
    # Id que viaja en download?stem= y en las respuestas de la API.
    id: str
    # La copia la traduce el frontend; el backend solo manda la clave.
    label_key: str
    # De donde sale el audio: "primary" = lo que el modelo infiere;
    # "secondary" = la resta compensada (mezcla - primario * compensate).
    source: Literal["primary", "secondary"]


@dataclass(frozen=True, slots=True)
class MdxModelSpec:
    id: str
    # Nombre propio del modelo: se muestra tal cual, no se traduce.
    name: str
    filename: str
    url: str
    # Hash UVR: MD5 de los ultimos 10000 KiB del archivo.
    uvr_hash: str
    # SHA-256 del archivo completo: el hash UVR solo cubre la cola y el mirror
    # TRvlvr/model_repo es de terceros; el script de descarga pinea ambos.
    sha256: str
    n_fft: int
    dim_f: int
    dim_t: int
    compensate: float
    # Que stem SACA el modelo (nombre UVR: "Instrumental" | "Vocals" |
    # "Reverb"); el otro se obtiene restando:
    # secundario = mezcla - primario * compensate.
    primary_stem: str
    # "karaoke" (separar voz/instrumental) o "cleanup" (pasada de limpieza,
    # p.ej. quitar reverb). La UI agrupa el picker con esto.
    category: str
    # Que hace el modelo, dicho para el usuario; el frontend traduce.
    description_key: str
    # Stems de cara al usuario, ORDENADOS: el primero es el que el usuario
    # quiere (lo sirve downloadUrl); el segundo va en ?stem=<id>.
    stems: tuple[MdxStem, MdxStem]

    @property
    def chunk_samples(self) -> int:
        return MDX_HOP * (self.dim_t - 1)

    @property
    def trim_samples(self) -> int:
        return self.n_fft // 2

    @property
    def gen_samples(self) -> int:
        return self.chunk_samples - 2 * self.trim_samples

    @property
    def main_stem(self) -> MdxStem:
        return self.stems[0]

    @property
    def other_stem(self) -> MdxStem:
        return self.stems[1]

    def stem_ids(self) -> tuple[str, str]:
        return (self.stems[0].id, self.stems[1].id)


_RELEASE_BASE = (
    "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models"
)

DEFAULT_SEPARATION_MODEL = "inst_hq_3"

_STEM_INSTRUMENTAL = "audio.stem.instrumental"
_STEM_VOCALS = "audio.stem.vocals"

SEPARATION_MODELS: dict[str, MdxModelSpec] = {
    "inst_hq_3": MdxModelSpec(
        id="inst_hq_3",
        name="MDX-Net Inst HQ 3",
        filename="UVR-MDX-NET-Inst_HQ_3.onnx",
        url=f"{_RELEASE_BASE}/UVR-MDX-NET-Inst_HQ_3.onnx",
        uvr_hash="55657dd70583b0fedfba5f67df11d711",
        sha256="317554b07fe1ea5279a77f2b1520a41ea4b93432560c4ffd08792c30fddf9adc",
        n_fft=6144,
        dim_f=3072,
        dim_t=256,
        compensate=1.022,
        primary_stem="Instrumental",
        category="karaoke",
        description_key="audio.karaoke.model.inst_hq_3.description",
        stems=(
            MdxStem("instrumental", _STEM_INSTRUMENTAL, "primary"),
            MdxStem("vocals", _STEM_VOCALS, "secondary"),
        ),
    ),
    "voc_ft": MdxModelSpec(
        id="voc_ft",
        name="MDX-Net Voc FT",
        filename="UVR-MDX-NET-Voc_FT.onnx",
        url=f"{_RELEASE_BASE}/UVR-MDX-NET-Voc_FT.onnx",
        uvr_hash="77d07b2667ddf05b9e3175941b4454a0",
        sha256="534b2070fcc7df514b13ef660dc8cbb328679c2374d04354a5c42bb14ecce111",
        n_fft=7680,
        dim_f=3072,
        dim_t=256,
        compensate=1.021,
        primary_stem="Vocals",
        category="karaoke",
        description_key="audio.karaoke.model.voc_ft.description",
        stems=(
            MdxStem("instrumental", _STEM_INSTRUMENTAL, "secondary"),
            MdxStem("vocals", _STEM_VOCALS, "primary"),
        ),
    ),
    # Limpieza post-karaoke: el modelo saca la COLA DE REVERB (wet); la pista
    # limpia que el usuario quiere es la resta (dry), por eso va primera.
    "reverb_hq": MdxModelSpec(
        id="reverb_hq",
        name="Reverb HQ by FoxJoy",
        filename="Reverb_HQ_By_FoxJoy.onnx",
        url=f"{_RELEASE_BASE}/Reverb_HQ_By_FoxJoy.onnx",
        uvr_hash="cd5b2989ad863f116c855db1dfe24e39",
        sha256="233bb5c6aaa365e568659a0a81211746fa881f8f47f82d9e864fce1f7692db80",
        n_fft=6144,
        dim_f=3072,
        dim_t=512,
        compensate=1.035,
        primary_stem="Reverb",
        category="cleanup",
        description_key="audio.karaoke.model.reverb_hq.description",
        stems=(
            MdxStem("dry", "audio.stem.dry", "secondary"),
            MdxStem("wet", "audio.stem.wet", "primary"),
        ),
    ),
}


def model_file(model_dir: Path, model_id: str) -> Path:
    return model_dir / SEPARATION_MODELS[model_id].filename


def installed_model_ids(model_dir: Path) -> list[str]:
    return [
        model_id
        for model_id in SEPARATION_MODELS
        if model_file(model_dir, model_id).exists()
    ]


def first_installed_model_path(model_dir: Path) -> Path | None:
    for model_id in installed_model_ids(model_dir):
        return model_file(model_dir, model_id)
    return None
