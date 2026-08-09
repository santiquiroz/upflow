"""Catalogo de modelos MDX-Net de separacion voz/instrumental.

Solo modelos publicados por el propio equipo de Ultimate Vocal Remover
(Anjok07 & aufr33, MIT) — nada de pesos de terceros con licencia dudosa.
Los parametros por modelo salen de model_data_new.json del repo
TRvlvr/application_data, indexado por el hash UVR (MD5 de los ultimos
10000 KiB del .onnx); ambos hashes se verificaron contra los archivos
reales el 2026-08-09.

Modulo de datos puro (sin imports de app.*): lo consumen config, engine,
rutas y el script de descarga sin riesgo de ciclos.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MDX_SAMPLE_RATE = 44100
MDX_HOP = 1024


@dataclass(frozen=True, slots=True)
class MdxModelSpec:
    id: str
    # Nombre propio del modelo: se muestra tal cual, no se traduce.
    name: str
    filename: str
    url: str
    # Hash UVR: MD5 de los ultimos 10000 KiB del archivo.
    uvr_hash: str
    n_fft: int
    dim_f: int
    dim_t: int
    compensate: float
    # Que stem SACA el modelo ("Instrumental" | "Vocals"); el otro se obtiene
    # restando: secundario = mezcla - primario * compensate.
    primary_stem: str

    @property
    def chunk_samples(self) -> int:
        return MDX_HOP * (self.dim_t - 1)

    @property
    def trim_samples(self) -> int:
        return self.n_fft // 2

    @property
    def gen_samples(self) -> int:
        return self.chunk_samples - 2 * self.trim_samples


_RELEASE_BASE = (
    "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models"
)

DEFAULT_SEPARATION_MODEL = "inst_hq_3"

SEPARATION_MODELS: dict[str, MdxModelSpec] = {
    "inst_hq_3": MdxModelSpec(
        id="inst_hq_3",
        name="MDX-Net Inst HQ 3",
        filename="UVR-MDX-NET-Inst_HQ_3.onnx",
        url=f"{_RELEASE_BASE}/UVR-MDX-NET-Inst_HQ_3.onnx",
        uvr_hash="55657dd70583b0fedfba5f67df11d711",
        n_fft=6144,
        dim_f=3072,
        dim_t=256,
        compensate=1.022,
        primary_stem="Instrumental",
    ),
    "voc_ft": MdxModelSpec(
        id="voc_ft",
        name="MDX-Net Voc FT",
        filename="UVR-MDX-NET-Voc_FT.onnx",
        url=f"{_RELEASE_BASE}/UVR-MDX-NET-Voc_FT.onnx",
        uvr_hash="77d07b2667ddf05b9e3175941b4454a0",
        n_fft=7680,
        dim_f=3072,
        dim_t=256,
        compensate=1.021,
        primary_stem="Vocals",
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
