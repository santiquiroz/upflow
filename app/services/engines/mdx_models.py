"""Modelos MDX-Net del catalogo de separacion de stems (karaoke y limpieza).

Politica de origen: modelos distribuidos por el canal oficial de descargas de
Ultimate Vocal Remover (el Download Center de la app), con credito por autor —
sea el equipo core (Anjok07 & aufr33, MIT) o contribuidores comunitarios como
FoxJoy (Reverb HQ). Nada de pesos que UVR no distribuya oficialmente.
Los parametros por modelo salen de model_data_new.json del repo
TRvlvr/application_data, indexado por el hash UVR (MD5 de los ultimos
10000 KiB del .onnx); los hashes se verificaron contra los archivos
reales el 2026-08-09.

El catalogo COMPLETO (MDX + VR) se arma en separation_models.py; este modulo
solo aporta la mitad MDX. Modulo de datos puro (sin imports de app.services
fuera de separation_spec).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from app.services.engines.separation_spec import (
    RESIDUAL,
    STEM_BACKING_VOCALS,
    STEM_INSTRUMENTAL,
    STEM_LEAD_VOCALS,
    STEM_VOCALS,
    Architecture,
    SeparationModelSpec,
    SeparationStem,
)

MDX_SAMPLE_RATE = 44100
MDX_HOP = 1024


@dataclass(frozen=True, slots=True)
class MdxModelSpec(SeparationModelSpec):
    # Hash UVR: MD5 de los ultimos 10000 KiB del archivo. El hash UVR solo cubre
    # la cola y el mirror TRvlvr/model_repo es de terceros, por eso el catalogo
    # pinea ADEMAS el sha256 de la clase base.
    uvr_hash: str
    n_fft: int
    dim_f: int
    dim_t: int
    # secundario = mezcla - primario * compensate.
    compensate: float

    architecture: ClassVar[Architecture] = "mdx"

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

MDX_MODELS: dict[str, MdxModelSpec] = {
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
            SeparationStem("instrumental", STEM_INSTRUMENTAL, 0),
            SeparationStem("vocals", STEM_VOCALS, RESIDUAL),
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
            SeparationStem("instrumental", STEM_INSTRUMENTAL, RESIDUAL),
            SeparationStem("vocals", STEM_VOCALS, 0),
        ),
    ),
    # Karaoke con armonias (F2a): KARA_2 saca la voz LIDER y deja los coros
    # DENTRO de la pista instrumental — el karaoke conserva las armonias de la
    # banda. Mirror Politrees byte-identico (sha256 verificado 2026-08-28
    # contra el archivo real); el release original TRvlvr no declara licencia,
    # el re-host lo publica como MIT. Los parametros salen de
    # model_data_new.json de TRvlvr/application_data, keyed por el hash UVR
    # (KARA_2 no trae yaml): dim_t 8 ahi significa 2^8 = 256 cuadros.
    "kara_2": MdxModelSpec(
        id="kara_2",
        name="MDX-Net Karaoke 2",
        filename="UVR_MDXNET_KARA_2.onnx",
        url=(
            "https://huggingface.co/Politrees/UVR_resources/resolve/main/"
            "models/MDXNet/UVR_MDXNET_KARA_2.onnx"
        ),
        uvr_hash="1d64a6d2c30f709b8c9b4ce1366d96ee",
        sha256="bf32e15105a09c0f7dddd2b67346146334d6f3ecb399ed7638eba2ab07cbf5f4",
        n_fft=5120,
        dim_f=2048,
        dim_t=256,
        compensate=1.065,
        primary_stem="Instrumental",
        category="karaoke",
        description_key="audio.karaoke.model.kara_2.description",
        stems=(
            SeparationStem("backing_vocals", STEM_BACKING_VOCALS, 0),
            SeparationStem("lead_vocals", STEM_LEAD_VOCALS, RESIDUAL),
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
            SeparationStem("dry", "audio.stem.dry", RESIDUAL),
            SeparationStem("wet", "audio.stem.wet", 0),
        ),
    ),
}
