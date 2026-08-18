"""Open-Unmix `umxhq`: el unico modelo de CUATRO pistas que se puede publicar.

Separar en bateria/bajo/voz/resto es facil de encontrar y dificil de
redistribuir. Los modelos de calidad alta —BS-RoFormer, Demucs, la familia de
MSST— tienen los pesos sin licencia declarada, con licencia que cubre solo el
codigo, o marcados "solo para uso cientifico". umxhq no: MIT declarado en el
propio record de Zenodo, SOBRE los .pth. Por eso este es el que esta.

Lo que hay que saber antes de elegirlo:

* Es de otra generacion. ~5.4 dB de SDR promedio contra ~9.4 de un RoFormer, y
  esa diferencia SE OYE: deja mas cruce entre pistas. Se ofrece por lo que hace
  (cuatro pistas) y no por como suena comparado con el carril de voz.
* Son CUATRO grafos, uno por pista, de 34 MiB cada uno. Los cuatro o ninguno:
  con tres instalados no hay separacion de cuatro pistas, hay un error.
* Es rapido igual: medido de punta a punta en el port, 19x tiempo real en
  DirectML con las cuatro pistas (unos 13 s por cancion de 4 minutos), contra
  el ~1.2x del RoFormer de una sola pista.
* NO tiene stem residual. El modelo estima las cuatro pistas, asi que no queda
  nada sin explicar; `stems_in_catalog_order` las mapea 0..3 y no calcula resta.

Port propio: github.com/santiquiroz/port-openunmix-onnx (paridad de 109.8 a
135.5 dB SI-SDR contra el separador oficial de upstream). Modulo de datos puro.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from app.services.engines.separation_spec import (
    Architecture,
    SeparationModelSpec,
    SeparationStem,
    STEM_BASS,
    STEM_DRUMS,
    STEM_OTHER,
    STEM_VOCALS,
)

_RELEASE = "https://github.com/santiquiroz/port-openunmix-onnx/releases/download/models-v1.0"

# n_fft/hop de umxhq. No son ajustables: el modelo aprendio su normalizacion
# sobre ESTA resolucion de frecuencia, y cambiarla lo rompe en silencio.
UMX_N_FFT = 4096
UMX_HOP = 1024


@dataclass(frozen=True, slots=True)
class UmxModelSpec(SeparationModelSpec):
    """Un modelo por pista, corriendo juntos como uno solo."""

    # (nombre de pista, archivo, sha256) en el MISMO orden que `stems`.
    graphs: tuple[tuple[str, str, str], ...] = field(default_factory=tuple, kw_only=True)
    n_fft: int = field(default=UMX_N_FFT, kw_only=True)
    hop: int = field(default=UMX_HOP, kw_only=True)

    architecture: ClassVar[Architecture] = "umx"

    @property
    def files(self) -> tuple[str, ...]:
        return tuple(archivo for _, archivo, _ in self.graphs)

    def graph_url(self, filename: str) -> str:
        return f"{_RELEASE}/{filename}"


UMX_MODELS: dict[str, UmxModelSpec] = {
    "umx_4stem": UmxModelSpec(
        id="umx_4stem",
        name="Open-Unmix umxhq",
        # `filename` es el primero de los cuatro: existe porque la base lo pide
        # para el caso normal de un archivo. Lo que decide todo es `files`.
        filename="umxhq_vocals.onnx",
        url=f"{_RELEASE}/umxhq_vocals.onnx",
        sha256="d1297b08e1849f0264da64d995d7489d85e07160b659ccce19b55b7e553ff854",
        primary_stem="Vocals / Drums / Bass / Other",
        category="karaoke",
        description_key="audio.karaoke.model.umx_4stem.description",
        warning_key="audio.karaoke.model.umx_4stem.warning",
        stems=(
            SeparationStem(id="vocals", label_key=STEM_VOCALS, source=0),
            SeparationStem(id="drums", label_key=STEM_DRUMS, source=1),
            SeparationStem(id="bass", label_key=STEM_BASS, source=2),
            SeparationStem(id="other", label_key=STEM_OTHER, source=3),
        ),
        graphs=(
            ("vocals", "umxhq_vocals.onnx", "d1297b08e1849f0264da64d995d7489d85e07160b659ccce19b55b7e553ff854"),
            ("drums", "umxhq_drums.onnx", "b5472fe5f683cfa27e44caa82859984400788e4dd8ea7602a721fa4ea365e8c2"),
            ("bass", "umxhq_bass.onnx", "cb9d14b180ed2e7dcf1c76acfa96cc373f6228c71b80a0f3096e282e9106c040"),
            ("other", "umxhq_other.onnx", "2650bc0def77e52de6ffeb32c5bb6063e84896e80a32cc33e53ebea010a54034"),
        ),
    ),
}
