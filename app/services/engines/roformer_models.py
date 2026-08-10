"""Modelos RoFormer del catalogo de separacion: el carril de MAXIMA CALIDAD.

Politica de origen: los pesos son de KimberleyJensen (KimberleyJSN), MIT
declarado en la model card de HuggingFace; el .onnx es un port propio, publico
y MIT (github.com/santiquiroz/port-bs-roformer-onnx, release models-v1.0),
conversion mecanica de formato sin re-entrenar nada.

POR QUE ESTE CHECKPOINT Y NO OTRO. Es el roformer vocal con mejor SDR que trae
licencia permisiva EXPLICITA (10.98 en el multisong de MSST). Los mas conocidos
no la traen: viperx BS-RoFormer (SDR 10.87) vive en un repo sin LICENSE, anvuew
es GPL-3, Sucial y becruily son CC-BY-NC, y unwa/gabox — los de SDR mas alto —
no declaran absolutamente nada. Asi que aca gano el unico que ademas puntua
mejor, no hubo que elegir entre licencia y calidad.

TRES cosas lo separan del resto del catalogo:

* NO es el default y no debe serlo. Cuesta ~20x lo que Inst HQ 3 DE PUNTA A
  PUNTA: medido en una RX 7800 XT (DirectML, sesion caliente, 24 s de audio)
  20.0 s contra 1.02 s, o sea 50 s de GPU por minuto de audio contra 2.5 s.
  Ojo con el numero del port, que es OTRO: alla el grafo mide 1852 ms por chunk
  de 8 s = 4.32x tiempo real contra ~38x de un grafo MDX, "~9x mas caro". Esa
  comparacion es de grafo contra grafo; el pipeline entero duplica la brecha
  porque estos chunks van solapados al 50% (cada segundo de audio pasa por el
  grafo DOS veces) y los de MDX no. El que le importa al usuario es el de punta
  a punta, y es el que va en la advertencia de la UI. Mismo lugar de producto
  que GMFSS en video: la opcion "maxima calidad, lento", elegida a proposito.
  En CPU da 0.69x (mas lento que reproducirlo), o sea que no es un carril viable.
* El eje temporal es FIJO. El rotary embedding cachea su tabla de frecuencias
  por seq_len, asi que el chunk quedo horneado al trazar el grafo: `chunk_size`
  no es un parametro que se pueda tocar, tiene que coincidir con el .onnx.
* El stem residual es una resta SIMPLE. El modelo infiere UN stem (vocals) y el
  otro es `mezcla - vocals`, sin el factor `compensate` que MDX-Net si necesita.

El catalogo COMPLETO (MDX + VR + RoFormer) se arma en separation_models.py;
este modulo solo aporta la parte RoFormer. Modulo de datos puro (sin imports de
app.services fuera de separation_spec): los numeros del grafo se declaran aca y
el motor arma con ellos el `RoformerSpec` del driver vendorizado, con un test
anti-drift que compara ambos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from app.services.engines.separation_spec import (
    STEM_INSTRUMENTAL,
    STEM_VOCALS,
    Architecture,
    SeparationModelSpec,
    SeparationStem,
)

ROFORMER_SAMPLE_RATE = 44100


@dataclass(frozen=True, slots=True)
class RoformerModelSpec(SeparationModelSpec):
    # Parametros del grafo, copiados del manifest.json del release. No son
    # tunables: el .onnx se trazo con estos y no acepta otros.
    n_fft: int
    hop_length: int
    # Muestras por chunk. FIJO al trazar (rotary embedding), no configurable.
    chunk_size: int
    # Chunks solapados al 50% (MSST demix con batch_size=1).
    num_overlap: int
    # Stems que emite el grafo, en el orden en que los apila. Es la lista del
    # DRIVER, no la del usuario: `stems` (la del supertipo) es la de la UI.
    graph_stems: tuple[str, ...]
    # Peso del grafo fp32 en disco/VRAM.
    graph_mb: int
    # Intermedio de atencion vivo a T=801. Es admision real de memoria, no un
    # detalle: sin esto libre el modelo no arranca, y el mensaje tiene que
    # decirlo antes de empezar a procesar.
    intermediate_mb: int

    architecture: ClassVar[Architecture] = "roformer"

    @property
    def frames(self) -> int:
        """T del grafo: el eje temporal horneado."""
        return self.chunk_size // self.hop_length + 1

    @property
    def chunk_seconds(self) -> float:
        return self.chunk_size / ROFORMER_SAMPLE_RATE

    @property
    def required_free_mb(self) -> int:
        """Memoria libre que hay que tener en el device para cargarlo y correrlo."""
        return self.graph_mb + self.intermediate_mb


_RELEASE_BASE = (
    "https://github.com/santiquiroz/port-bs-roformer-onnx/releases/download/models-v1.0"
)

ROFORMER_MODELS: dict[str, RoformerModelSpec] = {
    "mel_band_roformer_kim": RoformerModelSpec(
        id="mel_band_roformer_kim",
        name="Mel-Band RoFormer by KimberleyJSN",
        filename="mel_band_roformer_kim_T801.onnx",
        url=f"{_RELEASE_BASE}/mel_band_roformer_kim_T801.onnx",
        sha256="1b8afd7780d8a234527748821dee6bc746d346f2088751c12fd48e8c873f625a",
        n_fft=2048,
        hop_length=441,
        chunk_size=352800,
        num_overlap=2,
        graph_stems=("vocals",),
        graph_mb=931,
        intermediate_mb=1300,
        primary_stem="Vocals",
        category="karaoke",
        description_key="audio.karaoke.model.mel_band_roformer_kim.description",
        # La UI la muestra SIEMPRE, no solo con el modelo ya elegido: descubrir
        # el 20x cuando el trabajo ya arranco no es una eleccion informada.
        warning_key="audio.karaoke.model.mel_band_roformer_kim.warning",
        # Mismo par que voc_ft, y por el mismo motivo: el modelo saca la voz,
        # pero en karaoke lo que el usuario se lleva primero es la instrumental.
        stems=(
            SeparationStem("instrumental", STEM_INSTRUMENTAL, "secondary"),
            SeparationStem("vocals", STEM_VOCALS, "primary"),
        ),
    ),
}
