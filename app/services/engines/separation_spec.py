"""Forma COMPARTIDA de una entrada del catalogo de separacion de stems.

El catalogo tiene dos arquitecturas de modelo con parametros incompatibles:
MDX-Net (n_fft/dim_f/dim_t/compensate, un stem inferido + resta compensada) y
VR 5.1 CascadedNet (4band_v3, nout, aggression, mascara complementaria). Lo que
NO cambia entre ellas es todo lo que consumen el catalogo, la API, el pack de
descarga y el job: id, nombre propio, archivo, url, sha256, categoria, copia y
el PAR ORDENADO de stems.

Por eso esto es un supertipo y no dos catalogos sueltos: `separation_models.py`,
`routes.py`, `pack_provisioner.py` y `audio_job_manager.py` operan SOLO sobre
estos campos, y con un tipo base compartido no pueden desincronizarse entre
arquitecturas. Los campos propios de cada motor viven en la subclase y solo los
toca su motor.

Modulo de datos puro (sin imports de app.*): lo consumen config, engines, rutas
y el script de descarga sin riesgo de ciclos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

# Arquitecturas soportadas. Viaja en la API para que la UI pueda explicar de
# donde sale cada modelo, pero el usuario NUNCA la elige: elige un id de modelo
# y el pipeline resuelve el motor.
Architecture = Literal["mdx", "vr"]

STEM_INSTRUMENTAL = "audio.stem.instrumental"
STEM_VOCALS = "audio.stem.vocals"


@dataclass(frozen=True, slots=True)
class SeparationStem:
    # Id que viaja en download?stem= y en las respuestas de la API.
    id: str
    # La copia la traduce el frontend; el backend solo manda la clave.
    label_key: str
    # De donde sale el audio: "primary" = lo que el modelo infiere;
    # "secondary" = el resto (la resta compensada en MDX, la mascara
    # complementaria en VR).
    source: Literal["primary", "secondary"]


@dataclass(frozen=True, slots=True)
class SeparationModelSpec:
    """Campos comunes a toda entrada del catalogo, sea MDX o VR."""

    id: str
    # Nombre propio del modelo: se muestra tal cual, no se traduce.
    name: str
    filename: str
    url: str
    # SHA-256 del archivo completo, pineado contra el release real.
    sha256: str
    # Que stem SACA el modelo, con el nombre que le da su proyecto de origen
    # ("Instrumental" | "Vocals" | "Reverb" | "No Echo" | "No Reverb"); el otro
    # sale del complemento.
    primary_stem: str
    # "karaoke" (separar voz/instrumental) o "cleanup" (pasada de limpieza:
    # quitar reverb, quitar eco). La UI agrupa el picker con esto.
    category: str
    # Que hace el modelo, dicho para el usuario; el frontend traduce.
    description_key: str
    # Stems de cara al usuario, ORDENADOS: el primero es el que el usuario
    # quiere (lo sirve downloadUrl); el segundo va en ?stem=<id>.
    stems: tuple[SeparationStem, SeparationStem]

    # Lo fija la subclase, no los datos: una entrada no puede declarar una
    # arquitectura que no coincida con sus propios parametros.
    architecture: ClassVar[Architecture]

    @property
    def main_stem(self) -> SeparationStem:
        return self.stems[0]

    @property
    def other_stem(self) -> SeparationStem:
        return self.stems[1]

    def stem_ids(self) -> tuple[str, str]:
        return (self.stems[0].id, self.stems[1].id)
