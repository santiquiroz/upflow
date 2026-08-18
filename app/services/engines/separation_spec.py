"""Forma COMPARTIDA de una entrada del catalogo de separacion de stems.

El catalogo tiene tres arquitecturas de modelo con parametros incompatibles:
MDX-Net (n_fft/dim_f/dim_t/compensate, un stem inferido + resta compensada),
VR 5.1 CascadedNet (4band_v3, nout, aggression, mascara complementaria) y
RoFormer (grafo spec-in/mask-out que puede emitir de 1 a N stems). Lo que NO
cambia entre ellas es todo lo que consumen el catalogo, la API, el pack de
descarga y el job: id, nombre propio, archivo, url, sha256, categoria, copia y
la LISTA ORDENADA de stems.

Por eso esto es un supertipo y no tres catalogos sueltos: `separation_models.py`,
`routes.py`, `pack_provisioner.py` y `audio_job_manager.py` operan SOLO sobre
estos campos, y con un tipo base compartido no pueden desincronizarse entre
arquitecturas. Los campos propios de cada motor viven en la subclase y solo los
toca su motor.

Modulo de datos puro (sin imports de app.*): lo consumen config, engines, rutas
y el script de descarga sin riesgo de ciclos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Literal

# Arquitecturas soportadas. Viaja en la API para que la UI pueda explicar de
# donde sale cada modelo, pero el usuario NUNCA la elige: elige un id de modelo
# y el pipeline resuelve el motor.
Architecture = Literal["mdx", "vr", "roformer", "umx"]

STEM_INSTRUMENTAL = "audio.stem.instrumental"
STEM_VOCALS = "audio.stem.vocals"
# Estas tres existen desde que hay un modelo que las emite (umxhq, 2026-08-17).
# Antes no estaban a proposito: una clave de i18n sin modelo detras promete algo
# que no se puede hacer.
STEM_DRUMS = "audio.stem.drums"
STEM_BASS = "audio.stem.bass"
STEM_OTHER = "audio.stem.other"

# Un stem que NO sale del modelo sino de restarle a la mezcla todo lo que el
# modelo si infiere. Es el complemento exacto, y solo existe cuando el modelo
# deja algo sin explicar: un modelo que emite los cuatro stems de una mezcla no
# tiene "el resto".
RESIDUAL = "residual"

# De donde sale el audio de un stem. Los dos unicos casos que existen: el
# indice de la salida del modelo que lo produce, o el residuo. No hay un tercero
# ni uno es el default del otro.
StemSource = int | Literal["residual"]


@dataclass(frozen=True, slots=True)
class SeparationStem:
    # Id que viaja en download?stem= y en las respuestas de la API.
    id: str
    # La copia la traduce el frontend; el backend solo manda la clave.
    label_key: str
    # Indice de la salida del modelo, o RESIDUAL. Un modelo de una sola salida
    # numera esa salida 0 y declara su complemento RESIDUAL; uno de N salidas
    # reales las numera 0..N-1 y no declara residuo alguno. El motor no
    # necesita saber cual de los dos casos es: `stems_in_catalog_order` los
    # resuelve igual.
    source: StemSource


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
    # ("Instrumental" | "Vocals" | "Reverb" | "No Echo" | "No Reverb"). En un
    # modelo multi-stem es la lista de lo que infiere, separada por " / ".
    primary_stem: str
    # "karaoke" (separar voz/instrumental) o "cleanup" (pasada de limpieza:
    # quitar reverb, quitar eco). La UI agrupa el picker con esto.
    category: str
    # Que hace el modelo, dicho para el usuario; el frontend traduce.
    description_key: str
    # Stems de cara al usuario, ORDENADOS: el primero es el que el usuario
    # quiere (lo sirve downloadUrl); TODOS, incluido ese, se piden por
    # ?stem=<id>. Son dos en los modelos de karaoke y limpieza, cuatro en los
    # multi-stem; nada aguas abajo puede asumir un numero.
    stems: tuple[SeparationStem, ...]

    # Lo que el usuario tiene que saber ANTES de elegir el modelo, no al final
    # del trabajo: hoy es el costo del carril de maxima calidad (~9x el default).
    # None = el modelo no necesita advertir nada, que es el caso normal.
    #
    # kw_only para que un campo con default en la BASE no obligue a las
    # subclases a darle default a los suyos (MdxModelSpec/VrModelSpec declaran
    # parametros obligatorios despues de este).
    warning_key: str | None = field(default=None, kw_only=True)

    # Lo fija la subclase, no los datos: una entrada no puede declarar una
    # arquitectura que no coincida con sus propios parametros.
    architecture: ClassVar[Architecture]

    @property
    def files(self) -> tuple[str, ...]:
        """TODOS los archivos que el modelo necesita en disco.

        Casi siempre es uno, y por eso `filename` sigue siendo el campo; umxhq
        son cuatro grafos (uno por pista) que no sirven de a uno. Lo que decide
        si el modelo esta instalado es esta lista, no `filename`: con un solo
        archivo presente de cuatro, el picker lo ofrecia y el trabajo moria al
        cargar la segunda sesion.
        """
        return (self.filename,)

    @property
    def main_stem(self) -> SeparationStem:
        """El que sirve downloadUrl. NO es "el otro es el resto": con cuatro
        stems los otros tres son tan finales como este."""
        return self.stems[0]

    def stem_ids(self) -> tuple[str, ...]:
        return tuple(stem.id for stem in self.stems)
