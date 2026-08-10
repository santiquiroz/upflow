"""La cadena de LIMPIEZA: varios modelos de mascara aplicados uno tras otro.

Los modelos de la categoria "cleanup" del catalogo de separacion no son
separadores aunque compartan motor: un separador divide una mezcla en dos cosas
que el usuario quiere (voz e instrumental), y estos QUITAN UN DEFECTO — entra
audio, sale el mismo audio sin ruido, sin eco o sin reverb. Eso los vuelve
encadenables, que es lo que la gente hace a mano en UVR: pasar el archivo por
De-Noise, despues por De-Echo, despues por De-Reverb.

Este modulo es a la limpieza lo que voice_chain.py es a la voz: define QUE pasos
hay, en QUE orden corren y cuales se pisan entre si. El pipeline solo ejecuta lo
que sale de aca.

ORDEN FIJO, con causalidad (misma politica que build_filter_chain de la cadena
de voz: el orden sale del catalogo y NO del orden en que llegaron los ids, para
que un request no pueda invertirlo por accidente):

1. De-Noise primero. El ruido de banda ancha esta presente en TODO el espectro y
   en todo el tiempo, incluidos los silencios donde el eco y la reverb son mas
   audibles. Un modelo de eco entrenado para encontrar repeticiones discretas ve
   ese piso de ruido como energia que se repite, asi que quitarlo primero es lo
   que hace que las pasadas siguientes trabajen sobre lo que vinieron a buscar.
2. De-Echo despues. El eco son reflejos DISCRETOS y separados en el tiempo: se
   modelan como copias retardadas, y una mascara los encuentra mientras sigan
   siendo copias reconocibles. Correrlo antes del de-reverb evita que esas copias
   queden enterradas en la cola difusa.
3. De-Reverb al final. La reverb es la cola DIFUSA que queda cuando los reflejos
   ya no se distinguen uno de otro. Es lo ultimo que tiene sentido atacar: sacar
   la cola primero le borraria al de-echo justo el material del que deduce los
   reflejos.

EXCLUSION POR FAMILIA. Cada paso declara la familia a la que pertenece y que
familias CUBRE. Dos pasos que cubren la misma familia son redundantes — no es
una preferencia de la UI, es una propiedad del catalogo, y por eso vive aca y no
en ifs sueltos del frontend:

* deecho_normal vs deecho_aggressive: el MISMO modelo en dos intensidades
  (checkpoints distintos, misma tarea). Correr los dos es correr dos veces la
  misma pasada.
* deecho_dereverb hace eco y reverb en UNA pasada, asi que cubre las dos
  familias: elegirlo deja fuera a los dos de-echo y tambien a reverb_hq.

CADA PASADA ES CON PERDIDA. Son modelos de mascara: deciden que parte del
espectro sobrevive y descartan el resto, sin poder devolverlo. Encadenar tres
pasadas no es "tres veces mas limpio", es tres descartes acumulados, y a partir
de la tercera el resultado empieza a sonar procesado. No se bloquea (el usuario
puede querer exactamente eso) pero se avisa: ver OVERPROCESSING_PASS_THRESHOLD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Que defecto ataca un paso. No es cosmetico: es la unidad de exclusion.
CleanupFamily = Literal["denoise", "deecho", "dereverb"]

# A partir de cuantas pasadas se avisa que el resultado puede sonar
# sobreprocesado. Tres es el largo maximo posible de la cadena (una pasada por
# familia), asi que el aviso describe exactamente el caso de la cadena completa.
OVERPROCESSING_PASS_THRESHOLD = 3


@dataclass(frozen=True, slots=True)
class CleanupStepSpec:
    """Un paso de la cadena, atado a un modelo del catalogo de separacion."""

    # Id en SEPARATION_MODELS. El paso NO duplica el nombre, la descripcion ni
    # el archivo del modelo: los lee del catalogo, que es su unica fuente.
    model_id: str
    family: CleanupFamily
    # Todas las familias que este modelo resuelve en su pasada. Para casi todos
    # es solo la suya; deecho_dereverb resuelve dos.
    covers: tuple[CleanupFamily, ...]


# El ORDEN de esta tupla ES el orden de ejecucion de la cadena (ver el docstring
# del modulo). Dentro de una familia el orden solo decide como se lista en la
# UI, porque la exclusion garantiza que corra a lo sumo uno.
CLEANUP_CHAIN: tuple[CleanupStepSpec, ...] = (
    CleanupStepSpec("denoise", "denoise", ("denoise",)),
    CleanupStepSpec("deecho_normal", "deecho", ("deecho",)),
    CleanupStepSpec("deecho_aggressive", "deecho", ("deecho",)),
    CleanupStepSpec("deecho_dereverb", "deecho", ("deecho", "dereverb")),
    CleanupStepSpec("reverb_hq", "dereverb", ("dereverb",)),
)

_SPEC_BY_ID: dict[str, CleanupStepSpec] = {spec.model_id: spec for spec in CLEANUP_CHAIN}
_POSITION_BY_ID: dict[str, int] = {
    spec.model_id: index for index, spec in enumerate(CLEANUP_CHAIN)
}

# Como se nombra una familia dentro de un mensaje de error del backend. La copia
# de la UI vive en el catalogo de traducciones (clave audio.cleanup.family.<id>);
# esto es solo para que el 400 se lea como una frase y no como un id.
_FAMILY_IN_MESSAGE: dict[str, str] = {
    "denoise": "quitar ruido",
    "deecho": "quitar eco",
    "dereverb": "quitar reverb",
}


class UnknownCleanupStep(ValueError):
    """Un id que no esta en el catalogo de limpieza."""


class RedundantCleanupSelection(ValueError):
    """Dos pasos elegidos que atacan el mismo defecto."""


def cleanup_catalog() -> tuple[CleanupStepSpec, ...]:
    """Los pasos de la cadena, en el ORDEN correcto de ejecucion."""
    return CLEANUP_CHAIN


def is_cleanup_model(model_id: str) -> bool:
    return model_id in _SPEC_BY_ID


def cleanup_spec(model_id: str) -> CleanupStepSpec:
    try:
        return _SPEC_BY_ID[model_id]
    except KeyError as exc:
        raise UnknownCleanupStep(_unknown_message([model_id])) from exc


def _unknown_message(unknown: list[str]) -> str:
    known = ", ".join(spec.model_id for spec in CLEANUP_CHAIN)
    return (
        "Pasos de limpieza desconocidos: "
        + ", ".join(sorted(unknown))
        + f". Validos: {known}."
    )


def _redundant_message(first: CleanupStepSpec, second: CleanupStepSpec) -> str:
    shared = [family for family in first.covers if family in second.covers]
    tareas = ", ".join(_FAMILY_IN_MESSAGE[family] for family in shared)
    return (
        f"Pasos de limpieza redundantes: {first.model_id!r} y {second.model_id!r} "
        f"hacen la misma tarea ({tareas}). Elegi uno solo: correr los dos seria "
        "gastar una pasada con perdida en algo que ya se hizo."
    )


def _first_conflict(
    specs: list[CleanupStepSpec],
) -> tuple[CleanupStepSpec, CleanupStepSpec] | None:
    for index, first in enumerate(specs):
        for second in specs[index + 1 :]:
            if any(family in second.covers for family in first.covers):
                return first, second
    return None


def _ordered_unique_specs(selected_ids: list[str]) -> list[CleanupStepSpec]:
    """Los specs pedidos, deduplicados y en el orden del catalogo.

    Un id repetido NO es una segunda pasada: nadie puede pedir dos veces el
    mismo modelo por accidente y querer decir algo distinto, asi que se colapsa
    en vez de rechazarse. Lo que si se rechaza es pedir DOS modelos distintos
    para la misma tarea, que es donde no hay una intencion evidente.
    """
    seen: set[str] = set()
    specs: list[CleanupStepSpec] = []
    for model_id in selected_ids:
        if model_id in seen:
            continue
        seen.add(model_id)
        specs.append(_SPEC_BY_ID[model_id])
    return sorted(specs, key=lambda spec: _POSITION_BY_ID[spec.model_id])


def cleanup_steps_from_selection(selected_ids: list[str]) -> list[CleanupStepSpec]:
    """Traduce lo que el usuario tildo en la UI a pasadas ejecutables.

    El ORDEN sale del catalogo, NO del orden en que llegaron los ids: la cadena
    tiene causalidad documentada (ver el docstring del modulo) y un request no
    deberia poder invertirla por accidente.

    Una combinacion redundante se RECHAZA en vez de normalizarse. Normalizar
    exigiria elegir un ganador, y entre deecho_normal y deecho_aggressive no hay
    ninguno obvio: el ganador seria una invencion nuestra sobre una decision de
    calidad que es del usuario. Descartar uno en silencio es exactamente el modo
    de fallo que el resto del pipeline evita.
    """
    unknown = [model_id for model_id in selected_ids if model_id not in _SPEC_BY_ID]
    if unknown:
        raise UnknownCleanupStep(_unknown_message(unknown))

    specs = _ordered_unique_specs(selected_ids)
    conflict = _first_conflict(specs)
    if conflict is not None:
        raise RedundantCleanupSelection(_redundant_message(*conflict))
    return specs


def is_overprocessing(pass_count: int) -> bool:
    return pass_count >= OVERPROCESSING_PASS_THRESHOLD
