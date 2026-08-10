"""Cuanto mide DE VERDAD el objeto que se pidio: una SUGERENCIA, nunca una cota.

El carril de malla (Shap-E, desde texto y desde foto) no da cotas y no las va a
dar: la malla sale en escala arbitraria y hoy se escala a 100 mm porque hay que
escalarla a algo. Ese 100 mm no es un tamano, es un relleno — y una taza de cafe
mide unos 95 mm de alto mientras que un tornillo M3 mide 20: escalar los dos a lo
mismo garantiza que uno de los dos salga inservible.

Lo que un modelo de lenguaje SI sabe es cuanto mide una taza. Por eso esto le
pregunta al MISMO servidor que ya escribe el OpenSCAD (`openscad_llm`), con el
mismo cliente configurado, y por eso lo que devuelve NO se aplica solo: se
ofrece, el usuario lo confirma, y el trabajo registra de donde salio la medida.

Tres cosas que este modulo NO hace, a proposito:

  - No sale a internet a buscar medidas de referencia. Sin una fuente con datos
    verificables y licencia usable, una medida traida de la web se lee como un
    dato duro y sigue siendo una conjetura, solo que con mejor tipografia.
    Equivocarse aca cuesta plastico y horas de impresion.
  - No aplica nada por su cuenta. Una medida que se aplica sola es una cota
    inventada, y el modulo entero existe para NO inventar cotas.
  - No le cree al modelo. Un numero fuera de rango, sin forma de numero o sin
    objeto reconocido no produce sugerencia: preferir "no se" a un 4000 mm.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, replace

from app.services.openscad_llm import (
    ChatClient,
    LlmUnavailable,
    OpenAiCompatibleClient,
)

# Lo que una impresora de escritorio puede llegar a hacer de UNA pieza. Por
# debajo de 5 mm no queda pieza que valga la pena imprimir; por encima de 400 mm
# no entra en ninguna de las camas que conoce el banco. Un numero fuera de esto
# no es una estimacion, es el modelo alucinando.
MIN_ESTIMATE_MM = 5.0
MAX_ESTIMATE_MM = 400.0

# Una respuesta de una linea no justifica los diez minutos que espera el carril
# CAD: esto corre con el usuario mirando la pantalla.
ESTIMATE_TIMEOUT_S = 30

# El objeto de referencia es una etiqueta, no un parrafo. Cortarlo evita que un
# modelo hablador se coma la pantalla.
MAX_REFERENCE_CHARS = 80

MAX_HINT_CHARS = 400

SYSTEM_PROMPT = (
    "Estimas cuanto mide un objeto REAL, para imprimirlo en 3D.\n"
    "Respondes UNICAMENTE con un objeto JSON, sin explicaciones ni markdown:\n"
    '{"longest_mm": <numero>, "reference": "<objeto de referencia, 6 palabras '
    'como maximo>"}\n'
    "Reglas:\n"
    "- longest_mm es el lado MAS LARGO del objeto real, en milimetros.\n"
    "- Es el tamano del objeto de verdad, no el de una miniatura ni una maqueta.\n"
    '- Si no sabes que objeto es, respondes {"longest_mm": null, "reference": null}.'
)


class SizeEstimateUnavailable(RuntimeError):
    """No hay sugerencia que ofrecer.

    Nunca rompe un trabajo: quien llama sigue con el default, que es exactamente
    lo que hacia antes de que esto existiera.
    """


@dataclass(frozen=True, slots=True)
class SizeEstimate:
    longest_mm: float
    # Contra que objeto lo comparo el modelo. Es lo que vuelve REVISABLE la
    # sugerencia: "95 mm" no se discute, "como una taza de cafe" si.
    reference: str


# El primer bloque {...} de la respuesta. Los modelos chicos envuelven en
# markdown o anteponen una frase, y descartar por eso una respuesta correcta
# seria tirar la estimacion por un problema de formato.
_JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)

# Solo un numero, con "mm" opcional. Deliberadamente no acepta "cm" ni "in":
# leer 9.5 cm como 9.5 mm es el error caro que este modulo tiene que evitar.
_BARE_NUMBER = re.compile(r"^\s*(\d+(?:[.,]\d+)?)\s*(?:mm)?\s*$", re.IGNORECASE)


def estimate_longest_mm(prompt: str, *, client: ChatClient) -> SizeEstimate:
    """Le pregunta al modelo cuanto mide el objeto descrito. Puede no saber."""
    objeto = prompt.strip()[:MAX_HINT_CHARS]
    if not objeto:
        raise SizeEstimateUnavailable("Hace falta una descripcion del objeto.")
    try:
        respuesta = _short_lived(client).complete(SYSTEM_PROMPT, f"OBJETO: {objeto}")
    except (LlmUnavailable, OSError, ValueError) as exc:
        # Servidor caido, timeout de lectura o respuesta que no era JSON. Nada de
        # esto puede escalar a un error de la pantalla: es una sugerencia.
        raise SizeEstimateUnavailable(
            f"No se pudo estimar el tamano: {exc}"
        ) from exc
    return _parse(respuesta)


def _short_lived(client: ChatClient) -> ChatClient:
    """El MISMO cliente configurado, con la paciencia que corresponde a la pregunta.

    Reusar el cliente del carril CAD tal cual dejaria a la pantalla esperando
    hasta diez minutos por un numero de tres digitos.
    """
    if isinstance(client, OpenAiCompatibleClient):
        return replace(client, timeout_s=ESTIMATE_TIMEOUT_S)
    return client


def _parse(respuesta: str) -> SizeEstimate:
    datos = _first_json_object(respuesta)
    if datos is not None:
        return _from_json(datos)
    solo_numero = _BARE_NUMBER.match(respuesta or "")
    if solo_numero is not None:
        # Un modelo que contesta "95" o "95 mm" y nada mas no es ambiguo.
        return SizeEstimate(longest_mm=_valid_mm(solo_numero.group(1)), reference="")
    raise SizeEstimateUnavailable(
        "El modelo no devolvio una medida que se pueda leer."
    )


def _first_json_object(text: str) -> dict | None:
    for candidato in _JSON_BLOCK.findall(text or ""):
        try:
            datos = json.loads(candidato)
        except json.JSONDecodeError:
            continue
        if isinstance(datos, dict):
            return datos
    return None


def _from_json(datos: dict) -> SizeEstimate:
    if datos.get("longest_mm") is None:
        # El modelo dijo que no sabe. Eso es una respuesta valida y honesta, y se
        # trata igual que no tener sugerencia.
        raise SizeEstimateUnavailable("El modelo no reconocio el objeto.")
    return SizeEstimate(
        longest_mm=_valid_mm(datos.get("longest_mm")),
        reference=clean_reference(datos.get("reference")),
    )


def _valid_mm(raw: object) -> float:
    medida = _as_float(raw)
    if not math.isfinite(medida):
        raise SizeEstimateUnavailable("El modelo no devolvio un numero de milimetros.")
    if not MIN_ESTIMATE_MM <= medida <= MAX_ESTIMATE_MM:
        raise SizeEstimateUnavailable(
            f"La medida estimada ({medida:g} mm) queda fuera de lo que se puede "
            f"imprimir ({MIN_ESTIMATE_MM:g}-{MAX_ESTIMATE_MM:g} mm)."
        )
    return round(medida, 1)


def _as_float(raw: object) -> float:
    if isinstance(raw, bool):
        # True vale 1.0 y pasaria como "1 mm" si no se corta aca.
        raise SizeEstimateUnavailable("El modelo no devolvio un numero de milimetros.")
    if isinstance(raw, str):
        solo_numero = _BARE_NUMBER.match(raw)
        if solo_numero is None:
            raise SizeEstimateUnavailable(
                "El modelo no devolvio un numero de milimetros."
            )
        raw = solo_numero.group(1)
    try:
        return float(str(raw).replace(",", "."))
    except (TypeError, ValueError) as exc:
        raise SizeEstimateUnavailable(
            "El modelo no devolvio un numero de milimetros."
        ) from exc


def clean_reference(raw: object) -> str:
    """Una etiqueta corta de una linea, o nada.

    Lo que vuelve del modelo se muestra en pantalla y viaja en el trabajo: se
    aplasta a una sola linea y se corta, sin confiar en que sea breve.
    """
    if not isinstance(raw, str):
        return ""
    return " ".join(raw.split())[:MAX_REFERENCE_CHARS]
