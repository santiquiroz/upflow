"""Que pares de idiomas se PUEDEN instalar, no solo cuales ya estan.

Con cero pares bajados, la pantalla escondia el selector de traduccion entero: el
usuario no veia la funcion ni una forma de conseguirla. Un callejon sin salida
silencioso, que es peor que un cartel de error.

OPUS-MT (Helsinki-NLP, licencia Apache 2.0 / CC-BY 4.0 segun el par, comercial en
ambos casos) publica un modelo POR PAR de idiomas. La app ya sabe bajar el par que
se le pida — lo que faltaba era ofrecerlos.

La lista es corta A PROPOSITO. OPUS-MT tiene cientos de pares, la mayoria entre
combinaciones que nadie de aca va a usar, y una lista larga esconde los tres que
importan. Estos son los que `onnx-community` publica ya exportados a ONNX, que es
el formato que este motor corre.
"""

from __future__ import annotations

import re

# `xx-yy`, que es lo que el script de descarga valida antes de correr.
_PAIR = re.compile(r"^[a-z]{2,3}-[a-z]{2,3}$")

# Verificado uno por uno contra Hugging Face el 2026-08-06: cada par de esta
# lista tiene su repo publicado. De los 16 que arme de memoria, TRES no existian
# —en-pt, pt-en y en-ja— y habrian dejado al usuario esperando una descarga que
# nunca iba a llegar. Agregar un par sin verificarlo rompe una prueba.
INSTALLABLE_PAIRS: tuple[str, ...] = (
    "en-es",
    "es-en",
    "en-fr",
    "fr-en",
    "en-de",
    "de-en",
    "en-it",
    "it-en",
    # Al japones no hay: onnx-community publica ja-en pero no el sentido inverso.
    "ja-en",
    "en-zh",
    "zh-en",
    "en-ru",
    "ru-en",
)


def is_installable(pair: str) -> bool:
    """Si el valor tiene la forma de un par y esta en la lista.

    Se chequean las DOS cosas: la forma porque el valor termina en una linea de
    comandos, y la pertenencia porque pedir un par que no existe deja al usuario
    esperando una descarga que va a fallar.
    """
    return bool(_PAIR.match(pair)) and pair in INSTALLABLE_PAIRS


def pairs_for_language(source: str) -> list[str]:
    """A que idiomas se puede traducir desde este, con lo que hay para instalar."""
    prefijo = f"{source.lower()}-"
    return [par.removeprefix(prefijo) for par in INSTALLABLE_PAIRS if par.startswith(prefijo)]
