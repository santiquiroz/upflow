from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.services.audio_mastering import MASTERING_PRESETS
from app.services.realtime_service import PRESETS as REALTIME_PRESETS

# ---------------------------------------------------------------------------
# El texto que se ve en pantalla se traduce en el frontend, no se escribe en
# Python. El catalogo de voz ya lo hacia bien (manda `labelKey`/`descriptionKey`
# y el frontend traduce); los presets de Tiempo real y de Acabado se escribieron
# despues y mandaban la copia en espanol cruda, asi que esas dos pantallas se
# quedaban a medio traducir por mas que el frontend estuviera perfecto.
#
# Un `labelKey` que no exista en los dos catalogos es peor que no traducir: el
# frontend devuelve la clave y en pantalla se lee "realtime.preset.fsr.label".
# ---------------------------------------------------------------------------

FRONTEND_I18N = Path(__file__).resolve().parents[1] / "frontend" / "src" / "i18n"


def catalog_keys(locale: str) -> set[str]:
    source = (FRONTEND_I18N / f"{locale}.ts").read_text(encoding="utf-8")
    return set(re.findall(r'^\s*"([\w.]+)":', source, flags=re.MULTILINE))


def all_declared_keys() -> list[str]:
    return [
        *(p.label_key for p in REALTIME_PRESETS),
        *(p.description_key for p in REALTIME_PRESETS),
        *(p.label_key for p in MASTERING_PRESETS),
        *(p.description_key for p in MASTERING_PRESETS),
    ]


@pytest.mark.parametrize("locale", ["en", "es"])
def test_every_key_the_backend_promises_exists_in_the_catalog(locale: str) -> None:
    known = catalog_keys(locale)
    missing = [key for key in all_declared_keys() if key not in known]
    assert missing == []


def test_the_backend_ships_keys_not_sentences() -> None:
    """Una clave no lleva espacios; una frase si. Es la diferencia que separa
    'algo que el frontend puede traducir' de 'copia que ya se decidio'."""
    with_spaces = [key for key in all_declared_keys() if " " in key]
    assert with_spaces == []
