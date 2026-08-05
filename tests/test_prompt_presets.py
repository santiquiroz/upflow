from __future__ import annotations

import pytest

from app.services.prompt_presets import (
    PROMPT_PRESETS,
    presets_for_mode,
)

# ---------------------------------------------------------------------------
# Los presets son COPIA: viajan como claves de traduccion, igual que los presets
# de Tiempo real y de Acabado. Lo que NO es copia es el prompt en si — ese es el
# texto que se le manda al modelo y se manda tal cual, sin traducir: traducir
# "cinematic lighting, 35mm film" cambiaria lo que genera.
# ---------------------------------------------------------------------------


def test_every_preset_declares_the_mode_it_belongs_to() -> None:
    assert {preset.mode for preset in PROMPT_PRESETS} <= {
        "text-to-image",
        "image-to-image",
        "video",
    }


def test_every_mode_the_app_offers_has_at_least_one_preset() -> None:
    """El pedido fue que TODOS los casos de generacion tengan prompts listos."""
    for mode in ("text-to-image", "image-to-image", "video"):
        assert presets_for_mode(mode), f"el modo {mode} quedo sin presets"


def test_the_label_travels_as_a_key_not_as_a_sentence() -> None:
    # Una clave no lleva espacios; una frase si. Es la diferencia entre algo que
    # el frontend puede traducir y copia ya decidida en el backend.
    for preset in PROMPT_PRESETS:
        assert " " not in preset.label_key
        assert preset.label_key.startswith("generate.preset.")


def test_the_prompt_itself_is_literal_text_not_a_key() -> None:
    """El prompt se le manda al modelo tal cual. Traducirlo cambiaria el
    resultado, asi que NO es copia y no lleva clave."""
    for preset in PROMPT_PRESETS:
        assert preset.prompt.strip()
        assert not preset.prompt.startswith("generate.")


def test_preset_ids_are_unique() -> None:
    ids = [preset.id for preset in PROMPT_PRESETS]
    assert len(ids) == len(set(ids))


def test_asking_for_an_unknown_mode_gives_nothing_rather_than_everything() -> None:
    assert presets_for_mode("no-existe") == []


@pytest.mark.parametrize("mode", ["text-to-image", "video"])
def test_presets_only_come_back_for_their_own_mode(mode: str) -> None:
    for preset in presets_for_mode(mode):
        assert preset.mode == mode


def test_the_api_publishes_the_presets() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        payload = client.get("/api/v1/generation/prompt-presets").json()

    assert len(payload["presets"]) == len(PROMPT_PRESETS)
    first = payload["presets"][0]
    assert first["labelKey"].startswith("generate.preset.")
    # El prompt viaja literal, no como clave.
    assert " " in first["prompt"]


def test_every_label_key_exists_in_both_catalogs() -> None:
    """Una clave prometida y ausente se lee en pantalla como
    'generate.preset.portrait'. Peor que no traducir."""
    import re
    from pathlib import Path

    i18n = Path(__file__).resolve().parents[1] / "frontend" / "src" / "i18n"
    for locale in ("en", "es"):
        known = set(
            re.findall(r'^\s*"([\w.]+)":', (i18n / f"{locale}.ts").read_text(encoding="utf-8"), re.M)
        )
        missing = [p.label_key for p in PROMPT_PRESETS if p.label_key not in known]
        assert missing == [], f"faltan en {locale}: {missing}"
