from __future__ import annotations

import pytest

from app.api.routes import get_voice_catalog
from app.services.voice_chain import delivery_choices, step_catalog

# El endpoint no tenia NINGUN test y quedo roto en silencio al renombrar
# label -> label_key en voice_chain: la suite entera seguia verde porque nada lo
# ejecutaba. Estos tests existen para que un rename del contrato falle aca en vez
# de en la primera llamada real.


@pytest.mark.asyncio
async def test_the_endpoint_answers_without_exploding():
    response = await get_voice_catalog()
    assert response.steps
    assert response.deliveries


@pytest.mark.asyncio
async def test_every_step_of_the_catalog_is_exposed_in_order():
    # El orden NO es cosmetico: la cadena tiene causalidad documentada en
    # build_filter_chain y la UI la presenta tal cual.
    response = await get_voice_catalog()
    assert [step.id for step in response.steps] == [info.id for info in step_catalog()]


@pytest.mark.asyncio
async def test_every_delivery_target_is_exposed_with_its_numbers():
    response = await get_voice_catalog()
    expected = {
        choice["id"]: (choice["lufs"], choice["truePeakDb"])
        for choice in delivery_choices()
    }
    assert {d.id: (d.lufs, d.true_peak_db) for d in response.deliveries} == expected


@pytest.mark.asyncio
async def test_the_copy_travels_as_translation_keys():
    response = await get_voice_catalog()
    for step in response.steps:
        assert step.label_key.startswith("voice.step.")
        assert step.description_key.startswith("voice.step.")
    for delivery in response.deliveries:
        assert delivery.label_key.startswith("voice.delivery.")
        assert delivery.description_key.startswith("voice.delivery.")


@pytest.mark.asyncio
async def test_keys_are_unique_across_steps_and_deliveries():
    # Dos entradas compartiendo clave significaria que la UI muestra el mismo
    # texto en dos lugares distintos sin que nada lo detecte.
    response = await get_voice_catalog()
    keys = [step.label_key for step in response.steps]
    keys += [delivery.label_key for delivery in response.deliveries]
    assert len(keys) == len(set(keys))


@pytest.mark.asyncio
async def test_the_response_serializes_camel_case_aliases():
    response = await get_voice_catalog()
    step = response.model_dump(by_alias=True)["steps"][0]
    assert "labelKey" in step
    assert "descriptionKey" in step
    assert "defaultEnabled" in step
    delivery = response.model_dump(by_alias=True)["deliveries"][0]
    assert "truePeakDb" in delivery


@pytest.mark.asyncio
async def test_the_default_selection_is_a_usable_chain():
    # Lo que el usuario ve tildado al abrir el panel: si no viniera nada por
    # defecto, la mejora de voz no haria nada hasta que alguien adivine que
    # tildar.
    response = await get_voice_catalog()
    enabled = [step.id for step in response.steps if step.default_enabled]
    assert enabled


def test_the_catalog_answers_through_the_real_app():
    """Este endpoint se rompio en silencio una vez porque nada lo ejecutaba.

    Los tests de arriba llaman la corrutina directo; este va por el ruteo real,
    que es el camino que el frontend usa de verdad.
    """
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        response = client.get("/api/v1/audio/voice-catalog")

    assert response.status_code == 200
    payload = response.json()
    assert [step["id"] for step in payload["steps"]] == [info.id for info in step_catalog()]
    for step in payload["steps"]:
        assert "labelKey" in step
        assert "descriptionKey" in step
        assert "defaultEnabled" in step
    for delivery in payload["deliveries"]:
        assert "truePeakDb" in delivery
