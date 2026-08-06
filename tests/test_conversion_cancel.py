from __future__ import annotations

import pytest

from app.models import JobStatus
from tests.test_generation_converter import make_converter

# ---------------------------------------------------------------------------
# Cancelar una conversion.
#
# Convertir un SDXL tarda cerca de media hora y ocupa la maquina entera. Sin
# forma de cancelar, quien se equivoca de modelo solo puede cerrar la app —
# y eso deja archivos a medio escribir en el directorio de trabajo.
#
# El corte cae en el limite de cada submodelo, no en cualquier instante: el
# export corre dentro de una libreria que no se puede interrumpir a la mitad.
# Son cinco submodelos en un SDXL, asi que cancelar durante el mas largo espera
# a que ese termine. Es una limitacion real y esta dicha, no escondida.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancelar_una_conversion_en_cola_la_marca_sin_correrla(tmp_path):
    corridas: list[str] = []

    def export(*args, **kwargs):
        corridas.append("corrio")
        return ["unet"]

    convertidor, _i, _s, _r = make_converter(tmp_path, export)
    conversion_id = await convertidor.convert_from_hf("algo/modelo")

    assert convertidor.cancel(conversion_id) is True
    await convertidor._process_next()

    assert convertidor.status(conversion_id).status is JobStatus.cancelled
    assert corridas == [], "no tenia que llegar a exportar"


@pytest.mark.asyncio
async def test_cancelar_a_mitad_corta_en_el_siguiente_submodelo(tmp_path):
    exportados: list[str] = []

    def export(src, out, on_component, *args, **kwargs):
        for nombre in ("text_encoder", "unet", "vae_decoder"):
            on_component(nombre)
            exportados.append(nombre)
        return exportados

    convertidor, _i, _s, _r = make_converter(tmp_path, export)
    conversion_id = await convertidor.convert_from_hf("algo/modelo")

    # Se cancela apenas empieza el primero: el export tiene que cortar ahi.
    original = convertidor.cancel

    def cancelar_en_el_primero(nombre):
        original(conversion_id)

    convertidor._on_component_hook = cancelar_en_el_primero
    await convertidor._process_next()

    assert convertidor.status(conversion_id).status is JobStatus.cancelled
    assert len(exportados) < 3, "siguio exportando despues de cancelar"


@pytest.mark.asyncio
async def test_cancelar_una_conversion_que_no_existe_no_miente(tmp_path):
    convertidor, _i, _s, _r = make_converter(tmp_path, lambda *a, **k: [])

    assert convertidor.cancel("no-existe") is False


@pytest.mark.asyncio
async def test_una_conversion_terminada_ya_no_se_cancela(tmp_path):
    # Decir que si sobre algo que ya termino le haria creer al usuario que
    # deshizo un trabajo que en realidad quedo hecho.
    convertidor, _i, _s, _r = make_converter(tmp_path, lambda *a, **k: ["unet"])
    conversion_id = await convertidor.convert_from_hf("algo/modelo")
    convertidor.status(conversion_id).status = JobStatus.completed

    assert convertidor.cancel(conversion_id) is False
