from __future__ import annotations

import pytest

from app.services.translation_catalog import (
    INSTALLABLE_PAIRS,
    is_installable,
    pairs_for_language,
)

# ---------------------------------------------------------------------------
# Con cero pares instalados, la pantalla escondia el selector entero: el usuario
# no veia la traduccion NI una forma de conseguirla. Un callejon sin salida
# silencioso, que es peor que un cartel de error.
#
# OPUS-MT publica un modelo POR PAR de idiomas. La app ya sabe bajar el par que
# se le pida; lo que faltaba era OFRECERLOS.
# ---------------------------------------------------------------------------


def test_hay_pares_para_ofrecer() -> None:
    assert INSTALLABLE_PAIRS


def test_todos_los_pares_tienen_la_forma_que_el_script_acepta() -> None:
    # El valor termina en una linea de comandos y el script valida `xx-yy`.
    for par in INSTALLABLE_PAIRS:
        assert is_installable(par), par


class TestLosParesQueImportan:
    @pytest.mark.parametrize("par", ["en-es", "es-en"])
    def test_ingles_y_castellano_van_en_los_dos_sentidos(self, par: str) -> None:
        # Es el par que este usuario necesita, y traducir en un solo sentido
        # serviria para la mitad de los casos.
        assert par in INSTALLABLE_PAIRS

    def test_desde_un_idioma_se_puede_listar_a_donde_se_puede_traducir(self) -> None:
        destinos = pairs_for_language("es")

        assert "en" in destinos
        assert "es" not in destinos, "traducir de un idioma a si mismo no es un par"

    def test_un_idioma_sin_pares_devuelve_vacio_y_no_falla(self) -> None:
        assert pairs_for_language("xx") == []


class TestGuardas:
    @pytest.mark.parametrize("valor", ["", "es", "es-", "es-en-fr", "es en", "es-EN;rm"])
    def test_una_forma_rara_no_es_instalable(self, valor: str) -> None:
        assert is_installable(valor) is False


@pytest.mark.network
def test_todo_par_ofrecido_existe_de_verdad_en_hugging_face() -> None:
    """Ofrecer un par que no existe deja al usuario esperando una descarga que nunca llega.

    Esta prueba pega contra la red a proposito y por eso va marcada: es la unica
    forma de saberlo. Corriendola el 2026-08-06 sobre una lista armada de memoria,
    TRES de dieciseis pares no existian.
    """
    huggingface_hub = pytest.importorskip("huggingface_hub")
    api = huggingface_hub.HfApi()

    faltan = []
    for par in INSTALLABLE_PAIRS:
        try:
            api.model_info(f"onnx-community/opus-mt-{par}")
        except Exception:  # noqa: BLE001 - cualquier fallo cuenta como no disponible
            faltan.append(par)

    assert faltan == [], f"Pares ofrecidos que no existen: {faltan}"
