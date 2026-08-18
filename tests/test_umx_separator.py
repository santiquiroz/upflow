"""El primer modelo de CUATRO pistas del catálogo, que es lo que el contrato esperaba.

El contrato de N stems entró en v0.66.0 sin un modelo que lo usara — hasta el
comentario de `separation_spec.py` decía que agregar claves de batería/bajo/resto
antes de tiempo prometería algo que no existe. Ahora existe (umxhq, pesos MIT),
así que estas pruebas fijan lo que cambia al haber un modelo que emite cuatro
salidas reales y NINGÚN residuo.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.services.engines.separation_models import (
    SEPARATION_MODELS,
    installed_model_ids,
)
from app.services.engines.separation_spec import RESIDUAL
from app.services.engines.umx_models import UMX_MODELS
from app.services.engines.umx_separator import UmxSeparator

MODELO = "umx_4stem"
BINS = 2049


def spec():
    return SEPARATION_MODELS[MODELO]


class SesionFalsa:
    """Devuelve la magnitud de entrada escalada: separación falsa pero de la forma correcta."""

    def __init__(self, factor: float) -> None:
        self.factor = factor
        self.llamadas = 0

    def get_inputs(self):
        return [type("E", (), {"name": "mag"})()]

    def get_outputs(self):
        return [type("S", (), {"name": "estimate"})()]

    def run(self, _salidas, feeds):
        self.llamadas += 1
        return [feeds["mag"] * self.factor]


def separador(tmp_path: Path, sesiones: dict[str, SesionFalsa]) -> UmxSeparator:
    settings = Settings(RUNTIME_DIR=str(tmp_path), _env_file=None)
    motor = UmxSeparator(settings, gpu_coordinator=None)
    motor._create_session = lambda device, model_id: sesiones  # type: ignore[method-assign]
    return motor


def test_el_catalogo_declara_cuatro_pistas_y_ningun_residuo() -> None:
    modelo = spec()

    assert modelo.stem_ids() == ("vocals", "drums", "bass", "other")
    # Un modelo que estima las cuatro no deja nada sin explicar: si alguna
    # declarara RESIDUAL, se le estaría restando a la mezcla algo ya contado.
    assert all(stem.source != RESIDUAL for stem in modelo.stems)
    assert [stem.source for stem in modelo.stems] == [0, 1, 2, 3]


def test_son_cuatro_archivos_y_el_catalogo_los_lista_todos() -> None:
    modelo = spec()

    assert len(modelo.files) == 4
    assert modelo.filename in modelo.files
    # Cada grafo con su sha256 propio: un solo hash para cuatro archivos no
    # detectaría que bajó tres buenos y uno corrupto.
    assert len({sha for _, _, sha in modelo.graphs}) == 4


def test_un_modelo_de_un_archivo_sigue_declarando_uno() -> None:
    # La propiedad `files` es de la BASE: si se hubiera puesto solo en la
    # subclase, todo lo que la consume tendría que preguntar de qué tipo es.
    assert SEPARATION_MODELS["inst_hq_3"].files == ("UVR-MDX-NET-Inst_HQ_3.onnx",)


def test_con_tres_de_los_cuatro_grafos_el_modelo_no_figura_instalado(tmp_path: Path) -> None:
    for _, archivo, _ in spec().graphs[:3]:
        (tmp_path / archivo).write_bytes(b"x")

    assert MODELO not in installed_model_ids(tmp_path)

    (tmp_path / spec().graphs[3][1]).write_bytes(b"x")
    assert MODELO in installed_model_ids(tmp_path)


def test_separa_en_cuatro_pistas_en_el_orden_del_catalogo(tmp_path: Path) -> None:
    factores = {"vocals": 0.4, "drums": 0.3, "bass": 0.2, "other": 0.1}
    sesiones = {n: SesionFalsa(f) for n, f in factores.items()}
    motor = separador(tmp_path, sesiones)
    mezcla = np.random.default_rng(7).standard_normal((2, 44100)).astype(np.float64) * 0.1

    pistas = motor._separate_stems(
        mezcla, sesiones, spec(), threading.Event(), on_chunk=None
    )

    assert len(pistas) == 4
    assert all(p.shape == mezcla.shape for p in pistas)
    # Cada grafo corrió UNA vez: el archivo entra entero porque el eje temporal
    # del grafo es dinámico y cortarlo reiniciaría la recurrencia.
    assert all(s.llamadas == 1 for s in sesiones.values())
    # El orden lo fija el catálogo, no el dict de sesiones.
    energias = [float(np.abs(p).mean()) for p in pistas]
    assert energias == sorted(energias, reverse=True)


def test_cancelar_corta_antes_de_correr_el_siguiente_grafo(tmp_path: Path) -> None:
    sesiones = {n: SesionFalsa(0.25) for n in ("vocals", "drums", "bass", "other")}
    motor = separador(tmp_path, sesiones)
    evento = threading.Event()
    evento.set()

    from app.services.engines.separator_base import SeparationCancelled

    with pytest.raises(SeparationCancelled):
        motor._separate_stems(
            np.zeros((2, 4410)), sesiones, spec(), evento, on_chunk=None
        )
    assert all(s.llamadas == 0 for s in sesiones.values())


def test_un_grafo_con_otra_forma_falla_diciendo_cual(tmp_path: Path) -> None:
    class Rara(SesionFalsa):
        def run(self, _salidas, feeds):
            return [np.zeros((1, 5, BINS, 10), dtype=np.float32)]

    sesiones = {n: SesionFalsa(0.25) for n in ("vocals", "drums", "bass", "other")}
    sesiones["drums"] = Rara(1.0)
    motor = separador(tmp_path, sesiones)

    with pytest.raises(RuntimeError) as error:
        motor._separate_stems(
            np.zeros((2, 4410)), sesiones, spec(), threading.Event(), on_chunk=None
        )
    # Con cuatro grafos, "el .onnx no es el del catálogo" sin decir cuál manda a
    # revisar los cuatro archivos.
    assert "drums" in str(error.value)


def test_el_progreso_avanza_una_vez_por_pista(tmp_path: Path) -> None:
    sesiones = {n: SesionFalsa(0.25) for n in ("vocals", "drums", "bass", "other")}
    motor = separador(tmp_path, sesiones)
    avances: list[tuple[int, int]] = []

    motor._separate_stems(
        np.zeros((2, 4410)), sesiones, spec(), threading.Event(), on_chunk=lambda hechos, total: avances.append((hechos, total)),
    )

    assert avances == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_el_id_del_modelo_lo_rutea_al_motor_umx() -> None:
    # El pipeline elige motor por `architecture`; si esta entrada dijera otra
    # cosa, el job iría a un motor que no sabe cargar cuatro grafos.
    assert UMX_MODELS[MODELO].architecture == "umx"
    assert spec().category == "karaoke"


def test_cada_modelo_que_advierte_declara_su_propia_etiqueta() -> None:
    """La etiqueta del picker no puede estar cableada en la interfaz.

    Mientras hubo UN solo modelo con advertencia —el lento— la interfaz dibujaba
    "Lento" fijo cada vez que había `warningKey`. Al llegar el segundo, esa
    equivalencia se volvió falsa y el catálogo empezó a llamar lento al modelo
    MÁS RÁPIDO de los que advierten algo (19x tiempo real contra 1.2x).
    """
    advierten = [s for s in SEPARATION_MODELS.values() if s.warning_key]

    assert len(advierten) >= 2, "con uno solo este test no prueba nada"
    for spec in advierten:
        assert spec.badge_key, f"{spec.id} advierte pero no dice qué mostrar"
    # Y no puede ser la misma para todos: si lo fuera, volvería a ser un texto fijo.
    assert len({spec.badge_key for spec in advierten}) == len(advierten)


def test_el_de_cuatro_pistas_no_se_anuncia_como_lento() -> None:
    # Es el más rápido de los que advierten: la contra es que separa peor.
    assert SEPARATION_MODELS[MODELO].badge_key == "audio.karaoke.badge.lessPrecise"
