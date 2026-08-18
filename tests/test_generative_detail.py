"""El pase generativo: geometria de tiles, costura y el limite que lo mantiene honesto.

Lo que se prueba aca es todo lo que NO es el modelo: que la imagen se cubra
entera, que la costura no se vea, que un tile del tamaño equivocado falle en vez
de correr la imagen, y que la fuerza tenga un techo — arriba de cierto punto el
modelo deja de agregar textura y reinterpreta la escena.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.generative_detail import (
    MAX_STRENGTH,
    add_generative_detail,
    clamp_strength,
    plan_tiles,
)


# ---------------------------------------------------------------------------
# Geometria
# ---------------------------------------------------------------------------


def test_los_tiles_cubren_la_imagen_entera() -> None:
    planes = plan_tiles(1000, 700, 512, 64)
    cubierto = np.zeros((1000, 700), dtype=bool)

    for plan in planes:
        cubierto[plan.y0 : plan.y0 + plan.height, plan.x0 : plan.x0 + plan.width] = True

    # Un pixel sin cubrir sale como un cuadrado negro en la salida.
    assert cubierto.all()


def test_los_tiles_se_solapan_de_verdad() -> None:
    planes = plan_tiles(1000, 1000, 512, 64)
    columnas = sorted({plan.x0 for plan in planes})

    # Sin solape no hay banda donde desvanecer y la costura queda a la vista.
    assert all(b - a < 512 for a, b in zip(columnas, columnas[1:]))


def test_una_imagen_mas_chica_que_el_tile_es_un_solo_tile() -> None:
    planes = plan_tiles(300, 200, 512, 64)

    assert len(planes) == 1
    assert (planes[0].height, planes[0].width) == (300, 200)


def test_los_tiles_del_borde_se_saben_del_borde() -> None:
    planes = plan_tiles(1000, 1000, 512, 64)
    primero = planes[0]
    ultimo = planes[-1]

    # Un tile pegado al borde NO se desvanece de ese lado: no hay vecino con
    # quien mezclarse, y hacerlo dejaria el marco medio transparente.
    assert primero.is_top and primero.is_left
    assert ultimo.is_bottom and ultimo.is_right
    assert not primero.is_bottom


def test_ningun_tile_se_sale_de_la_imagen() -> None:
    for plan in plan_tiles(777, 513, 512, 64):
        assert plan.y0 + plan.height <= 777
        assert plan.x0 + plan.width <= 513


# ---------------------------------------------------------------------------
# El limite que lo mantiene siendo "mas detalle" y no "otra imagen"
# ---------------------------------------------------------------------------


def test_la_fuerza_tiene_techo() -> None:
    # Arriba del techo el modelo reinterpreta: aparecen ojos donde habia un boton.
    assert clamp_strength(0.9) == MAX_STRENGTH
    assert clamp_strength(0.2) == 0.2


def test_fuerza_cero_no_es_un_pase_generativo() -> None:
    with pytest.raises(ValueError):
        clamp_strength(0.0)


# ---------------------------------------------------------------------------
# La pasada completa
# ---------------------------------------------------------------------------


def correr(imagen: np.ndarray, run_tile, **extra) -> np.ndarray:
    return add_generative_detail(imagen, run_tile=run_tile, **extra)


def test_un_pase_que_no_cambia_nada_devuelve_la_misma_imagen() -> None:
    imagen = np.random.default_rng(5).integers(0, 255, (600, 600, 3), dtype=np.uint8)

    def identidad(recorte, _plan):
        return recorte

    salida = correr(imagen, identidad, tile=256, overlap=64)

    # Si la union sola ya deformara la imagen, cualquier diferencia despues
    # seria imposible de atribuir al modelo.
    assert np.array_equal(salida, imagen)


def test_la_costura_no_se_ve() -> None:
    imagen = np.full((600, 600, 3), 128, dtype=np.uint8)
    llamadas = {"n": 0}

    def cada_tile_distinto(recorte, _plan):
        # Cada tile "inventa" su propia textura, que es lo que pasa de verdad
        # con difusion: dos tiles vecinos no coinciden en el solape.
        llamadas["n"] += 1
        return np.full_like(recorte, 60 + llamadas["n"] * 15)

    salida = correr(imagen, cada_tile_distinto, tile=256, overlap=64).astype(np.int16)

    # Con feather sobre TODO el solape el cambio es gradual: ningun par de
    # columnas vecinas salta de golpe.
    salto_maximo = int(np.abs(np.diff(salida[:, :, 0].astype(np.int16), axis=1)).max())
    assert salto_maximo <= 3, f"costura visible: salto de {salto_maximo}"


def test_un_tile_de_otro_tamano_falla_en_vez_de_correr_la_imagen() -> None:
    imagen = np.zeros((600, 600, 3), dtype=np.uint8)

    def redondea_a_multiplo(recorte, _plan):
        return np.zeros((recorte.shape[0] + 8, recorte.shape[1], 3), dtype=np.uint8)

    with pytest.raises(RuntimeError) as error:
        correr(imagen, redondea_a_multiplo, tile=256, overlap=64)

    # Los modelos redondean a multiplos de 8 o 64. Sin este chequeo el sintoma
    # seria una imagen corrida, no un error.
    assert "tile" in str(error.value)


def test_el_progreso_cuenta_los_tiles_hechos() -> None:
    imagen = np.zeros((600, 600, 3), dtype=np.uint8)
    avances: list[tuple[int, int]] = []

    def identidad(recorte, _plan):
        return recorte

    correr(
        imagen,
        identidad,
        tile=256,
        overlap=64,
        on_progress=lambda hechos, total: avances.append((hechos, total)),
    )

    # Un pase generativo tarda minutos: sin progreso por tile la barra se queda
    # quieta desde el primero hasta el ultimo.
    assert avances[0][0] == 1
    assert avances[-1][0] == avances[-1][1] == len(avances)


def test_cada_tile_recibe_su_propio_recorte() -> None:
    imagen = np.arange(600 * 600 * 3, dtype=np.uint8).reshape(600, 600, 3)
    vistos: list[tuple] = []

    def anota(recorte, plan):
        vistos.append((plan.y0, plan.x0, recorte.shape))
        return recorte

    correr(imagen, anota, tile=256, overlap=64)

    assert len({v[:2] for v in vistos}) == len(vistos)
    assert all(forma[:2] == (256, 256) for _, _, forma in vistos)


# ---------------------------------------------------------------------------
# Cuando pedirlo tiene sentido
# ---------------------------------------------------------------------------


def test_sin_imagen_de_partida_no_hay_a_que_agregarle_detalle() -> None:
    from app.services.generation_job_manager import GenerationJobManager

    with pytest.raises(ValueError):
        GenerationJobManager._validate_tiled_detail(True, None, None)


def test_no_se_combina_con_la_edicion_por_mascara(tmp_path) -> None:
    from app.services.generation_job_manager import GenerationJobManager

    with pytest.raises(ValueError):
        GenerationJobManager._validate_tiled_detail(
            True, tmp_path / "foto.png", tmp_path / "marca.png"
        )


def test_sin_pedirlo_no_se_valida_nada() -> None:
    from app.services.generation_job_manager import GenerationJobManager

    # Apagado es el caso normal: no puede exigir imagen de partida.
    GenerationJobManager._validate_tiled_detail(False, None, None)
