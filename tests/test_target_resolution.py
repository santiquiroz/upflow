from __future__ import annotations

import pytest

from app.services.target_resolution import (
    TARGET_PRESETS,
    megapixels_per_frame,
    plan_for_scale,
    plan_for_target,
    resolve_target_height,
    smallest_scale_reaching,
)

# El caso real que motivo todo esto: un 4K con escala 4 pidio 15360x8640 y tardo
# 2,8 horas solo en escalar, sin que nada avisara.
UHD = (3840, 2160)
FHD = (1920, 1080)
SD_600P = (1067, 600)


# ---------------------------------------------------------------------------
# Elegir el escalado
# ---------------------------------------------------------------------------


def test_a_source_that_already_reaches_the_target_needs_no_model():
    """El caso que convierte horas en segundos.

    Un 4K que se quiere en 1080p no necesita modelo: es un resize de ffmpeg. Hoy la
    app corre el modelo igual porque solo entiende multiplicadores.
    """
    assert smallest_scale_reaching(2160, 1080, (2, 3, 4)) is None


def test_a_source_exactly_at_the_target_needs_no_model():
    assert smallest_scale_reaching(1080, 1080, (2, 3, 4)) is None


def test_the_smallest_scale_that_reaches_is_chosen():
    # 540 -> 1080 alcanza con 2x. Correr 4x para despues bajar gastaria horas de GPU
    # en pixeles que se tiran.
    assert smallest_scale_reaching(540, 1080, (2, 3, 4)) == 2


def test_a_scale_that_overshoots_is_used_when_the_smaller_one_falls_short():
    # 500 x2 = 1000 < 1080, asi que hace falta 3x y despues bajar.
    assert smallest_scale_reaching(500, 1080, (2, 3, 4)) == 3


def test_the_max_scale_is_used_when_nothing_reaches():
    # 200 x4 = 800 < 2160: ningun escalado llega, se usa el mayor.
    assert smallest_scale_reaching(200, 2160, (2, 3, 4)) == 4


def test_a_model_with_a_single_scale_is_respected():
    # Varios modelos del catalogo solo ofrecen 4x.
    assert smallest_scale_reaching(540, 1080, (4,)) == 4


# ---------------------------------------------------------------------------
# Resolver el objetivo
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,height", sorted(TARGET_PRESETS.items()))
def test_every_preset_resolves_to_its_height(name: str, height: int):
    assert resolve_target_height(name) == height


def test_an_explicit_height_is_accepted():
    assert resolve_target_height(900) == 900


def test_an_unknown_preset_is_rejected_loudly():
    with pytest.raises(ValueError, match="Unknown target"):
        resolve_target_height("8k")


def test_a_non_positive_height_is_rejected():
    with pytest.raises(ValueError):
        resolve_target_height(0)


# ---------------------------------------------------------------------------
# El plan completo
# ---------------------------------------------------------------------------


def test_downscaling_4k_to_1080p_runs_no_model_at_all():
    plan = plan_for_target(*UHD, "1080p")

    assert plan.model_scale is None
    assert (plan.output_width, plan.output_height) == (1920, 1080)
    assert plan.needs_resize is True
    assert plan.exceeds_model_reach is False


def test_upscaling_600p_to_1080p_uses_two_x_and_then_resizes():
    # 600 x2 = 1200 > 1080, asi que sobra y hay que bajar a la medida exacta.
    plan = plan_for_target(*SD_600P, "1080p")

    assert plan.model_scale == 2
    assert plan.output_height == 1080
    assert plan.needs_resize is True


def test_a_clean_double_needs_no_resize():
    plan = plan_for_target(960, 540, "1080p")

    assert plan.model_scale == 2
    assert (plan.output_width, plan.output_height) == (1920, 1080)
    # 540 x2 da exactamente 1080: redimensionar seria una pasada de ffmpeg al vicio.
    assert plan.needs_resize is False


def test_the_aspect_ratio_is_preserved():
    plan = plan_for_target(1440, 1080, "2160p")  # 4:3
    assert plan.output_width / plan.output_height == pytest.approx(1440 / 1080, abs=0.01)


def test_odd_dimensions_are_rounded_to_even():
    # yuv420p necesita dimensiones pares: un ancho impar hace fallar el encode.
    plan = plan_for_target(1235, 555, "1080p")
    assert plan.output_width % 2 == 0
    assert plan.output_height % 2 == 0


def test_a_target_beyond_the_model_reach_is_flagged():
    # 200p a 2160p: ni 4x llega. Se entrega lo mejor posible y se AVISA, en vez de
    # agrandar con ffmpeg en silencio y hacer pasar el desenfoque por detalle.
    plan = plan_for_target(356, 200, "2160p")

    assert plan.model_scale == 4
    assert plan.exceeds_model_reach is True
    assert plan.output_height == 2160


def test_a_reachable_target_is_not_flagged():
    plan = plan_for_target(*SD_600P, "1080p")
    assert plan.exceeds_model_reach is False


def test_zero_or_negative_source_dimensions_are_rejected():
    for width, height in ((0, 1080), (1920, 0), (-1, 100)):
        with pytest.raises(ValueError):
            plan_for_target(width, height, "1080p")


# ---------------------------------------------------------------------------
# El costo del camino viejo contra el nuevo
# ---------------------------------------------------------------------------


def test_the_blind_multiplier_reproduces_the_case_that_motivated_this():
    plan = plan_for_scale(*UHD, 4)

    assert (plan.output_width, plan.output_height) == (15360, 8640)
    assert megapixels_per_frame(plan.output_width, plan.output_height) == pytest.approx(
        132.7, abs=0.1
    )


def test_the_target_path_is_dramatically_cheaper_for_the_same_intent():
    """Mismo pedido razonable, dos ordenes de magnitud de diferencia.

    Alguien con un 4K que quiere "mejor calidad" y elige x4 pide 132.7 MP por frame.
    Si en cambio pide 1080p, son 2.1 MP y ni corre el modelo.
    """
    blind = plan_for_scale(*UHD, 4)
    targeted = plan_for_target(*UHD, "1080p")

    blind_cost = megapixels_per_frame(blind.output_width, blind.output_height)
    targeted_cost = megapixels_per_frame(targeted.output_width, targeted.output_height)

    assert blind_cost / targeted_cost > 60
    assert targeted.model_scale is None


def test_upscaling_a_small_source_is_far_cheaper_than_upscaling_a_large_one():
    # El diagnostico del caso real: no era lento por ir de chico a grande, era lento
    # por ir de 4K a 16K.
    small = plan_for_scale(*SD_600P, 4)
    large = plan_for_scale(*UHD, 4)

    small_cost = megapixels_per_frame(small.output_width, small.output_height)
    large_cost = megapixels_per_frame(large.output_width, large.output_height)

    assert large_cost / small_cost > 12
