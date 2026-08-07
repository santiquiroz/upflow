from __future__ import annotations

from app.services.generation_speed import (
    SPEED_LCM,
    SPEED_LIGHTNING,
    SPEED_TURBO,
    anchored_params,
    speed_class,
)


def test_speed_class_detects_families_from_repo_names() -> None:
    assert speed_class("stabilityai/sdxl-turbo") == SPEED_TURBO
    assert speed_class("ByteDance/SDXL-Lightning") == SPEED_LIGHTNING
    assert speed_class("ByteDance/Hyper-SD") == SPEED_LIGHTNING
    assert speed_class("SimianLuo/LCM_Dreamshaper_v7") == SPEED_LCM
    assert speed_class("latent-consistency/lcm-sdxl") == SPEED_LCM
    assert speed_class("John6666/epicrealism-xl-vxvi") is None


def test_speed_class_ignores_the_org_segment() -> None:
    # Una org llamada "TurboVision" no vuelve destilados a todos sus modelos.
    assert speed_class("TurboVision/realistic-base") is None
    assert speed_class("stabilityai/sdxl-turbo") == SPEED_TURBO


def test_turbo_clamps_steps_and_caps_guidance() -> None:
    # Un Turbo con guidance 7 produce basura quemada: el anclaje protege. El
    # techo es 2.0 y no 0: los finetunes turbo documentan CFG 1.5-2.5, y con
    # guidance <= 1 diffusers ignora el negative prompt.
    assert anchored_params(SPEED_TURBO, 30, 7.5, None) == (8, 2.0, None)


def test_turbo_keeps_a_lower_user_choice() -> None:
    assert anchored_params(SPEED_TURBO, 2, 0.0, None) == (2, 0.0, None)
    assert anchored_params(SPEED_TURBO, 6, 1.8, None) == (6, 1.8, None)


def test_lightning_forces_trailing_scheduler_by_default() -> None:
    assert anchored_params(SPEED_LIGHTNING, 25, 7.0, None) == (8, 2.0, "euler_trailing")


def test_lcm_caps_guidance_and_uses_lcm_scheduler() -> None:
    assert anchored_params(SPEED_LCM, 25, 7.0, None) == (8, 2.0, "lcm")


def test_explicit_scheduler_choice_wins_over_the_default() -> None:
    _, _, scheduler = anchored_params(SPEED_LCM, 8, 1.5, "euler_a")
    assert scheduler == "euler_a"


def test_standard_models_pass_through_untouched() -> None:
    assert anchored_params(None, 25, 7.5, None) == (25, 7.5, None)
