"""Clase de velocidad de un modelo de generación y anclaje de parámetros.

Los modelos destilados (Turbo/Lightning/LCM) generan en 1-8 pasos pero se
ROMPEN con los defaults de un modelo normal: Turbo/Lightning corren sin CFG
(guidance 0 — con 7 producen basura quemada) y LCM necesita su scheduler y
guidance ~1-2. La clase se deriva del nombre del repo: no hay migración de
registry y los repos reales la llevan en el id (sdxl-turbo, SDXL-Lightning,
lcm-lora-sdxl...).
"""

from __future__ import annotations

SPEED_TURBO = "turbo"
SPEED_LIGHTNING = "lightning"
SPEED_LCM = "lcm"

_SPEED_TOKENS: tuple[tuple[str, str], ...] = (
    ("turbo", SPEED_TURBO),
    ("lightning", SPEED_LIGHTNING),
    ("hyper-sd", SPEED_LIGHTNING),
    ("hypersd", SPEED_LIGHTNING),
    ("lcm", SPEED_LCM),
    ("latent-consistency", SPEED_LCM),
)


def speed_class(model_name: str) -> str | None:
    # Solo el segmento final del repo id: una org llamada "TurboVision" no
    # convierte a todos sus modelos en destilados (falso positivo confirmado
    # en review adversarial).
    lowered = model_name.rsplit("/", 1)[-1].lower()
    for token, speed in _SPEED_TOKENS:
        if token in lowered:
            return speed
    return None


def anchored_params(
    speed: str | None, steps: int, guidance: float, scheduler: str | None
) -> tuple[int, float, str | None]:
    """Parámetros seguros para la clase de velocidad del modelo.

    Solo APRIETA (clamp): un usuario que pidió 2 pasos en un Turbo los
    conserva. El techo de guidance es 2.0 y no 0: los finetunes destilados
    reales (DreamShaper XL Turbo, RealVis Turbo) documentan CFG 1.5-2.5, y
    con guidance <= 1 diffusers ignora el negative prompt en silencio.
    """
    if speed == SPEED_TURBO:
        # El scheduler correcto ya viene en el scheduler_config.json del repo.
        return min(steps, 8), min(guidance, 2.0), scheduler
    if speed == SPEED_LIGHTNING:
        return min(steps, 8), min(guidance, 2.0), scheduler or "euler_trailing"
    if speed == SPEED_LCM:
        return min(steps, 8), min(guidance, 2.0), scheduler or "lcm"
    return steps, guidance, scheduler
