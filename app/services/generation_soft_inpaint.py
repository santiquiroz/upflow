"""Soft inpainting: denoise proporcional al gris de la máscara, por paso.

El pipeline de diffusers BINARIZA la máscara al preparar los latents: el
degradado del feather solo actúa al PEGAR el resultado, no al generar — el
borde del parche se regenera al 100% igual que el centro, y de ahí la costura
y los bordes que no calzan. Acá se re-inyecta el latente del original,
ruidificado al nivel del timestep SIGUIENTE, mezclado por píxel con la máscara
suave: centro de la marca regenera todo, la banda del feather conserva
progresivamente el original, afuera queda intacto.

Corre en el callback del loop de denoising: diffusers pasa el tensor de
latents VIVO al callback viejo (i, t, latents) y la mutación in-place es
visible en los pasos siguientes. Se mezcla al nivel de ruido del timestep
SIGUIENTE porque el callback corre DESPUÉS de scheduler.step (los latents ya
están un paso más limpios que `t`); en el último paso no se mezcla — los
latents ya son la imagen final y ruidificarlos la mancharía.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

SoftInpaintCallback = Callable[[int, Any, Any], None]


def build_soft_inpaint_callback(
    pipeline: Any, image: Any, mask: Any, seed: int | None
) -> SoftInpaintCallback:
    import numpy as np
    import torch

    image_t = pipeline.image_processor.preprocess(image)
    generator = torch.Generator().manual_seed(seed if seed is not None else 0)
    original_latents = pipeline._encode_vae_image(image_t, generator=generator)

    mask_np = np.asarray(mask.convert("L"), dtype=np.float32) / 255.0
    mask_t = torch.from_numpy(mask_np)[None, None]
    latent_mask = torch.nn.functional.interpolate(
        mask_t, size=original_latents.shape[-2:], mode="bilinear", align_corners=False
    )
    noise = torch.randn(
        original_latents.shape, generator=generator, dtype=torch.float32
    ).to(original_latents.dtype)

    def blend(_step_index: int, timestep: Any, latents: Any) -> None:
        timesteps = pipeline.scheduler.timesteps
        t = timestep if torch.is_tensor(timestep) else torch.tensor(timestep)
        matches = (timesteps == t).nonzero()
        if len(matches) == 0:
            return
        next_index = int(matches[0]) + 1
        if next_index >= len(timesteps):
            return
        next_t = timesteps[next_index].reshape(1)
        noised = pipeline.scheduler.add_noise(original_latents, noise, next_t)
        weight = latent_mask.to(latents.dtype)
        latents.copy_(latents * weight + noised * (1.0 - weight))

    return blend


def chain_step_callbacks(
    soft_callback: SoftInpaintCallback | None,
    progress_callback: Callable[[int, Any, Any], None],
) -> Callable[[int, Any, Any], None]:
    """Un solo callback para el pipeline: mezcla suave primero, progreso después.

    Un fallo del blend deshabilita el soft inpainting para el resto del job
    (con warning: un no-op silencioso sería indistinguible de "funciona") pero
    JAMÁS tumba la generación.
    """
    state = {"soft": soft_callback}

    def step(step_index: int, timestep: Any, latents: Any) -> None:
        soft = state["soft"]
        if soft is not None:
            try:
                soft(step_index, timestep, latents)
            except Exception:  # noqa: BLE001 -- calidad extra jamás tumba el job
                state["soft"] = None
                logger.warning(
                    "soft inpainting deshabilitado a mitad del job (falló el blend)",
                    exc_info=True,
                )
        progress_callback(step_index, timestep, latents)

    return step
