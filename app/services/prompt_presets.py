"""Prompts listos para cada modo de generacion.

Sigue el patron de VIDEO_PROFILE_CATALOG: una tabla de datos, no logica.

La distincion que importa:
  - El NOMBRE del preset es copia, y viaja como clave de traduccion, igual que
    los presets de Tiempo real y de Acabado.
  - El PROMPT en si NO es copia: es el texto que se le manda al modelo. Se manda
    tal cual y no se traduce nunca — traducir "cinematic lighting, 35mm film"
    cambiaria lo que el modelo genera.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptPreset:
    id: str
    mode: str
    label_key: str
    prompt: str
    negative_prompt: str = ""


# Negativo compartido por los presets fotorrealistas: son los defectos que mas
# arruinan una imagen generada.
_PHOTO_NEGATIVE = "blurry, low quality, watermark, text, extra fingers, deformed hands"

PROMPT_PRESETS: tuple[PromptPreset, ...] = (
    # --- texto a imagen -----------------------------------------------------
    PromptPreset(
        id="portrait",
        mode="text-to-image",
        label_key="generate.preset.portrait",
        prompt=(
            "portrait photograph, natural soft window light, shallow depth of field, "
            "85mm lens, sharp eyes, realistic skin texture"
        ),
        negative_prompt=_PHOTO_NEGATIVE,
    ),
    PromptPreset(
        id="cinematic",
        mode="text-to-image",
        label_key="generate.preset.cinematic",
        prompt=(
            "cinematic still, anamorphic lens, dramatic rim lighting, film grain, "
            "35mm, muted teal and orange palette"
        ),
        negative_prompt=_PHOTO_NEGATIVE,
    ),
    PromptPreset(
        id="anime",
        mode="text-to-image",
        label_key="generate.preset.anime",
        prompt=(
            "anime illustration, clean line art, cel shading, vibrant colours, "
            "detailed background, studio quality"
        ),
        negative_prompt="blurry, low quality, watermark, text, extra limbs",
    ),
    PromptPreset(
        id="landscape",
        mode="text-to-image",
        label_key="generate.preset.landscape",
        prompt=(
            "wide landscape photograph, golden hour light, high dynamic range, "
            "deep focus, ultra detailed"
        ),
        negative_prompt=_PHOTO_NEGATIVE,
    ),
    PromptPreset(
        id="product",
        mode="text-to-image",
        label_key="generate.preset.product",
        prompt=(
            "product photograph on seamless background, soft box lighting, "
            "crisp reflections, centred composition, commercial quality"
        ),
        negative_prompt=_PHOTO_NEGATIVE,
    ),
    # --- imagen a imagen ----------------------------------------------------
    PromptPreset(
        id="restyle-painting",
        mode="image-to-image",
        label_key="generate.preset.restylePainting",
        prompt="oil painting, visible brush strokes, canvas texture, rich impasto",
        negative_prompt="blurry, low quality, watermark, text",
    ),
    PromptPreset(
        id="restyle-anime",
        mode="image-to-image",
        label_key="generate.preset.restyleAnime",
        prompt="anime style, cel shading, clean line art, flat vibrant colours",
        negative_prompt="blurry, low quality, watermark, text",
    ),
    PromptPreset(
        id="restyle-photo",
        mode="image-to-image",
        label_key="generate.preset.restylePhoto",
        prompt="photorealistic, natural lighting, realistic materials, sharp detail",
        negative_prompt=_PHOTO_NEGATIVE,
    ),
    # --- video --------------------------------------------------------------
    # Los prompts de video describen MOVIMIENTO: sin eso el clip sale como una
    # foto que tiembla.
    PromptPreset(
        id="video-slow-pan",
        mode="video",
        label_key="generate.preset.videoSlowPan",
        prompt="slow cinematic camera pan across the scene, steady motion, soft natural light",
    ),
    PromptPreset(
        id="video-closeup",
        mode="video",
        label_key="generate.preset.videoCloseup",
        prompt="close-up shot, subtle head movement, shallow depth of field, gentle breathing motion",
    ),
    PromptPreset(
        id="video-nature",
        mode="video",
        label_key="generate.preset.videoNature",
        prompt="nature footage, leaves moving in the wind, drifting clouds, flowing water, ambient daylight",
    ),
)


def presets_for_mode(mode: str) -> list[PromptPreset]:
    return [preset for preset in PROMPT_PRESETS if preset.mode == mode]
