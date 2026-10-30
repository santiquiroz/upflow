from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image

from app.services.inpaint_mask import (
    clean_protect_mask,
    grow_per_component,
    grow_small_hulls,
    mask_bbox,
)
from app.services.inpaint_pipeline import MaskedEditSettings, edit_groups
from app.services.inpaint_score import fill_score

LamaFill = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
SdxlEdit = Callable[[Image.Image, np.ndarray, float], Image.Image]


def _restore_protected(
    image: Image.Image, original: np.ndarray, protect: np.ndarray
) -> Image.Image:
    restored = np.array(image.convert("RGB"), copy=True)
    restored[protect] = original[protect]
    return Image.fromarray(restored)


def _choose_large_fill(
    current: Image.Image,
    group_mask: np.ndarray,
    protect: np.ndarray,
    lama_fill: LamaFill,
    sdxl_edit: SdxlEdit,
    hybrid_strength: float,
) -> tuple[Image.Image, dict]:
    original = np.array(current, copy=True)
    prefilled = Image.fromarray(lama_fill(original.copy(), group_mask, protect))
    prefilled = _restore_protected(prefilled, original, protect)
    hybrid = sdxl_edit(prefilled, group_mask, hybrid_strength)
    hybrid = _restore_protected(hybrid, original, protect)
    direct = sdxl_edit(current.copy(), group_mask, 1.0)
    direct = _restore_protected(direct, original, protect)
    hard_mask = group_mask > 0
    scores = {
        "hibrido": fill_score(np.asarray(hybrid), hard_mask, protect),
        "sdxl": fill_score(np.asarray(direct), hard_mask, protect),
    }
    engine = min(scores, key=lambda name: scores[name]["score"])
    chosen = hybrid if engine == "hibrido" else direct
    return chosen, {"engine": engine, "scores": scores}


def auto_erase(
    base_rgb: Image.Image,
    mask: np.ndarray,
    protect: np.ndarray | None,
    lama_fill: LamaFill,
    sdxl_edit: SdxlEdit,
    small_side: int = 700,
    hybrid_strength: float = 0.5,
) -> tuple[Image.Image, list[dict]]:
    current = base_rgb.convert("RGB")
    original = np.array(current, copy=True)
    protect = (
        np.zeros(mask.shape, dtype=bool)
        if protect is None else clean_protect_mask(protect.astype(bool))
    )
    mask = np.where(protect, 0, mask).astype(np.uint8)
    if mask_bbox(mask) is None:
        return current, []
    prepared = grow_per_component(grow_small_hulls(mask), None, None)
    prepared[protect] = 0
    settings = MaskedEditSettings(0, 0, None, 1600)
    choices = []
    for _, group_mask in edit_groups(prepared, current.size, settings):
        left, top, right, bottom = mask_bbox((group_mask > 0).astype(np.uint8) * 255)
        if max(right - left, bottom - top) <= small_side:
            filled = lama_fill(np.array(current, copy=True), group_mask, protect)
            current, choice = Image.fromarray(filled), {"engine": "lama"}
        else:
            current, choice = _choose_large_fill(
                current, group_mask, protect, lama_fill, sdxl_edit, hybrid_strength
            )
        current = _restore_protected(current, original, protect)
        choices.append(choice)
    return current, choices
