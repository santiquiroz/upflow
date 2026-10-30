from __future__ import annotations

from collections.abc import Callable

import numpy as np
from PIL import Image

from app.services.fill_faces import faces_inside
from app.services.fill_people import PeopleFinder
from app.services.inpaint_mask import (
    clean_protect_mask,
    grow_per_component,
    grow_small_hulls,
    mask_bbox,
    matte_core,
    matte_radius,
    subject_matte,
)
from app.services.inpaint_pipeline import MaskedEditSettings, edit_groups
from app.services.inpaint_score import fill_score

LamaFill = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
SdxlEdit = Callable[[Image.Image, np.ndarray, float], Image.Image]
FaceFinder = Callable[[np.ndarray, np.ndarray], list[tuple[int, int, int, int]]]


def _has_people(
    find_people: PeopleFinder | None, rgb: np.ndarray, mask: np.ndarray
) -> bool:
    return bool(find_people(rgb, mask)) if find_people is not None else False


def _fill_large_with_osor(
    current: Image.Image,
    group_mask: np.ndarray,
    core: np.ndarray,
    lama_fill: LamaFill,
    osor_fill: LamaFill,
    find_faces: FaceFinder,
    find_people: PeopleFinder | None,
) -> tuple[np.ndarray, dict]:
    filled = osor_fill(np.array(current, copy=True), group_mask, core)
    has_face = bool(find_faces(filled, group_mask))
    has_person = _has_people(find_people, filled, group_mask)
    if not has_face and not has_person:
        return filled, {"engine": "osor", "guard": None, "residualPerson": False}

    prefilled = lama_fill(np.array(current, copy=True), group_mask, core)
    retried = osor_fill(prefilled, group_mask, core)
    residual_face = bool(find_faces(retried, group_mask))
    residual_person = _has_people(find_people, retried, group_mask)
    return retried, {
        "engine": "lama+osor",
        "guard": "face" if has_face else "person",
        "residualPerson": residual_face or residual_person,
    }


def _composite_subject(
    image: Image.Image, original: np.ndarray, alpha: np.ndarray
) -> Image.Image:
    generated = np.asarray(image.convert("RGB"), dtype=np.float32)
    weight = alpha[:, :, None]
    blended = weight * original + (1.0 - weight) * generated
    return Image.fromarray(np.clip(np.rint(blended), 0, 255).astype(np.uint8))


def _choose_large_fill(
    current: Image.Image,
    group_mask: np.ndarray,
    core: np.ndarray,
    lama_fill: LamaFill,
    sdxl_edit: SdxlEdit,
    hybrid_strength: float,
    original: np.ndarray,
    alpha: np.ndarray,
) -> tuple[Image.Image, dict]:
    prefilled = Image.fromarray(lama_fill(np.array(current, copy=True), group_mask, core))
    prefilled = _composite_subject(prefilled, original, alpha)
    hybrid = sdxl_edit(prefilled, group_mask, hybrid_strength)
    hybrid = _composite_subject(hybrid, original, alpha)
    direct = sdxl_edit(current.copy(), group_mask, 1.0)
    direct = _composite_subject(direct, original, alpha)
    hard_mask = group_mask > 0
    scores = {
        "hibrido": fill_score(np.asarray(hybrid), hard_mask, core),
        "sdxl": fill_score(np.asarray(direct), hard_mask, core),
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
    osor_fill: LamaFill | None = None,
    find_faces: FaceFinder = faces_inside,
    find_people: PeopleFinder | None = None,
) -> tuple[Image.Image, list[dict]]:
    current = base_rgb.convert("RGB")
    original = np.array(current, copy=True)
    protect = (
        np.zeros(mask.shape, dtype=bool)
        if protect is None else clean_protect_mask(protect.astype(bool))
    )
    radius = matte_radius(protect.shape)
    alpha = subject_matte(original, protect, radius)
    core = matte_core(protect, radius)
    mask = np.where(core, 0, mask).astype(np.uint8)
    if mask_bbox(mask) is None:
        return current, []
    prepared = grow_per_component(grow_small_hulls(mask), None, None)
    prepared[core] = 0
    settings = MaskedEditSettings(0, 0, None, 1600)
    choices = []
    groups = edit_groups(prepared, current.size, settings)
    for index, (box, group_mask) in enumerate(groups):
        left, top, right, bottom = mask_bbox((group_mask > 0).astype(np.uint8) * 255)
        if max(right - left, bottom - top) <= small_side:
            filled = lama_fill(np.array(current, copy=True), group_mask, core)
            current, choice = Image.fromarray(filled), {"engine": "lama"}
        elif osor_fill is not None:
            filled, choice = _fill_large_with_osor(
                current, group_mask, core, lama_fill, osor_fill, find_faces, find_people
            )
            current = Image.fromarray(filled)
        else:
            current, choice = _choose_large_fill(
                current, group_mask, core, lama_fill, sdxl_edit, hybrid_strength,
                original, alpha,
            )
        current = _composite_subject(current, original, alpha)
        choice["index"] = index
        choice["box"] = [int(coordinate) for coordinate in box]
        choices.append(choice)
    return current, choices
