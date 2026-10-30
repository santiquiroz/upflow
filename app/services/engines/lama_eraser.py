from __future__ import annotations

import asyncio
import threading
from typing import Any

import numpy as np

from app.config import Settings
from app.services.inpaint_grain import match_grain
from app.services.inpaint_mask import (
    grow_per_component,
    grow_small_hulls,
    mask_bbox,
    soft_composite,
)
from app.services.inpaint_pipeline import MaskedEditSettings, edit_groups
from app.services.missing_pack import missing_pack_message

ERASER_LAMA_MODEL_ID = "eraser-lama"
ERASER_LAMA_MODEL_LABEL = "Borrado LaMa (continúa el fondo, sin IA generativa)"
MAX_WORK_SIDE = 1600
PROTECT_KERNEL_SIZE = 301


class LamaUnavailableError(RuntimeError):
    pass


def _mask_array(image: Any, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    gray = image.convert("L")
    if gray.size != size:
        gray = gray.resize(size, Image.Resampling.NEAREST)
    return np.asarray(gray)


def _unknown_hole(mask: np.ndarray, protect: np.ndarray) -> np.ndarray:
    import cv2

    hard = (mask > 0).astype(np.uint8)
    if not protect.any():
        return hard
    bbox = mask_bbox(hard * 255)
    if bbox is None:
        return hard
    left, top, right, bottom = bbox
    radius = PROTECT_KERNEL_SIZE // 2
    height, width = hard.shape
    window = (
        slice(max(0, top - radius), min(height, bottom + radius)),
        slice(max(0, left - radius), min(width, right + radius)),
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (PROTECT_KERNEL_SIZE, PROTECT_KERNEL_SIZE)
    )
    near = cv2.dilate(hard[window], kernel) > 0
    # El sujeto protegido también es desconocido para que no se copie en el fondo.
    hard[window] |= (protect[window] & near).astype(np.uint8)
    return hard


def _context_crop(hole: np.ndarray) -> tuple[int, int, int, int]:
    left, top, right, bottom = mask_bbox(hole * 255)
    margin = round(0.5 * max(right - left, bottom - top))
    height, width = hole.shape
    return (
        max(0, left - margin),
        max(0, top - margin),
        min(width, right + margin),
        min(height, bottom + margin),
    )


def _work_arrays(image: np.ndarray, hole: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    height, width = hole.shape
    scale = min(1.0, MAX_WORK_SIDE / max(width, height))
    if scale == 1.0:
        return image, hole
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return (
        cv2.resize(image, size, interpolation=cv2.INTER_AREA),
        cv2.resize(hole, size, interpolation=cv2.INTER_NEAREST),
    )


def _model_inputs(image: np.ndarray, hole: np.ndarray) -> tuple[Any, Any]:
    import torch

    height, width = hole.shape
    padding = ((0, -height % 8), (0, -width % 8))
    padded_image = np.pad(image, padding + ((0, 0),), mode="reflect")
    padded_hole = np.pad(hole, padding, mode="constant")
    rgb = padded_image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    mask = padded_hole[None, None].astype(np.float32)
    return torch.from_numpy(rgb), torch.from_numpy(mask)


def _run_model(module: Any, image: np.ndarray, hole: np.ndarray) -> np.ndarray:
    import cv2
    import torch

    work_image, work_hole = _work_arrays(image, hole)
    image_tensor, mask_tensor = _model_inputs(work_image, work_hole)
    with torch.inference_mode():
        output = module(image_tensor, mask_tensor)
    height, width = work_hole.shape
    generated = output[0, :, :height, :width].detach().cpu().numpy().transpose(1, 2, 0)
    generated = np.clip(np.rint(generated * 255.0), 0, 255).astype(np.uint8)
    if generated.shape[:2] != hole.shape:
        generated = cv2.resize(
            generated, (hole.shape[1], hole.shape[0]), interpolation=cv2.INTER_CUBIC
        )
    return generated


def _erase_group(
    canvas: np.ndarray, group_mask: np.ndarray, protect: np.ndarray, module: Any
) -> None:
    hole = _unknown_hole(group_mask, protect)
    left, top, right, bottom = _context_crop(hole)
    window = (slice(top, bottom), slice(left, right))
    original = canvas[window].copy()
    blend = np.where(protect[window], 0, group_mask[window]).astype(np.uint8)
    generated = _run_model(module, original, hole[window])
    stitched = soft_composite(generated, original, blend)
    canvas[window] = match_grain(stitched, original, blend)


class LamaEraser:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._module: Any = None

    def available(self) -> bool:
        return self.settings.lama_available()

    async def erase(
        self, base_image: Any, mask_image: Any, device: str, protect_image: Any = None
    ) -> Any:
        if not self.available():
            raise LamaUnavailableError(missing_pack_message("lama"))
        return await asyncio.to_thread(
            self._erase_blocking, base_image, mask_image, protect_image
        )

    def _erase_blocking(
        self, base_image: Any, mask_image: Any, protect_image: Any = None
    ) -> Any:
        from PIL import Image

        base_rgb = base_image.convert("RGB")
        mask = _mask_array(mask_image, base_rgb.size)
        if mask_bbox(mask) is None:
            return base_rgb
        blended = grow_per_component(grow_small_hulls(mask), None, None)
        protect = np.zeros(mask.shape, dtype=bool)
        if protect_image is not None:
            protect = _mask_array(protect_image, base_rgb.size) > 127
        settings = MaskedEditSettings(None, None, None, MAX_WORK_SIDE)
        groups = edit_groups(blended, base_rgb.size, settings)
        module = self._get_module()
        canvas = np.array(base_rgb, copy=True)
        for _, group_mask in groups:
            _erase_group(canvas, group_mask, protect, module)
        return Image.fromarray(canvas)

    def _get_module(self) -> Any:
        with self._lock:
            if self._module is None:
                self._module = self._create_module()
            return self._module

    def _create_module(self) -> Any:
        import torch

        return torch.jit.load(str(self.settings.lama_model_path), map_location="cpu").eval()
