from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from app.config import Settings
from app.services.inpaint_grain import match_grain

ERASER_OSOR_MODEL_ID = "eraser-osor"
OSOR_RESOLUTION = 512
OSOR_TIMESTEP = 400
OSOR_SEED = 42
OSOR_CONTEXT = 1.5
OSOR_CONTEXT_PAD = 64
OSOR_GROW_RATIO = 0.03
OSOR_FEATHER = 0.003
OSOR_SHIFT_SIGMA = 0.06
TOUCHED_GROW_RATIO = 0.02
REMOVED_GROW_RATIO = 0.004
ALPHA_REMOVE = 0.5
ALPHA_TOUCH = 0.1


def grow(mask: np.ndarray, ratio: float) -> np.ndarray:
    import cv2

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return mask.astype(bool)
    side = max(np.ptp(ys), np.ptp(xs)) + 1
    radius = max(8, int(round(ratio * side)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def osor_crop_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return 0, 0, 0, 0
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    height, width = mask.shape
    side = int(max(x1 - x0, y1 - y0) * OSOR_CONTEXT + OSOR_CONTEXT_PAD)
    crop_width = min(width, max(side, x1 - x0))
    crop_height = min(height, max(side, y1 - y0))
    center_x, center_y = (x0 + x1) // 2, (y0 + y1) // 2
    left = int(np.clip(center_x - crop_width // 2, 0, width - crop_width))
    top = int(np.clip(center_y - crop_height // 2, 0, height - crop_height))
    return left, top, left + crop_width, top + crop_height


def remove_render_shift(
    filled: np.ndarray, original: np.ndarray, changed: np.ndarray
) -> np.ndarray:
    import cv2

    known = (~changed).astype(np.float32)
    sigma = OSOR_SHIFT_SIGMA * max(filled.shape[:2])
    num = cv2.GaussianBlur((filled - original) * known[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(known, (0, 0), sigma)[..., None]
    fallback = (
        np.median((filled - original)[known > 0], axis=0)
        if known.any()
        else np.zeros(3, np.float32)
    )
    shift = np.where(den > 0.05, num / np.maximum(den, 1e-6), fallback)
    return filled - shift


def calc_dims(width: int, height: int, resolution: int) -> tuple[int, int]:
    if min(width, height) < resolution:
        return (width + 15) // 16 * 16, (height + 15) // 16 * 16
    if width < height:
        return resolution, (int(height * resolution / width) + 15) // 16 * 16
    return (int(width * resolution / height) + 15) // 16 * 16, resolution


def _resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    resized = Image.fromarray(np.clip(np.rint(image * 255), 0, 255).astype(np.uint8))
    return np.asarray(
        resized.resize(size, Image.Resampling.LANCZOS), dtype=np.uint8
    )


def _resize_alpha(alpha: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    resized = Image.fromarray(np.clip(np.rint(alpha * 255), 0, 255).astype(np.uint8))
    return np.asarray(
        resized.resize(size, Image.Resampling.BILINEAR), dtype=np.float32
    ) / 255.0


class _OsorModel:
    def __init__(self, pack_path: Path) -> None:
        import torch
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

        self.torch = torch
        load_args = {
            "variant": "fp16",
            "torch_dtype": torch.float32,
            "local_files_only": True,
        }
        self.unet = UNet2DConditionModel.from_pretrained(
            pack_path / "unet", **load_args
        ).eval()
        self.vae = AutoencoderKL.from_pretrained(
            pack_path / "vae", **load_args
        ).eval()
        self.scheduler = DDPMScheduler.from_pretrained(
            pack_path / "scheduler", local_files_only=True
        )
        self.embeddings = torch.load(
            pack_path / "prompt_embeds.pt",
            map_location="cpu",
            weights_only=True,
        )

    def infer(
        self, image_rgb: np.ndarray, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        height, width = mask.shape
        image_tensor = torch.from_numpy(
            image_rgb.transpose(2, 0, 1).copy()
        ).unsqueeze(0).float() / 127.5 - 1
        mask_tensor = torch.from_numpy((mask > 0).astype(np.float32))[
            None, None
        ]
        timestep = torch.tensor([OSOR_TIMESTEP])
        prompt_embeds = self.embeddings["prompt_embeds"].float()
        pooled_prompt_embeds = self.embeddings["pooled_prompt_embeds"].float()
        time_ids = torch.tensor(
            [[height, width, 0, 0, height, width]], dtype=torch.float32
        )

        with torch.inference_mode():
            torch.manual_seed(OSOR_SEED)
            latent_lq = (
                self.vae.encode(image_tensor).latent_dist.sample()
                * self.vae.config.scaling_factor
            )
            latent = self.scheduler.add_noise(
                latent_lq, torch.randn_like(latent_lq), timestep
            )
            unet_input = torch.cat(
                [latent, torch.nn.functional.max_pool2d(mask_tensor, 8, 8), latent_lq],
                dim=1,
            )
            output = self.unet(
                unet_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={
                    "text_embeds": pooled_prompt_embeds,
                    "time_ids": time_ids,
                },
            ).sample
            alpha = output[:, 4:].sigmoid()
            restored = self.scheduler.step(
                output[:, :4], timestep, latent
            ).pred_original_sample
            restored = restored * alpha + latent_lq * (1 - alpha)
            decoded = self.vae.decode(
                restored / self.vae.config.scaling_factor
            ).sample
            pixels = ((decoded + 1) / 2).clamp(0, 1)[0].permute(1, 2, 0)
            opacity = torch.nn.functional.interpolate(
                alpha, size=(height, width), mode="bilinear"
            )[0, 0]
        if not bool(torch.isfinite(pixels).all()) or not bool(
            torch.isfinite(opacity).all()
        ):
            raise ValueError("OSOR produjo valores no finitos")
        return (
            pixels.cpu().numpy().astype(np.float32),
            opacity.cpu().numpy().astype(np.float32),
        )


class OsorEraser:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._model: Any = None

    def available(self) -> bool:
        return self.settings.osor_available()

    def fill_group(
        self, canvas: np.ndarray, group_mask: np.ndarray, protect: np.ndarray
    ) -> np.ndarray:
        mask = grow(group_mask > 0, OSOR_GROW_RATIO) & ~protect
        result = canvas.copy()
        if not mask.any():
            return result
        left, top, right, bottom = osor_crop_box(mask)
        window = (slice(top, bottom), slice(left, right))
        original = canvas[window].copy()
        mask_crop = mask[window]
        protect_crop = protect[window]
        crop_height, crop_width = mask_crop.shape
        input_width, input_height = calc_dims(
            crop_width, crop_height, OSOR_RESOLUTION
        )
        input_size = (input_width, input_height)
        image_input = np.asarray(
            Image.fromarray(original).resize(
                input_size, Image.Resampling.BICUBIC
            ),
            dtype=np.uint8,
        )
        mask_input = np.asarray(
            Image.fromarray(mask_crop.astype(np.uint8) * 255).resize(
                input_size, Image.Resampling.NEAREST
            ),
            dtype=np.uint8,
        ) > 0
        rgb_input, alpha_input = self._get_model().infer(image_input, mask_input)
        filled = _resize_rgb(rgb_input, (crop_width, crop_height)).astype(np.float32)
        learned = _resize_alpha(alpha_input, (crop_width, crop_height))
        removed = (mask_crop | (learned > ALPHA_REMOVE)) & ~protect_crop
        touched = grow(
            mask_crop | (learned > ALPHA_TOUCH), TOUCHED_GROW_RATIO
        ) | protect_crop
        filled = remove_render_shift(
            filled, original.astype(np.float32), touched
        )
        feather = max(
            2.0, OSOR_FEATHER * max(right - left, bottom - top)
        )
        import cv2

        soft = cv2.GaussianBlur(
            grow(removed, REMOVED_GROW_RATIO).astype(np.float32),
            (0, 0),
            feather,
        )[..., None]
        stitched = filled * soft + original * (1 - soft)
        stitched = np.clip(np.rint(stitched), 0, 255).astype(np.uint8)
        stitched = match_grain(stitched, original, removed.astype(np.uint8))
        result[window] = stitched
        result[protect] = canvas[protect]
        return result

    def _get_model(self) -> Any:
        with self._lock:
            if self._model is None:
                self._model = self._create_model()
            return self._model

    def _create_model(self) -> _OsorModel:
        return _OsorModel(self.settings.osor_model_path)
