from __future__ import annotations

import asyncio
from collections.abc import Callable

import numpy as np
from PIL import Image

from app.config import Settings
from app.services.erase_auto import SdxlEdit, auto_erase
from app.services.engines.lama_eraser import LamaEraser
from app.services.engines.osor_eraser import OsorEraser
from app.services.fill_people import PeopleFinder
from app.services.inpaint_mask import adjust_mask

ERASER_AUTO_MODEL_ID = "eraser-auto"
ERASER_AUTO_MODEL_LABEL = "Borrado automático (LaMa + OSOR)"
SdxlEditFactory = Callable[[], SdxlEdit]


class AutoEraser:
    def __init__(
        self,
        settings: Settings,
        lama_eraser: LamaEraser,
        osor_eraser: OsorEraser,
        people_finder_factory: Callable[[str], PeopleFinder | None],
        sdxl_edit_factory: Callable[[], SdxlEditFactory | None],
    ) -> None:
        self.settings = settings
        self.lama_eraser = lama_eraser
        self.osor_eraser = osor_eraser
        self.people_finder_factory = people_finder_factory
        self.sdxl_edit_factory = sdxl_edit_factory

    def available(self) -> bool:
        return self.lama_eraser is not None and self.lama_eraser.available()

    async def erase(
        self,
        base_image: Image.Image,
        mask_image: Image.Image,
        device: str,
        *,
        protect_image: Image.Image | None = None,
        engine: str | None = None,
        grow_px: int = 0,
        feather_px: int = 0,
    ) -> tuple[Image.Image, list[dict]]:
        return await asyncio.to_thread(
            self._erase_blocking,
            base_image,
            mask_image,
            device,
            protect_image,
            engine,
            grow_px,
            feather_px,
        )

    def _erase_blocking(
        self,
        base_image: Image.Image,
        mask_image: Image.Image,
        device: str,
        protect_image: Image.Image | None,
        engine: str | None,
        grow_px: int,
        feather_px: int,
    ) -> tuple[Image.Image, list[dict]]:
        base_rgb = base_image.convert("RGB")
        mask = np.asarray(
            mask_image.convert("L").resize(base_rgb.size, Image.Resampling.NEAREST)
        )
        mask = adjust_mask(mask, grow_px, feather_px)
        protect = None
        if protect_image is not None:
            protect = np.asarray(
                protect_image.convert("L").resize(
                    base_rgb.size, Image.Resampling.NEAREST
                )
            ) > 127

        sdxl_edit: SdxlEditFactory | None = None
        factory_resolved = False

        def resolve_sdxl_edit() -> SdxlEdit | None:
            nonlocal sdxl_edit, factory_resolved
            if not factory_resolved:
                sdxl_edit = self.sdxl_edit_factory()
                factory_resolved = True
            return sdxl_edit

        lama_fill = self.lama_eraser.fill_group
        osor_fill = (
            self.osor_eraser.fill_group
            if self.osor_eraser is not None and self.osor_eraser.available()
            else None
        )
        return auto_erase(
            base_rgb,
            mask,
            protect,
            lama_fill,
            None,
            osor_fill=osor_fill,
            find_people=self.people_finder_factory(device),
            engine=engine,
            sdxl_edit_factory=resolve_sdxl_edit,
        )
