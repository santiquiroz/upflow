from __future__ import annotations

import asyncio
import threading
from typing import Any

import numpy as np

from app.config import Settings
from app.services.inpaint_mask import dilate_mask, feather_mask, mask_bbox, soft_composite
from app.services.missing_pack import missing_pack_message

# ---------------------------------------------------------------------------
# Borrado rápido con MI-GAN (Picsart, migan_pipeline_v2.onnx, 28 MB).
#
# Es un motor MÁS, no un reemplazo: rellena el hueco continuando el entorno, sin
# prompts ni pasos de difusión (medido en DirectML: 23-35 ms de 512px a 1080p,
# 13-20x más rápido que en CPU). Sirve para SACAR algo y que quede el fondo; para
# poner algo concreto en su lugar sigue estando el camino de difusión.
#
# Contrato del grafo, verificado corriéndolo (no leído de docs):
#   image  uint8 [B,3,H,W] RGB NCHW, 0..255 crudo (sin normalizar)
#   mask   uint8 [B,1,H,W] con la polaridad AL REVÉS de la nuestra:
#          255 = región conocida que se conserva, 0 = hueco a rellenar.
#   result uint8 [B,3,H,W], listo para usar.
# Tres trampas que este módulo resuelve:
#   1. La máscara se trata como alpha CONTINUO: un borde difuminado da un relleno
#      a medias, así que acá entra dura (0/255) aunque afuera se use con feather.
#   2. Resolución libre: el recorte, el escalado y el pegado están DENTRO del
#      grafo, así que no se hace crop-and-stitch acá (y el costo depende del
#      tamaño de lo marcado, no de la foto).
#   3. Sangra fuera del hueco (medido hasta 26/255 alrededor), así que el
#      resultado se compone contra el original para que lo no marcado no cambie.
# ---------------------------------------------------------------------------

ERASER_MODEL_ID = "eraser-migan"
ERASER_MODEL_LABEL = "Borrado rápido (sin IA generativa)"
MASK_THRESHOLD = 127
DEFAULT_DILATE_PX = 8
DEFAULT_FEATHER_PX = 6


class MiganUnavailableError(RuntimeError):
    pass


def build_migan_inputs(base_image: Any, mask_image: Any) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(base_image.convert("RGB"))
    mask = np.asarray(mask_image.convert("L"))
    image_nchw = np.transpose(rgb, (2, 0, 1))[None, ...].astype(np.uint8)
    known_region = np.where(mask > MASK_THRESHOLD, 0, 255).astype(np.uint8)
    return image_nchw, known_region[None, None, ...]


def _to_image(result: np.ndarray) -> Any:
    from PIL import Image

    return Image.fromarray(np.transpose(result[0], (1, 2, 0)).astype(np.uint8), mode="RGB")


class MiganEraser:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._sessions: dict[str, Any] = {}

    def available(self) -> bool:
        return self.settings.migan_available()

    async def erase(
        self,
        base_image: Any,
        mask_image: Any,
        device: str,
        dilate_px: int = DEFAULT_DILATE_PX,
        feather_px: int = DEFAULT_FEATHER_PX,
    ) -> Any:
        if not self.available():
            raise MiganUnavailableError(missing_pack_message("migan"))
        return await asyncio.to_thread(
            self._erase_blocking, base_image, mask_image, device, dilate_px, feather_px
        )

    def _erase_blocking(
        self, base_image: Any, mask_image: Any, device: str, dilate_px: int, feather_px: int
    ) -> Any:
        from PIL import Image

        base_rgb = base_image.convert("RGB")
        mask_gray = mask_image.convert("L")
        if mask_gray.size != base_rgb.size:
            mask_gray = mask_gray.resize(base_rgb.size, Image.NEAREST)

        mask_array = np.asarray(mask_gray)
        if mask_bbox(mask_array) is None:
            # Con la máscara vacía el modelo NO devuelve la imagen intacta: se corta acá.
            return base_rgb

        # Se agranda para tragarse el halo del contorno, igual que en el camino de difusión.
        grown = dilate_mask(mask_array, dilate_px)
        image_input, mask_input = build_migan_inputs(base_rgb, Image.fromarray(grown, mode="L"))

        session = self._get_session(device)
        generated = _to_image(session.run(None, {"image": image_input, "mask": mask_input})[0])

        blend_mask = feather_mask(grown, feather_px)
        stitched = soft_composite(np.asarray(generated), np.asarray(base_rgb), blend_mask)
        return Image.fromarray(stitched)

    def _get_session(self, device: str) -> Any:
        with self._lock:
            cached = self._sessions.get(device)
            if cached is not None:
                return cached
        session = self._create_session(device)
        with self._lock:
            self._sessions[device] = session
        return session

    def _create_session(self, device: str) -> Any:
        # Monkeypatchable seam, igual que el resto de los motores.
        from app.services import ep_registry

        return ep_registry.create_session(
            str(self.settings.migan_model_path), device, self.settings
        )
