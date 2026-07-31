from __future__ import annotations

import asyncio
import io
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings

# ---------------------------------------------------------------------------
# Editor - "tocar un objeto -> máscara" con MobileSAM (Acly/MobileSAM rev
# 0d3b4033, MIT). Dos etapas: encoder una vez por imagen (embeddings cacheados
# por (imagen, device)) + decoder por click (~ms). Contrato IO verificado
# contra los scripts de export del repo HF y dlimgedit (que corrió estos
# mismos ONNX sobre DirectML en producción):
#   - encoder: input_image [H,W,3] f32 RGB 0-255; la normalización, el pad a
#     1024 y el layout CHW viven DENTRO del grafo. Se paddea acá con el pixel
#     medio (que tras normalizar es ~0) a 1024x1024 fijo para que DirectML no
#     recompile kernels por cada shape distinta.
#   - decoder: point_coords en espacio lado-largo-1024, labels 1=click,
#     -1=padding; masks ya upsampleadas al tamaño original; threshold en 0.0.
# Sesiones vía ep_registry (nativo -> DirectML -> CPU, mismo camino que todo).
# ---------------------------------------------------------------------------

SAM_INPUT_SIDE = 1024
# Pixel medio de ImageNet que usa SAM: el pad con este valor queda ~0 tras la
# normalización embebida en el grafo (semántica pad-after-normalize de SAM).
SAM_MEAN_PIXEL = (123.675, 116.28, 103.53)
EMBEDDINGS_CACHE_SIZE = 2


class SegmenterUnavailableError(RuntimeError):
    pass


def prepare_encoder_input(rgb: np.ndarray) -> tuple[np.ndarray, float]:
    from PIL import Image

    height, width = rgb.shape[0], rgb.shape[1]
    scale = SAM_INPUT_SIDE / max(height, width, 1)
    resized_h, resized_w = round(height * scale), round(width * scale)
    resized = np.asarray(
        Image.fromarray(rgb).resize((resized_w, resized_h), Image.BILINEAR), dtype=np.float32
    )
    canvas = np.tile(
        np.array(SAM_MEAN_PIXEL, dtype=np.float32), (SAM_INPUT_SIDE, SAM_INPUT_SIDE, 1)
    )
    canvas[:resized_h, :resized_w] = resized
    return canvas, scale


def build_decoder_inputs(
    embeddings: np.ndarray, x: float, y: float, scale: float, orig_h: int, orig_w: int
) -> dict[str, np.ndarray]:
    return {
        "image_embeddings": embeddings,
        "point_coords": np.array([[[x * scale, y * scale], [0.0, 0.0]]], dtype=np.float32),
        "point_labels": np.array([[1.0, -1.0]], dtype=np.float32),
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.zeros((1,), dtype=np.float32),
        "orig_im_size": np.array([orig_h, orig_w], dtype=np.float32),
    }


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    from PIL import Image

    image = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _sam_session_options_factory() -> Any:
    import onnxruntime as ort

    options = ort.SessionOptions()
    # Restricciones documentadas del DML EP: sin memory pattern y ejecución
    # secuencial (mismo tuning que _tune_session_options_for_device de gmfss).
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


class EditorSegmenter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._sessions: dict[str, Any] = {}
        self._embeddings: OrderedDict[tuple[str, str], tuple[np.ndarray, float, int, int]] = (
            OrderedDict()
        )

    def available(self) -> bool:
        return self.settings.editor_segmenter_available()

    async def segment(self, image_path: Path, x: float, y: float, device: str) -> bytes:
        if not self.available():
            raise SegmenterUnavailableError(
                "La selección por toque no está instalada: corré scripts/download-mobilesam.ps1"
            )
        return await asyncio.to_thread(self._segment_blocking, image_path, x, y, device)

    def _segment_blocking(self, image_path: Path, x: float, y: float, device: str) -> bytes:
        embeddings, scale, orig_h, orig_w = self._embeddings_for(image_path, device)
        decoder = self._get_session("decoder", self.settings.editor_segmenter_decoder_path, device)
        clamped_x = min(max(x, 0.0), float(orig_w - 1))
        clamped_y = min(max(y, 0.0), float(orig_h - 1))
        inputs = build_decoder_inputs(embeddings, clamped_x, clamped_y, scale, orig_h, orig_w)
        masks = decoder.run(None, inputs)[0]
        return mask_to_png_bytes(masks[0, 0] > 0.0)

    def _embeddings_for(self, image_path: Path, device: str) -> tuple[np.ndarray, float, int, int]:
        cache_key = (f"{image_path}:{image_path.stat().st_mtime_ns}", device)
        with self._lock:
            cached = self._embeddings.get(cache_key)
            if cached is not None:
                self._embeddings.move_to_end(cache_key)
                return cached

        from PIL import Image

        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        orig_h, orig_w = rgb.shape[0], rgb.shape[1]
        canvas, scale = prepare_encoder_input(rgb)
        encoder = self._get_session("encoder", self.settings.editor_segmenter_encoder_path, device)
        embeddings = encoder.run(None, {"input_image": canvas})[0]

        entry = (embeddings, scale, orig_h, orig_w)
        with self._lock:
            self._embeddings[cache_key] = entry
            self._embeddings.move_to_end(cache_key)
            while len(self._embeddings) > EMBEDDINGS_CACHE_SIZE:
                self._embeddings.popitem(last=False)
        return entry

    def _get_session(self, name: str, model_path: Path, device: str) -> Any:
        key = f"{name}:{device}"
        with self._lock:
            cached = self._sessions.get(key)
            if cached is not None:
                return cached
        session = self._create_session(model_path, device)
        with self._lock:
            self._sessions[key] = session
        return session

    def _create_session(self, model_path: Path, device: str) -> Any:
        # Monkeypatchable seam, mismo patrón que los engines.
        from app.services import ep_registry

        return ep_registry.create_session(
            str(model_path), device, self.settings, sess_options_factory=_sam_session_options_factory
        )
