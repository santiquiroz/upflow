"""Genera una malla 3D desde texto o desde una imagen, con Shap-E.

VERIFICADO EN ESTA MAQUINA el 2026-08-05, sin CUDA:
  - `openai/shap-e` es MIT — codigo y pesos, de OpenAI.
  - `diffusers` ya viaja con Upflow para generar imagenes, y ya trae
    `ShapEPipeline` y `ShapEImg2ImgPipeline`. CERO dependencias nuevas.
  - Corre en CPU: 116-137 s por malla en una maquina sin GPU NVIDIA.
  - De cuatro pruebas, TRES salieron estancas, manifold y solidas directamente,
    y las cuatro entraron en una Ender-3 al escalarlas a 80 mm.

LO QUE NO DA, y hay que decirlo cada vez: COTAS. Ninguna malla generada sabe que
tiene que medir 80 mm. Sirve para prototipos, carcasas, piezas decorativas y como
punto de partida. Para una pieza que TIENE que encajar, el camino sigue siendo el
carril parametrico.

Por eso todo lo que sale de aca pasa obligatoriamente por el banco antes de
llegar al usuario: una malla generada que no cierra no es una pieza.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from app.services.missing_pack import missing_pack_message

# 16 pasos alcanzaron en la medicion. Mas pasos cuestan minutos de CPU y la malla
# no mejora lo suficiente como para notarlo despues de escalar e imprimir.
DEFAULT_STEPS = 16
DEFAULT_GUIDANCE = 15.0
# El tamano de la grilla del decodificador. 64 da mallas de 60k a 260k triangulos,
# que es mas que suficiente para imprimir.
DEFAULT_FRAME_SIZE = 64


class Shape3dUnavailable(RuntimeError):
    pass


def _as_triangles(mesh: Any) -> np.ndarray:
    """Pasa la salida del decodificador a triangulos (n, 3, 3)."""
    verts = np.asarray(mesh.verts, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if verts.size == 0 or faces.size == 0:
        raise Shape3dUnavailable("El modelo devolvio una malla vacia.")
    return verts[faces]


def assert_usable_mesh(triangles: np.ndarray) -> None:
    """Mismo guard que en TTS y conversion de voz, por el mismo motivo.

    El fallo caro de estos modelos no es una excepcion sino una salida con la
    forma correcta y el contenido roto.
    """
    if triangles.size == 0:
        raise Shape3dUnavailable("El modelo devolvio una malla vacia.")
    if not np.all(np.isfinite(triangles)):
        raise Shape3dUnavailable(
            "La malla generada tiene coordenadas invalidas (NaN o infinito)."
        )


class Shape3dEngine:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self._text_pipe: Any | None = None
        self._image_pipe: Any | None = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        return (self.model_dir / "model_index.json").exists()

    def generate_from_text(
        self,
        prompt: str,
        *,
        steps: int = DEFAULT_STEPS,
        guidance: float = DEFAULT_GUIDANCE,
        frame_size: int = DEFAULT_FRAME_SIZE,
    ) -> np.ndarray:
        if not prompt.strip():
            raise Shape3dUnavailable("Hace falta una descripcion de la pieza.")
        pipe = self._load_text()
        salida = pipe(
            prompt,
            guidance_scale=guidance,
            num_inference_steps=steps,
            frame_size=frame_size,
            output_type="mesh",
        )
        triangulos = _as_triangles(salida.images[0])
        assert_usable_mesh(triangulos)
        return triangulos

    def generate_from_image(
        self,
        image: Any,
        *,
        steps: int = DEFAULT_STEPS,
        guidance: float = 3.0,
        frame_size: int = DEFAULT_FRAME_SIZE,
    ) -> np.ndarray:
        pipe = self._load_image()
        salida = pipe(
            image,
            guidance_scale=guidance,
            num_inference_steps=steps,
            frame_size=frame_size,
            output_type="mesh",
        )
        triangulos = _as_triangles(salida.images[0])
        assert_usable_mesh(triangulos)
        return triangulos

    def _load_text(self) -> Any:
        with self._lock:
            if self._text_pipe is None:
                self._text_pipe = self._build("ShapEPipeline")
            return self._text_pipe

    def _load_image(self) -> Any:
        with self._lock:
            if self._image_pipe is None:
                self._image_pipe = self._build("ShapEImg2ImgPipeline")
            return self._image_pipe

    def _build(self, clase: str) -> Any:
        if not self.available():
            raise Shape3dUnavailable(
                missing_pack_message(
                    "shap-e", detail=f"Se esperaba en {self.model_dir}."
                )
            )
        import diffusers
        import torch

        constructor = getattr(diffusers, clase)
        # CPU explicito: en esta maquina no hay CUDA, y dejar que diffusers
        # elija dispositivo esconderia el dia que si la haya y el resultado
        # cambie sin que nadie lo pida.
        # `variant="fp16"` y `torch_dtype=float32`: se bajan los pesos fp16
        # —la mitad del tamano— y se castean a fp32 para inferir. Medido: la
        # malla sale igual. El cast es a proposito y no una omision; correr la
        # inferencia en fp16 sobre CPU no acelera nada y arriesga NaN, que es
        # exactamente lo que paso con la variante fp16 de Kokoro en este repo.
        return constructor.from_pretrained(
            str(self.model_dir), variant="fp16", torch_dtype=torch.float32
        ).to("cpu")
