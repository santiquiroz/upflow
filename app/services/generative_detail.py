"""Pase generativo por tiles: INVENTA detalle donde el reescalado clasico no lo tiene.

La diferencia con el reescalado normal hay que decirla, no suavizarla: un
ESRGAN reconstruye lo que estaba y no puede ir mas alla de la informacion del
original. Este pase le pide a un modelo de difusion que dibuje textura nueva
sobre lo reescalado. Queda mas nitido y mas convincente, y **no es una foto mas
fiel**: los pelos, los poros y las letras chicas que aparecen no estaban.

Por eso corre DESPUES del reescalado clasico y con `strength` bajo. El clasico
pone la estructura —donde va cada cosa— y este solo la textura. Al reves, o con
strength alto, el modelo se inventa la escena.

Los tiles existen por VRAM: una imagen de 4000x3000 no entra en un paso de
difusion. Se reusan `tile_starts`/`tile_weights`/`blend_tiles` del reescalador
clasico, que ya resuelven la costura con un feather lineal sobre el solape.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np

from app.services.engines.onnx_common import blend_tiles, tile_starts

logger = logging.getLogger(__name__)

# 512 es el lado nativo de SD 1.5 y un submultiplo comodo de SDXL. Un tile mas
# chico multiplica las pasadas; uno mas grande se sale de lo que el modelo vio
# entrenando y empieza a repetir composicion.
DEFAULT_TILE = 512
# Suficiente para que el feather tenga donde desvanecer sin duplicar trabajo:
# cada pixel de solape se difunde dos veces.
DEFAULT_OVERLAP = 64
# Arriba de esto el modelo deja de retocar y reinterpreta: aparecen ojos donde
# habia un boton. El tope es del dominio, no de la interfaz.
MAX_STRENGTH = 0.45


@dataclass(frozen=True, slots=True)
class TilePlan:
    y0: int
    x0: int
    height: int
    width: int
    is_top: bool
    is_bottom: bool
    is_left: bool
    is_right: bool


def plan_tiles(height: int, width: int, tile: int, overlap: int) -> list[TilePlan]:
    """Donde cae cada tile, con sus banderas de borde.

    Las banderas importan para el feather: un tile pegado al borde de la imagen
    NO se desvanece de ese lado —no hay vecino con quien mezclarse— y hacerlo
    dejaria el marco de la imagen medio transparente.
    """
    filas = tile_starts(height, tile, overlap)
    columnas = tile_starts(width, tile, overlap)
    planes: list[TilePlan] = []
    for y0 in filas:
        alto = min(tile, height)
        for x0 in columnas:
            ancho = min(tile, width)
            planes.append(
                TilePlan(
                    y0=y0,
                    x0=x0,
                    height=alto,
                    width=ancho,
                    is_top=y0 == 0,
                    is_bottom=y0 + alto >= height,
                    is_left=x0 == 0,
                    is_right=x0 + ancho >= width,
                )
            )
    return planes


def clamp_strength(strength: float) -> float:
    """Lo que se acepta como "agregar detalle" y no "dibujar otra cosa"."""
    if strength <= 0:
        raise ValueError("La fuerza del pase generativo tiene que ser mayor que cero.")
    return min(strength, MAX_STRENGTH)


# Sincrono a proposito: el que lo llama ya corre dentro de un hilo aparte —el
# modelo bloquea— y una capa async ahi no gana nada y complica el que lo usa.
TileRunner = Callable[[np.ndarray, TilePlan], np.ndarray]


def add_generative_detail(
    image: np.ndarray,
    *,
    run_tile: TileRunner,
    tile: int = DEFAULT_TILE,
    overlap: int = DEFAULT_OVERLAP,
    on_progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Recorre la imagen por tiles, los pasa por `run_tile` y los vuelve a unir.

    `run_tile` se inyecta y no se importa: el que sabe de modelos, dispositivos y
    checkpoints es el manager, y asi este modulo —que es toda la geometria— se
    prueba sin cargar un solo peso.
    """
    height, width = image.shape[0], image.shape[1]
    planes = plan_tiles(height, width, tile, overlap)
    procesados: list[tuple[int, int, int, int, np.ndarray]] = []
    for indice, plan in enumerate(planes, start=1):
        recorte = image[plan.y0 : plan.y0 + plan.height, plan.x0 : plan.x0 + plan.width]
        generado = run_tile(recorte, plan)
        if generado.shape[:2] != recorte.shape[:2]:
            # El modelo redondea a multiplos de 8 o 64. Devolver un tile de otro
            # tamaño desalinea TODO lo que sigue, y el error saldria como una
            # imagen corrida en vez de como un fallo.
            raise RuntimeError(
                f"El pase generativo devolvio un tile de {generado.shape[:2]} "
                f"donde se esperaba {recorte.shape[:2]}."
            )
        procesados.append((plan.y0, plan.x0, plan.height, plan.width, generado))
        if on_progress is not None:
            on_progress(indice, len(planes))
    # Feather = solape entero: asi el peso pasa de 1 a 0 a lo largo de toda la
    # banda compartida y solo una linea queda al 50%.
    return blend_tiles(
        procesados, height, width, image.shape[2], scale=1, feather=overlap
    )
