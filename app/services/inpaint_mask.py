from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Preparación de máscaras para edición con calidad de producto.
#
# Tres problemas reales que resuelve este módulo, en el orden en que aparecen
# al borrar algo (reportado por el usuario: "al eliminar la boca quedan los
# bordes o genera cosas raras"):
#
# 1. La máscara pintada queda PEGADA al objeto, así que el halo del contorno
#    (píxeles del objeto que el pincel no llegó a tapar) sobrevive dentro de la
#    zona conservada y el modelo lo usa como pista: reconstruye el objeto o
#    deja su silueta. Se arregla DILATANDO la máscara unos píxeles.
# 2. El corte binario entre lo generado y lo original deja una costura visible
#    aunque el relleno sea bueno. Se arregla con un borde difuminado (feather)
#    y componiendo por alpha en vez de con un if.
# 3. Editar la imagen entera a 1024 desperdicia casi toda la capacidad del
#    modelo en píxeles que no cambian y deja la zona editada con poco detalle.
#    Se arregla recortando la región marcada con contexto, editándola sola y
#    pegándola de vuelta (el flujo "solo el área marcada" de A1111/ComfyUI).
#
# Sin dependencias nuevas: numpy + PIL, ambas ya usadas por el proyecto.
# ---------------------------------------------------------------------------

MASK_THRESHOLD = 127


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Agranda el área marcada `pixels` hacia afuera."""
    if pixels <= 0:
        return mask
    from PIL import Image, ImageFilter

    image = Image.fromarray(mask, mode="L")
    # MaxFilter necesita tamaño impar; se aplica en pasos de 2 px de radio para
    # que un radio grande no explote el costo del kernel.
    remaining = pixels
    while remaining > 0:
        step = min(2, remaining)
        image = image.filter(ImageFilter.MaxFilter(step * 2 + 1))
        remaining -= step
    return np.asarray(image)


def feather_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Agrega una rampa HACIA AFUERA del área marcada.

    Un blur directo sobre la máscara también baja el interior, y en una región
    chica (una boca, un cartel) el centro deja de estar del todo marcado: el
    modelo edita a medias y vuelve a asomar lo que se quería borrar. Por eso la
    rampa se construye sobre la máscara agrandada y después se vuelve a fijar el
    área original en 255: el interior queda 100% editable y la transición vive
    en los píxeles de afuera, que es donde tiene que estar para que no se vea la
    unión.
    """
    if pixels <= 0:
        return mask
    from PIL import Image, ImageFilter

    grown = Image.fromarray(dilate_mask(mask, pixels), mode="L")
    ramp = np.asarray(grown.filter(ImageFilter.GaussianBlur(pixels)))
    return np.maximum(mask, ramp)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(left, top, right, bottom) del área marcada, o None si no hay nada."""
    marked = mask > MASK_THRESHOLD
    rows = np.any(marked, axis=1)
    cols = np.any(marked, axis=0)
    if not rows.any() or not cols.any():
        return None
    top, bottom = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    left, right = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
    return left, top, right, bottom


def _expand_to_square(
    left: int, top: int, right: int, bottom: int, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    side = min(max(right - left, bottom - top), image_width, image_height)
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    new_left = int(round(center_x - side / 2))
    new_top = int(round(center_y - side / 2))
    new_left = max(0, min(new_left, image_width - side))
    new_top = max(0, min(new_top, image_height - side))
    return new_left, new_top, new_left + side, new_top + side


def compute_crop_box(
    bbox: tuple[int, int, int, int], padding: int, image_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Región a editar: el área marcada más contexto alrededor, cuadrada.

    Cuadrada porque los modelos de difusión trabajan mejor en su relación de
    aspecto nativa; el contexto (padding) es lo que le permite al modelo
    continuar el fondo en vez de inventar un parche aislado.
    """
    image_width, image_height = image_size
    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image_width, right + padding)
    bottom = min(image_height, bottom + padding)
    if right - left < 1:
        right = min(image_width, left + 1)
    if bottom - top < 1:
        bottom = min(image_height, top + 1)
    return _expand_to_square(left, top, right, bottom, image_width, image_height)


def soft_composite(generated: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mezcla lo generado sobre el original usando la máscara como alpha.

    Un `where(mask > 127)` deja escalón; esto respeta el degradado del feather,
    y con máscara 0 devuelve el original bit a bit (la garantía de que lo no
    marcado no se toca sigue valiendo).
    """
    if mask.shape[:2] != original.shape[:2] or generated.shape != original.shape:
        raise ValueError("mask and images must share the same height and width")
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    blended = generated.astype(np.float32) * alpha + original.astype(np.float32) * (1.0 - alpha)
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)
