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


def _centered_span(low: int, high: int, length: int, limit: int) -> int:
    start = int(round((low + high) / 2 - length / 2))
    return max(0, min(start, limit - length))


def _expand_to_cover(
    left: int, top: int, right: int, bottom: int, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    """Crop cuadrado cuando cabe; si no, cada eje se expande por su cuenta.

    El clamp cruzado min(lado, ancho, alto) era un bug (reproducido 2026-08-07):
    una marca más ancha que la dimensión menor de la imagen (1500x300 en
    1920x1080) daba un cuadrado de 1080 que NO cubría lo marcado — solo se
    editaba la franja central y quedaba una costura dura a los costados, sin
    error ni aviso. Cubrir el bbox manda sobre el aspecto: cada eje se clampa
    SOLO contra su propia dimensión de la imagen, y como el bbox ya viene
    dentro de la imagen, el crop siempre lo contiene entero.
    """
    target = max(right - left, bottom - top)
    width = min(target, image_width)
    height = min(target, image_height)
    new_left = _centered_span(left, right, width, image_width)
    new_top = _centered_span(top, bottom, height, image_height)
    return new_left, new_top, new_left + width, new_top + height


# Contexto MINIMO, para marcas diminutas donde una fraccion no daria con que
# continuar ni siquiera una textura.
MIN_CONTEXT_PX = 48

# Cuanto contexto alrededor, como fraccion del lado mayor de lo marcado.
#
# Era una constante ABSOLUTA de 48 px, y ese era el bug: en una foto grande una
# boca ocupa unos 300 px, asi que el modelo recibia el agujero y un pedacito de
# menton. Sin ver la cara no tiene con que saber hacia donde mira, y devolvia una
# boca plausible en si misma pero girada respecto del cuerpo (reportado
# 2026-08-06).
#
# Con 0,75 el recorte mide dos veces y media lo marcado, asi que alrededor de una
# boca entran los ojos y el contorno de la cabeza. Es un compromiso: mas contexto
# significa que lo marcado ocupa menos del recorte y recibe menos detalle al
# llevarlo a la resolucion del modelo. Se elige de este lado a proposito — una
# boca bien orientada con algo menos de detalle sirve; una nitida y al reves, no.
CONTEXT_RATIO = 0.75


def context_padding_for(bbox: tuple[int, int, int, int]) -> int:
    """Cuanto contexto darle al modelo alrededor de lo marcado.

    Escala con el LADO MAYOR: una mancha larga y finita necesita el contexto de
    su largo, porque mirar solo su alto daria una franja sin nada a los costados.
    """
    left, top, right, bottom = bbox
    lado_mayor = max(right - left, bottom - top)
    return max(MIN_CONTEXT_PX, round(CONTEXT_RATIO * lado_mayor))


def compute_crop_box(
    bbox: tuple[int, int, int, int],
    padding: int,
    image_size: tuple[int, int],
    max_side: int | None = None,
) -> tuple[int, int, int, int]:
    """Región a editar: el área marcada más contexto alrededor.

    Cuadrada cuando cabe, porque los modelos de difusión trabajan mejor cerca de
    su relación de aspecto nativa; si la marca es más larga que la dimensión
    menor de la imagen, rectangular — cubrir lo marcado entero manda sobre el
    aspecto (ver _expand_to_cover). El contexto (padding) es lo que le permite
    al modelo continuar el fondo en vez de inventar un parche aislado.

    `max_side` es la resolución a la que se va a editar. El contexto se toma del
    margen que sobra hasta ahí, NUNCA del detalle: un recorte más grande que esa
    resolución se achica para editarlo y vuelve a estirarse al pegarlo, y la zona
    queda más blanda que lo que la rodea — que es justo lo que delata un retoque.
    Medido el 2026-08-06: una marca de 900 px daba un recorte de 1675 px, o sea
    1,6x más blando que el resto de la foto.

    Si la marca SOLA ya pasa ese tope no hay nada que hacer: se prefiere el
    achique antes que recortar la marca, porque recortarla sería editar otra cosa.
    """
    image_width, image_height = image_size
    left, top, right, bottom = bbox
    if max_side is not None:
        # Lo que sobra hasta el tope, repartido a los dos lados. Nunca negativo:
        # con la marca ya pasada de tope, el contexto es cero y listo.
        holgura = max(0, max_side - max(right - left, bottom - top))
        padding = min(padding, holgura // 2)
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image_width, right + padding)
    bottom = min(image_height, bottom + padding)
    if right - left < 1:
        right = min(image_width, left + 1)
    if bottom - top < 1:
        bottom = min(image_height, top + 1)
    return _expand_to_cover(left, top, right, bottom, image_width, image_height)


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
