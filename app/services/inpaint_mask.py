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
MAX_COMPONENTS = 64

# Dilatación y feather proporcionales al tamaño de lo marcado: 8 px fijos no
# alcanzan para tapar el halo de una persona de 2700 px (medido 2026-10).
DILATE_RATIO = 0.015
FEATHER_RATIO = 0.005
MIN_DILATE_PX = 8
MIN_FEATHER_PX = 8
MATTE_RATIO = 0.006
MATTE_SHARPEN = 3.0
MATTE_OUTWARD_FRACTION = 4
MATTE_SMOOTH_FRACTION = 0.33


def matte_radius(shape: tuple[int, ...]) -> int:
    return max(12, round(MATTE_RATIO * max(shape[:2])))


def matte_core(protect: np.ndarray, radius: int) -> np.ndarray:
    import cv2

    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    return cv2.erode(protect.astype(np.uint8), kernel) > 0


def _matte_outer(protect: np.ndarray, radius: int) -> np.ndarray:
    import cv2

    outward = max(2, radius // MATTE_OUTWARD_FRACTION)
    kernel = np.ones((2 * outward + 1, 2 * outward + 1), dtype=np.uint8)
    return cv2.dilate(protect.astype(np.uint8), kernel) == 0


def _matte_window(band: np.ndarray, radius: int) -> tuple[slice, slice]:
    rows = np.flatnonzero(band.any(axis=1))
    columns = np.flatnonzero(band.any(axis=0))
    margin = 3 * radius
    return (
        slice(max(0, rows[0] - margin), min(band.shape[0], rows[-1] + margin + 1)),
        slice(max(0, columns[0] - margin), min(band.shape[1], columns[-1] + margin + 1)),
    )


def _matte_local_mean(
    image: np.ndarray, samples: np.ndarray, radius: int, factor: int
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    weights = samples.astype(np.float32)
    if factor > 1:
        weights = cv2.resize(weights, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
    sigma = radius / factor
    density = cv2.GaussianBlur(weights, (0, 0), sigmaX=sigma)
    weighted = cv2.GaussianBlur(image * weights[:, :, None], (0, 0), sigmaX=sigma)
    mean = weighted / np.maximum(density[:, :, None], 1e-6)
    size = (samples.shape[1], samples.shape[0])
    if factor > 1:
        mean = cv2.resize(mean, size, interpolation=cv2.INTER_LINEAR)
        density = cv2.resize(density, size, interpolation=cv2.INTER_LINEAR)
    return mean, density


def _matte_band_alpha(
    original: np.ndarray, core: np.ndarray, outer: np.ndarray, band: np.ndarray, radius: int
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    factor = max(1, radius // 6)
    image = original.astype(np.float32)
    if factor > 1:
        size = (max(1, original.shape[1] // factor), max(1, original.shape[0] // factor))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    subject, subject_weight = _matte_local_mean(image, core, radius, factor)
    outside, outside_weight = _matte_local_mean(image, outer, radius, factor)
    valid = band & (subject_weight > 1e-3) & (outside_weight > 1e-3)
    pixels = original[valid].astype(np.float32)
    subject_distance = np.linalg.norm(pixels - subject[valid], axis=1)
    outside_distance = np.linalg.norm(pixels - outside[valid], axis=1)
    ratio = outside_distance / np.maximum(subject_distance + outside_distance, 1e-6)
    smooth = _smooth_matte_score(ratio, valid, radius)
    return valid, np.clip((smooth - 0.5) * MATTE_SHARPEN + 0.5, 0, 1)


def _smooth_matte_score(
    scores: np.ndarray, valid: np.ndarray, radius: int
) -> np.ndarray:
    import cv2

    weights = valid.astype(np.float32)
    weighted = np.zeros(valid.shape, dtype=np.float32)
    weighted[valid] = scores
    sigma = radius * MATTE_SMOOTH_FRACTION
    numerator = cv2.GaussianBlur(weighted, (0, 0), sigmaX=sigma)
    denominator = cv2.GaussianBlur(weights, (0, 0), sigmaX=sigma)
    return numerator[valid] / np.maximum(denominator[valid], 1e-6)


def subject_matte(
    original_rgb: np.ndarray, protect: np.ndarray, radius: int | None = None
) -> np.ndarray:
    import cv2

    alpha = protect.astype(np.float32)
    if not protect.any():
        return alpha
    radius = matte_radius(protect.shape) if radius is None else radius
    core = matte_core(protect, radius)
    outer = _matte_outer(protect, radius)
    band = ~core & ~outer
    if not band.any():
        return np.clip(cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.0), 0, 1)
    window = _matte_window(band, radius)
    valid, values = _matte_band_alpha(
        original_rgb[window], core[window], outer[window], band[window], radius
    )
    alpha[window][valid] = values
    matte = np.clip(cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.0), 0, 1)
    matte[core] = 1.0
    return matte


def clean_protect_mask(
    protect: np.ndarray, island_fraction: float = 0.02, grow_ratio: float = 0.003
) -> np.ndarray:
    if not protect.any():
        return protect
    import cv2

    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        protect.astype(np.uint8), connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(len(stats), dtype=bool)
    keep[1:] = areas >= island_fraction * areas.max()
    cleaned = keep[labels].astype(np.uint8)
    radius = max(6, round(grow_ratio * max(protect.shape)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    return cv2.dilate(cleaned, kernel) > 0


def _lado_mayor(bbox: tuple[int, int, int, int]) -> int:
    left, top, right, bottom = bbox
    return max(right - left, bottom - top)


def dilation_for(bbox: tuple[int, int, int, int]) -> int:
    """Cuánto agrandar lo marcado según su tamaño (nunca menos que el mínimo)."""
    return max(MIN_DILATE_PX, round(DILATE_RATIO * _lado_mayor(bbox)))


def feather_for(bbox: tuple[int, int, int, int]) -> int:
    """Ancho de la transición según el tamaño de lo marcado."""
    return max(MIN_FEATHER_PX, round(FEATHER_RATIO * _lado_mayor(bbox)))


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


def adjust_mask(mask: np.ndarray, grow_px: int, feather_px: int) -> np.ndarray:
    adjusted = np.asarray(mask, dtype=np.uint8)
    if grow_px > 0:
        adjusted = dilate_mask(adjusted, grow_px)
    elif grow_px < 0:
        import cv2

        radius = abs(grow_px)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        adjusted = cv2.erode(adjusted, kernel)
    if feather_px > 0:
        adjusted = feather_mask(adjusted, feather_px)
    return np.asarray(adjusted, dtype=np.uint8)


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


def _grow_component_window(
    mask: np.ndarray,
    labels: np.ndarray,
    label: int,
    bbox: tuple[int, int, int, int],
    dilation: int,
    feather: int,
) -> tuple[tuple[slice, slice], np.ndarray]:
    left, top, right, bottom = bbox
    margin = dilation + 3 * feather + 2
    height, width = mask.shape
    window = (
        slice(max(0, top - margin), min(height, bottom + margin)),
        slice(max(0, left - margin), min(width, right + margin)),
    )
    component = np.where(labels[window] == label, mask[window], 0)
    return window, feather_mask(dilate_mask(component, dilation), feather)


def grow_per_component(
    mask: np.ndarray, dilate_px: int | None, feather_px: int | None
) -> np.ndarray:
    if dilate_px == 0 and feather_px == 0:
        return mask
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > MASK_THRESHOLD).astype(np.uint8), connectivity=8
    )
    if count - 1 > MAX_COMPONENTS:
        bbox = mask_bbox(mask)
        dilation = dilate_px if dilate_px is not None else dilation_for(bbox)
        feather = feather_px if feather_px is not None else feather_for(bbox)
        return feather_mask(dilate_mask(mask, dilation), feather)
    grown = mask.copy()
    for label in range(1, count):
        left, top, width, height, _ = (int(value) for value in stats[label])
        bbox = (left, top, left + width, top + height)
        dilation = dilate_px if dilate_px is not None else dilation_for(bbox)
        feather = feather_px if feather_px is not None else feather_for(bbox)
        window, component = _grow_component_window(
            mask, labels, label, bbox, dilation, feather
        )
        np.maximum(grown[window], component, out=grown[window])
    return grown


def _grow_hull_window(
    labels: np.ndarray,
    label: int,
    bbox: tuple[int, int, int, int],
    dilation: int,
) -> tuple[tuple[slice, slice], np.ndarray]:
    import cv2

    left, top, right, bottom = bbox
    height, width = labels.shape
    window = (
        slice(max(0, top - dilation), min(height, bottom + dilation)),
        slice(max(0, left - dilation), min(width, right + dilation)),
    )
    rows, columns = np.nonzero(labels[window] == label)
    points = np.column_stack((columns, rows)).astype(np.int32)
    hull = cv2.convexHull(points)
    component = np.zeros(labels[window].shape, dtype=np.uint8)
    cv2.fillConvexPoly(component, hull, 255)
    return window, dilate_mask(component, dilation)


def grow_small_hulls(
    mask: np.ndarray, small_side: int = 700, ratio: float = 0.08
) -> np.ndarray:
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > MASK_THRESHOLD).astype(np.uint8), connectivity=8
    )
    if count - 1 > MAX_COMPONENTS:
        return mask
    grown = mask.copy()
    for label in range(1, count):
        left, top, width, height, _ = (int(value) for value in stats[label])
        longer_side = max(width, height)
        if longer_side > small_side:
            continue
        bbox = (left, top, left + width, top + height)
        dilation = max(MIN_DILATE_PX, round(ratio * longer_side))
        window, component = _grow_hull_window(labels, label, bbox, dilation)
        np.maximum(grown[window], component, out=grown[window])
    return grown


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
