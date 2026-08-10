from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from PIL import Image, ImageFilter

from app.services.inpaint_mask import mask_bbox

# ---------------------------------------------------------------------------
# Fase 0 de "transferir un objeto entre imágenes": compositing puro CPU.
#
# El caller ya trae el objeto segmentado (RGB + máscara L de MobileSAM) y el
# destino donde pegarlo. Acá se recorta, se escala, se iguala el color contra
# la zona donde va a caer y se pega con borde suave. La segunda imagen que se
# devuelve es la máscara de armonización que la fase 1 le pasa tal cual al
# inpaint: NO es binaria, trae un degradado (máximo en la costura, casi cero en
# el centro de lo pegado) para que la difusión diferencial re-genere la unión
# sin reinventar el objeto. Ver harmonization_mask.
# ---------------------------------------------------------------------------

# 70% mapeado / 30% original: el mapeo MK lleva la distribución de color del
# objeto EXACTAMENTE a la del entorno destino, y eso sobre-corrige — un objeto
# legítimamente distinto de su fondo debe seguir siéndolo. La mezcla deja la
# corrección dominante sin borrar la identidad de color del objeto.
MK_MAPPED_WEIGHT = 0.7

# Una covarianza degenerada (objeto o región de color plano) tiene eigenvalues
# en 0 y la raíz inversa explota; el piso la vuelve invertible.
EIGEN_FLOOR = 1e-8

# El anillo de referencia de color: rectángulo de pegado expandido 25%.
RING_EXPAND_RATIO = 0.25

# La máscara de armonización se agranda para que el inpaint pueda fundir
# también el borde exterior de la costura, no solo lo pegado.
HARMONIZE_DILATE_RATIO = 0.10
HARMONIZE_DILATE_MIN_PX = 8

# Qué fracción de la PROFUNDIDAD del objeto entra en la banda de costura. El
# centro de lo pegado no tiene nada roto que arreglar: re-generarlo solo lo
# aleja del objeto que el usuario eligió. Con 0.35 la banda queda holgada (en un
# objeto de 400 px de ancho son ~70 px hacia adentro) y el interior se conserva.
DEFAULT_HARMONIZE_BLEND = 0.35

# Piso de la banda, en píxeles: por debajo de esto el modelo no tiene con qué
# fundir nada, por chico que sea el objeto o por bajo que venga el parámetro.
HARMONIZE_SEAM_MIN_PX = 12


@dataclass(slots=True, kw_only=True)
class PasteSpec:
    x: int
    y: int
    width: int
    height: int
    feather_px: int = 6
    match_color: bool = True
    # Ver harmonization_mask: 0 = solo la costura, 1 = el objeto entero a
    # intensidad uniforme (el comportamiento clásico).
    harmonize_blend: float = DEFAULT_HARMONIZE_BLEND


def transfer_object(
    source: Image.Image,
    source_mask: Image.Image,
    target: Image.Image,
    spec: PasteSpec,
) -> tuple[Image.Image, Image.Image]:
    _validate_inputs(source, source_mask, spec)
    source = source.convert("RGB")
    source_mask = source_mask.convert("L")
    target = target.convert("RGB")

    object_image, object_mask = crop_object(source, source_mask)
    # Fit con aspecto PRESERVADO dentro del rectángulo pedido, centrado: el
    # rect del caller suele venir de la foto origen entera y el bbox del
    # objeto casi nunca comparte su aspecto — estirar deformaba todo insert
    # (confirmado en review adversarial).
    fit_w, fit_h = _fit_preserving_aspect(object_image.size, spec.width, spec.height)
    object_image = object_image.resize((fit_w, fit_h), Image.Resampling.LANCZOS)
    object_mask = object_mask.resize((fit_w, fit_h), Image.Resampling.BILINEAR)
    paste_x = spec.x + (spec.width - fit_w) // 2
    paste_y = spec.y + (spec.height - fit_h) // 2

    alpha = feather_alpha(object_mask, spec.feather_px)
    object_rgb = np.asarray(object_image, dtype=np.float32)
    if spec.match_color:
        reference = _target_ring_pixels(np.asarray(target, dtype=np.float32), spec)
        object_rgb = match_color_mk(object_rgb, alpha, reference)

    composed = paste(target, object_rgb, alpha, paste_x, paste_y)
    return composed, harmonization_mask(target.size, alpha, replace(spec, x=paste_x, y=paste_y, width=fit_w, height=fit_h))


def _fit_preserving_aspect(size: tuple[int, int], max_w: int, max_h: int) -> tuple[int, int]:
    width, height = size
    scale = min(max_w / max(width, 1), max_h / max(height, 1))
    return max(1, round(width * scale)), max(1, round(height * scale))


def _validate_inputs(source: Image.Image, source_mask: Image.Image, spec: PasteSpec) -> None:
    if source.size != source_mask.size:
        raise ValueError("source_mask must be the same size as source")
    if spec.width < 1 or spec.height < 1:
        raise ValueError("spec width and height must be >= 1")
    if spec.feather_px < 0:
        raise ValueError("feather_px must be >= 0")
    if not 0.0 <= spec.harmonize_blend <= 1.0:
        raise ValueError("harmonize_blend must be between 0 and 1")


def crop_object(source: Image.Image, source_mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    """Recorta imagen y máscara al bbox del área marcada."""
    bbox = mask_bbox(np.asarray(source_mask))
    if bbox is None:
        raise ValueError("source_mask has no marked pixels")
    return source.crop(bbox), source_mask.crop(bbox)


def feather_alpha(mask: Image.Image, feather_px: int) -> np.ndarray:
    """Alpha 0..1 con rampa HACIA ADENTRO del borde del objeto.

    Hacia adentro porque hacia afuera no hay objeto: extender el alpha allí
    pegaría fondo de la imagen origen sobre el destino.
    """
    base = np.asarray(mask, dtype=np.float32) / 255.0
    if feather_px <= 0:
        return base
    from scipy.ndimage import distance_transform_edt

    # El borde del recorte cuenta como fondo: un objeto cortado por el borde de
    # la imagen origen también necesita costura suave ahí.
    inside = np.pad(np.asarray(mask) > 127, 1)
    ramp = np.clip(distance_transform_edt(inside)[1:-1, 1:-1] / feather_px, 0.0, 1.0)
    return ramp.astype(np.float32) * base


def match_color_mk(
    object_rgb: np.ndarray, object_alpha: np.ndarray, target_region_rgb: np.ndarray
) -> np.ndarray:
    """Igualación lineal Monge-Kantorovich (Pitié & Kokaram 2007).

    Reimplementación del paper; el paquete pip `color-matcher` es GPL-3.0 y no
    se puede importar ni copiar.

    Sigue siendo clásico a propósito: la armonización aprendida (PCT-Net y
    familia) mide ~9x mejor, pero TODOS los checkpoints publicados se entrenan
    sobre HAdobe5k, cuya licencia de Adobe es explícitamente de investigación.
    Ver docs/superpowers/specs/2026-08-10-pctnet-harmonization-license-findings.md
    """
    pixels = object_rgb.reshape(-1, 3).astype(np.float32)
    weights = object_alpha.reshape(-1).astype(np.float32)
    references = target_region_rgb.reshape(-1, 3).astype(np.float32)
    if weights.sum() <= 0.0 or len(references) == 0:
        return object_rgb.astype(np.float32)
    mean_o, cov_o = _weighted_mean_cov(pixels, weights)
    mean_t, cov_t = _weighted_mean_cov(references, np.ones(len(references), dtype=np.float32))
    mapped = (pixels - mean_o) @ _mk_transform(cov_o, cov_t).T + mean_t
    blended = MK_MAPPED_WEIGHT * mapped + (1.0 - MK_MAPPED_WEIGHT) * pixels
    return np.clip(blended, 0.0, 255.0).reshape(object_rgb.shape).astype(np.float32)


def _weighted_mean_cov(pixels: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = float(weights.sum())
    mean = (pixels * weights[:, None]).sum(axis=0) / total
    centered = pixels - mean
    cov = (centered * weights[:, None]).T @ centered / total
    return mean.astype(np.float32), cov.astype(np.float64)


def _sym_matrix_power(matrix: np.ndarray, exponent: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, EIGEN_FLOOR)
    return (vectors * values**exponent) @ vectors.T


def _mk_transform(cov_object: np.ndarray, cov_target: np.ndarray) -> np.ndarray:
    # T = Σt^(1/2) · (Σt^(1/2) · Σo · Σt^(1/2))^(-1/2) · Σt^(1/2)
    root_t = _sym_matrix_power(cov_target, 0.5)
    inner = _sym_matrix_power(root_t @ cov_object @ root_t, -0.5)
    return (root_t @ inner @ root_t).astype(np.float32)


def _target_ring_pixels(target_arr: np.ndarray, spec: PasteSpec) -> np.ndarray:
    """Píxeles de referencia: anillo alrededor del rectángulo de pegado.

    Se excluye lo que el objeto va a tapar: igualar contra píxeles que van a
    desaparecer sesgaría el color hacia justamente lo que se está cubriendo.
    """
    height, width = target_arr.shape[:2]
    expand_x = max(1, round(spec.width * RING_EXPAND_RATIO))
    expand_y = max(1, round(spec.height * RING_EXPAND_RATIO))
    left = max(0, spec.x - expand_x)
    top = max(0, spec.y - expand_y)
    right = min(width, spec.x + spec.width + expand_x)
    bottom = min(height, spec.y + spec.height + expand_y)
    if right <= left or bottom <= top:
        return np.empty((0, 3), dtype=np.float32)
    region = target_arr[top:bottom, left:right]
    keep = np.ones(region.shape[:2], dtype=bool)
    keep[
        max(0, spec.y - top) : max(0, spec.y + spec.height - top),
        max(0, spec.x - left) : max(0, spec.x + spec.width - left),
    ] = False
    ring = region[keep]
    # Anillo vacío (el pegado tapa toda la zona visible): rectángulo entero.
    return ring if len(ring) else region.reshape(-1, 3)


def paste(
    target: Image.Image, object_rgb: np.ndarray, alpha: np.ndarray, x: int, y: int
) -> Image.Image:
    """Composite alpha clásico; lo que cae fuera del destino se recorta."""
    overlap = _overlap(x, y, alpha.shape[1], alpha.shape[0], target.width, target.height)
    if overlap is None:
        return target.copy()
    (t_left, t_top, t_right, t_bottom), (o_left, o_top, o_right, o_bottom) = overlap
    canvas = np.asarray(target, dtype=np.float32)
    patch_alpha = alpha[o_top:o_bottom, o_left:o_right, None]
    patch = object_rgb[o_top:o_bottom, o_left:o_right]
    region = canvas[t_top:t_bottom, t_left:t_right]
    canvas[t_top:t_bottom, t_left:t_right] = patch * patch_alpha + region * (1.0 - patch_alpha)
    return Image.fromarray(np.clip(np.rint(canvas), 0, 255).astype(np.uint8), mode="RGB")


def _overlap(
    x: int, y: int, width: int, height: int, target_width: int, target_height: int
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    """(caja en destino, caja en objeto) del solape, o None si no hay."""
    left, top = max(0, x), max(0, y)
    right, bottom = min(target_width, x + width), min(target_height, y + height)
    if right <= left or bottom <= top:
        return None
    return (left, top, right, bottom), (left - x, top - y, right - x, bottom - y)


def harmonization_mask(
    target_size: tuple[int, int], alpha: np.ndarray, spec: PasteSpec
) -> Image.Image:
    """Máscara de armonización: máxima en la costura, decreciente hacia adentro.

    Un strength plano sobre toda la zona pegada re-genera también el centro del
    objeto, que no tiene ninguna costura que arreglar — y ahí es justo donde se
    pierde la identidad de lo que el usuario eligió pegar. Con la intensidad
    CONTINUA, el inpaint por difusión diferencial (generation_soft_inpaint
    re-inyecta el original en proporción al gris de la máscara, paso a paso)
    conserva el interior casi intacto y concentra el trabajo en la banda de la
    costura, que es el único lugar donde el pegado se nota.

    `spec.harmonize_blend` es la fracción de la profundidad del objeto que entra
    en esa banda; 1.0 devuelve la máscara uniforme clásica.
    """
    width, height = target_size
    canvas = np.zeros((height, width), dtype=np.float32)
    overlap = _overlap(spec.x, spec.y, spec.width, spec.height, width, height)
    if overlap is None:
        return Image.fromarray(canvas.astype(np.uint8), mode="L")
    (t_left, t_top, t_right, t_bottom), (o_left, o_top, o_right, o_bottom) = overlap
    canvas[t_top:t_bottom, t_left:t_right] = alpha[o_top:o_bottom, o_left:o_right]
    dilate_px = max(
        HARMONIZE_DILATE_MIN_PX, round(HARMONIZE_DILATE_RATIO * max(spec.width, spec.height))
    )
    ring = _dilate_and_soften(canvas, dilate_px)
    return Image.fromarray(_taper_interior(ring, canvas > 0.0, spec.harmonize_blend), mode="L")


def _smoothstep(ramp: np.ndarray) -> np.ndarray:
    """Rampa suave 0→1 (3t²-2t³): llega y sale sin quiebre en los extremos."""
    return ramp * ramp * (3.0 - 2.0 * ramp)


def _taper_interior(mask: np.ndarray, covered: np.ndarray, blend: float) -> np.ndarray:
    """Baja la intensidad hacia el centro de lo pegado; 1.0 la deja uniforme."""
    if blend >= 1.0 or not covered.any():
        return mask
    from scipy.ndimage import distance_transform_edt

    # SIN padding a propósito, al revés que feather_alpha: un objeto que toca el
    # borde del DESTINO no tiene costura ahí (no hay nada del otro lado), así que
    # esos píxeles cuentan como profundos y se preservan.
    depth = distance_transform_edt(covered)
    band = max(HARMONIZE_SEAM_MIN_PX, round(blend * float(depth.max())))
    tapered = mask.astype(np.float32) * _smoothstep(np.clip(1.0 - depth / band, 0.0, 1.0))
    return np.where(covered, np.rint(tapered), mask).astype(np.uint8)


def _dilate_and_soften(alpha_canvas: np.ndarray, dilate_px: int) -> np.ndarray:
    """Zona pegada agrandada y con borde blando, lista para el inpaint.

    El máximo con la versión sin blur mantiene el interior en 255: un blur solo
    también bajaría el centro y el inpaint armonizaría a medias (mismo problema
    que resuelve feather_mask en inpaint_mask.py).
    """
    covered = alpha_canvas > 0.0
    if not covered.any():
        return np.zeros(alpha_canvas.shape, dtype=np.uint8)
    from scipy.ndimage import distance_transform_edt

    dilated = np.where(distance_transform_edt(~covered) <= dilate_px, 255, 0).astype(np.uint8)
    blurred = Image.fromarray(dilated, mode="L").filter(ImageFilter.GaussianBlur(dilate_px / 2))
    return np.maximum(dilated, np.asarray(blurred))
