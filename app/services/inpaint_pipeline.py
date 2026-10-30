from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from app.services.inpaint_grain import match_grain
from app.services.inpaint_mask import (
    compute_crop_box,
    context_padding_for,
    grow_per_component,
    mask_bbox,
    soft_composite,
)

# ---------------------------------------------------------------------------
# Edición "solo el área marcada" (crop-and-stitch), el flujo que usan A1111 y
# ComfyUI y el que hace la diferencia entre un parche que se nota y uno que no.
#
# En vez de mandarle al modelo la foto entera encogida a 1024 (donde la boca que
# se quiere borrar ocupa 40 px y sale reconstruida con 40 px de detalle), se
# recorta la región marcada CON CONTEXTO alrededor, se la lleva a la resolución
# nativa del modelo, se edita ahí, y se pega de vuelta en la foto ORIGINAL a su
# resolución completa. Dos beneficios que se notan enseguida: la zona editada
# recibe todo el detalle del modelo, y el resto de la foto no pierde un solo
# píxel de calidad porque nunca pasó por el modelo.
#
# El contexto (padding) no es decorativo: es lo que le permite al modelo
# continuar la piel, la pared o el pasto que rodean al objeto. Sin contexto el
# modelo pinta un parche que no pega con nada.
# ---------------------------------------------------------------------------

# Los shapes que se le piden al modelo se redondean HACIA ARRIBA a multiplos de
# 128 (bucketing): en DirectML cada shape distinto re-crea/re-compila la sesion
# ONNX, asi que recortes de 310 y 350 px pedirian dos sesiones; bucketeados caen
# los dos en 384 y la segunda edicion reusa la sesion de la primera (leccion de
# webui-directml/SD.Next). Antes era 64 al mas cercano; 128 sigue siendo multiplo
# del 64/8 que piden los latentes de Stable Diffusion, solo que junta mas shapes
# en el mismo balde.
SIZE_MULTIPLE = 128


def resolve_padding(explicito: int | None, bbox: tuple[int, int, int, int]) -> int:
    """El contexto pedido, o el que corresponde al tamaño de lo marcado.

    Se deja pisar a proposito: hay marcas que solo necesitan continuar una
    textura y prefieren gastar el recorte en detalle.
    """
    return explicito if explicito is not None else context_padding_for(bbox)


class MaskedEditModel(Protocol):
    def __call__(self, image: Any, mask: Any, width: int, height: int) -> Any: ...


@dataclass(frozen=True, slots=True)
class MaskedEditSettings:
    # Cuánto se agranda lo marcado: se come el halo del contorno que el pincel
    # no llegó a tapar y que, si queda, le sirve al modelo de plantilla para
    # volver a dibujar lo que se quería borrar. None = proporcional al tamaño
    # de lo marcado; 0 = no agrandar (la máscara de inserción ya trae su perfil).
    dilate_px: int | None
    # Ancho de la transición hacia afuera para que la unión no se vea. None =
    # proporcional; 0 = respeta el degradado que ya trae la máscara.
    feather_px: int | None
    # Contexto alrededor del área marcada que ve el modelo. `None` = se calcula
    # a partir del tamaño de lo marcado, que es lo correcto casi siempre: un
    # valor fijo deja sin cara al modelo cuando la marca es grande.
    padding_px: int | None
    # TECHO: resolución máxima a la que se edita el recorte. Un recorte más
    # grande se achica hasta acá para editarlo (ver compute_crop_box).
    target_side: int
    # PISO: resolución nativa del modelo (SDXL 1024, SD1.5 512). Un recorte más
    # chico se difundiría con menos píxeles de los que el modelo sabe usar y
    # devolvería anatomía blanda; se escala hacia arriba hasta este lado antes
    # de editar y se vuelve al tamaño real del recorte al pegar. `None` = sin
    # piso, el recorte se edita a su propio tamaño (bucketeado).
    native_side: int | None = None
    # Semilla del ruido que empareja el grano al pegar (ver match_grain).
    seed: int = 0
    # Copiar el grano del sensor del original a la zona editada para que el
    # relleno no se vea plástico.
    grain: bool = True


def _bucket_up(value: int, multiple: int = SIZE_MULTIPLE) -> int:
    return max(multiple, -(-value // multiple) * multiple)


def resolve_model_side(
    crop_side: int, target_side: int, native_side: int | None
) -> tuple[int, int]:
    """(lado de trabajo, lado del canvas) para el LADO MAYOR del recorte.

    El trabajo es el recorte tal cual, subido al piso nativo si quedó corto o
    bajado al techo si se pasó; el canvas es ese lado redondeado al bucket. El
    piso gana sobre el techo: `native_side` dice dónde rinde el modelo, y
    negárselo por un techo pensado para no ablandar recortes grandes sería
    devolver la anatomía blanda que el piso viene a arreglar.
    """
    if native_side is not None and crop_side < native_side:
        return native_side, _bucket_up(native_side)
    canvas = min(_bucket_up(crop_side), _bucket_up(target_side))
    return min(crop_side, canvas), canvas


def resolve_model_dims(
    crop_size: tuple[int, int], target_side: int, native_side: int | None
) -> tuple[tuple[int, int], tuple[int, int]]:
    """((ancho, alto) de trabajo, (ancho, alto) del canvas) para el modelo.

    Piso nativo y techo de edición se aplican sobre el LADO MAYOR con escala
    uniforme: el aspecto del recorte se preserva, porque deformarlo haría que
    el modelo devuelva contenido estirado. Cada eje del canvas bucketea por su
    cuenta. Un recorte cuadrado cae exactamente en el caso histórico de
    `resolve_model_side` en ambos ejes.
    """
    crop_width, crop_height = crop_size
    major = max(crop_width, crop_height)
    work_major, _ = resolve_model_side(major, target_side, native_side)
    scale = work_major / major
    work_width = work_major if crop_width == major else max(1, round(crop_width * scale))
    work_height = work_major if crop_height == major else max(1, round(crop_height * scale))
    return (work_width, work_height), (_bucket_up(work_width), _bucket_up(work_height))


def _resize(image: Any, size: tuple[int, int], resample: Any) -> Any:
    return image if image.size == size else image.resize(size, resample)


def _pad_to_canvas(image: Any, canvas_size: tuple[int, int], mode: str) -> Any:
    """Completa hasta el canvas del modelo SIN escalar el contenido.

    `edge` para la imagen (el modelo ve el borde continuado, no un marco negro
    que se cuele en la generación); `constant` (cero) para la máscara, porque
    el relleno no está marcado y no debe editarse. El resultado del modelo se
    recorta después de vuelta al área de trabajo, así que el relleno se tira.
    """
    from PIL import Image

    if image.size == canvas_size:
        return image
    canvas_width, canvas_height = canvas_size
    array = np.asarray(image)
    faltante = ((0, canvas_height - array.shape[0]), (0, canvas_width - array.shape[1]))
    faltante += ((0, 0),) * (array.ndim - 2)
    return Image.fromarray(np.pad(array, faltante, mode=mode))


def _solapan(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _union_cajas(cajas: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(c[0] for c in cajas),
        min(c[1] for c in cajas),
        max(c[2] for c in cajas),
        max(c[3] for c in cajas),
    )


def _fusionar_primer_solape(
    grupos: list[list[int]], cajas: dict[int, tuple[int, int, int, int]]
) -> bool:
    uniones = [_union_cajas([cajas[e] for e in grupo]) for grupo in grupos]
    for i in range(len(grupos)):
        for j in range(i + 1, len(grupos)):
            if _solapan(uniones[i], uniones[j]):
                grupos[i] += grupos.pop(j)
                return True
    return False


def _fusionar_solapadas(
    cajas: dict[int, tuple[int, int, int, int]]
) -> list[list[int]]:
    """Agrupa etiquetas cuyas cajas de recorte se solapan, hasta que no queden solapes."""
    grupos = [[etiqueta] for etiqueta in cajas]
    while _fusionar_primer_solape(grupos, cajas):
        pass
    return grupos


def _grupos_de_edicion(
    blended_mask: np.ndarray, image_size: tuple[int, int], settings: MaskedEditSettings
) -> list[tuple[tuple[int, int, int, int], np.ndarray]]:
    """Un (recorte, máscara) por región marcada, fusionando recortes solapados.

    Una sola caja para toda la máscara obligaba a dos marcas en extremos
    opuestos a compartir un recorte del tamaño de la foto, que al editarse a
    1024 se achicaba ~6x y volvía borroso: cada región se edita con su propio
    recorte a su resolución.
    """
    import cv2

    cantidad, etiquetas, stats, _ = cv2.connectedComponentsWithStats(
        (blended_mask > 0).astype(np.uint8), connectivity=8
    )
    cajas: dict[int, tuple[int, int, int, int]] = {}
    for etiqueta in range(1, cantidad):
        left, top, width, height, _ = stats[etiqueta]
        comp = (int(left), int(top), int(left + width), int(top + height))
        cajas[etiqueta] = compute_crop_box(
            comp,
            resolve_padding(settings.padding_px, comp),
            image_size,
            # El contexto se toma del margen que sobra hasta la resolucion de
            # edicion, nunca del detalle: un recorte mas grande que esto se achica
            # para editarlo y vuelve a estirarse al pegarlo, dejando la zona mas
            # blanda que lo que la rodea.
            max_side=_bucket_up(settings.target_side),
        )
    return [
        (
            _union_cajas([cajas[e] for e in miembros]),
            np.where(np.isin(etiquetas, miembros), blended_mask, 0),
        )
        for miembros in _fusionar_solapadas(cajas)
    ]


def _editar_grupo(
    canvas: Any,
    group_mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
    model: MaskedEditModel,
    settings: MaskedEditSettings,
    on_prepared: Callable[[Any, Any], None] | None,
) -> Any:
    """Edita un grupo con su propio recorte y lo devuelve pegado en el canvas."""
    from PIL import Image

    crop_image = canvas.crop(crop_box)
    crop_mask = Image.fromarray(group_mask, mode="L").crop(crop_box)

    work_size, canvas_size = resolve_model_dims(
        crop_image.size, settings.target_side, settings.native_side
    )
    work_image = _resize(crop_image, work_size, Image.LANCZOS)
    # NEAREST no: la máscara ya trae su degradado y un resample duro lo perdería.
    work_mask = _resize(crop_mask, work_size, Image.BILINEAR)
    model_input = _pad_to_canvas(work_image, canvas_size, "edge")
    model_mask = _pad_to_canvas(work_mask, canvas_size, "constant")
    if on_prepared is not None:
        on_prepared(model_input, model_mask)

    edited = model(model_input, model_mask, canvas_size[0], canvas_size[1]).convert("RGB")
    # Por si el pipeline redondea internamente y devuelve otro tamaño.
    edited = _resize(edited, canvas_size, Image.LANCZOS)
    # El canvas puede ser un poco más grande que el trabajo (bucketing): primero
    # se recorta lo trabajado y recién después se vuelve al tamaño real del
    # recorte, que es el que espera el stitch.
    edited_crop = edited.crop((0, 0, work_size[0], work_size[1]))
    if edited_crop.size != crop_image.size:
        edited_crop = edited_crop.resize(crop_image.size, Image.LANCZOS)

    if settings.grain:
        # El difusor no reproduce el grano del sensor: se copia del original.
        edited_array = match_grain(
            np.asarray(edited_crop),
            np.asarray(crop_image),
            np.asarray(crop_mask),
            seed=settings.seed,
        )
    else:
        edited_array = np.asarray(edited_crop)
    stitched = soft_composite(
        edited_array, np.asarray(crop_image), np.asarray(crop_mask)
    )
    result = canvas.copy()
    result.paste(Image.fromarray(stitched), crop_box[:2])
    return result


def run_masked_edit(
    base_image: Any,
    mask_image: Any,
    model: MaskedEditModel,
    settings: MaskedEditSettings,
    on_prepared: Callable[[Any, Any], None] | None = None,
) -> Any:
    """Edita solo las regiones marcadas y las devuelve pegadas en la original."""
    from PIL import Image

    base_rgb = base_image.convert("RGB")
    mask_gray = mask_image.convert("L")
    if mask_gray.size != base_rgb.size:
        mask_gray = mask_gray.resize(base_rgb.size, Image.NEAREST)

    mask_array = np.asarray(mask_gray)
    bbox = mask_bbox(mask_array)
    if bbox is None:
        # Nada marcado: devolver la foto tal cual es más honesto que llamar al
        # modelo y regalarle al usuario una imagen distinta sin motivo.
        return base_rgb

    blended_mask = grow_per_component(mask_array, settings.dilate_px, settings.feather_px)

    result = base_rgb
    for crop_box, group_mask in _grupos_de_edicion(blended_mask, base_rgb.size, settings):
        result = _editar_grupo(result, group_mask, crop_box, model, settings, on_prepared)
    return result
