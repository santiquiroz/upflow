from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from app.services.inpaint_mask import (
    compute_crop_box,
    context_padding_for,
    dilate_mask,
    feather_mask,
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
    # volver a dibujar lo que se quería borrar.
    dilate_px: int
    # Ancho de la transición hacia afuera para que la unión no se vea.
    feather_px: int
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


def _bucket_up(value: int, multiple: int = SIZE_MULTIPLE) -> int:
    return max(multiple, -(-value // multiple) * multiple)


def resolve_model_side(
    crop_side: int, target_side: int, native_side: int | None
) -> tuple[int, int]:
    """(lado de trabajo, lado del canvas) con los que se llama al modelo.

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


def _resize_square(image: Any, side: int, resample: Any) -> Any:
    return image if image.size == (side, side) else image.resize((side, side), resample)


def _pad_to_canvas(image: Any, canvas_side: int, mode: str) -> Any:
    """Completa hasta el canvas del modelo SIN escalar el contenido.

    `edge` para la imagen (el modelo ve el borde continuado, no un marco negro
    que se cuele en la generación); `constant` (cero) para la máscara, porque
    el relleno no está marcado y no debe editarse. El resultado del modelo se
    recorta después de vuelta al área de trabajo, así que el relleno se tira.
    """
    from PIL import Image

    if image.size == (canvas_side, canvas_side):
        return image
    array = np.asarray(image)
    faltante = ((0, canvas_side - array.shape[0]), (0, canvas_side - array.shape[1]))
    faltante += ((0, 0),) * (array.ndim - 2)
    return Image.fromarray(np.pad(array, faltante, mode=mode))


def run_masked_edit(
    base_image: Any,
    mask_image: Any,
    model: MaskedEditModel,
    settings: MaskedEditSettings,
    on_prepared: Callable[[Any, Any], None] | None = None,
) -> Any:
    """Edita solo la región marcada y la devuelve pegada en la imagen original."""
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

    grown = dilate_mask(mask_array, settings.dilate_px)
    blended_mask = feather_mask(grown, settings.feather_px)

    caja_marcada = mask_bbox(blended_mask) or bbox
    crop_box = compute_crop_box(
        caja_marcada,
        resolve_padding(settings.padding_px, caja_marcada),
        base_rgb.size,
        # El contexto se toma del margen que sobra hasta la resolucion de
        # edicion, nunca del detalle: un recorte mas grande que esto se achica
        # para editarlo y vuelve a estirarse al pegarlo, dejando la zona mas
        # blanda que lo que la rodea.
        max_side=_bucket_up(settings.target_side),
    )
    crop_image = base_rgb.crop(crop_box)
    crop_mask = Image.fromarray(blended_mask, mode="L").crop(crop_box)

    work_side, canvas_side = resolve_model_side(
        max(crop_image.size), settings.target_side, settings.native_side
    )
    work_image = _resize_square(crop_image, work_side, Image.LANCZOS)
    # NEAREST no: la máscara ya trae su degradado y un resample duro lo perdería.
    work_mask = _resize_square(crop_mask, work_side, Image.BILINEAR)
    model_input = _pad_to_canvas(work_image, canvas_side, "edge")
    model_mask = _pad_to_canvas(work_mask, canvas_side, "constant")
    if on_prepared is not None:
        on_prepared(model_input, model_mask)

    edited = model(model_input, model_mask, canvas_side, canvas_side).convert("RGB")
    # Por si el pipeline redondea internamente y devuelve otro tamaño.
    edited = _resize_square(edited, canvas_side, Image.LANCZOS)
    # El canvas puede ser un poco más grande que el trabajo (bucketing): primero
    # se recorta lo trabajado y recién después se vuelve al tamaño real del
    # recorte, que es el que espera el stitch.
    edited_crop = edited.crop((0, 0, work_side, work_side))
    if edited_crop.size != crop_image.size:
        edited_crop = edited_crop.resize(crop_image.size, Image.LANCZOS)

    stitched = soft_composite(
        np.asarray(edited_crop), np.asarray(crop_image), np.asarray(crop_mask)
    )
    result = base_rgb.copy()
    result.paste(Image.fromarray(stitched), crop_box[:2])
    return result
