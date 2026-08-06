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

SIZE_MULTIPLE = 64


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
    # Resolución nativa a la que se edita el recorte.
    target_side: int


def _snap(value: int, multiple: int = SIZE_MULTIPLE) -> int:
    return max(multiple, int(round(value / multiple)) * multiple)


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
        caja_marcada, resolve_padding(settings.padding_px, caja_marcada), base_rgb.size
    )
    crop_image = base_rgb.crop(crop_box)
    crop_mask = Image.fromarray(blended_mask, mode="L").crop(crop_box)

    side = _snap(settings.target_side)
    model_input = crop_image.resize((side, side), Image.LANCZOS)
    # NEAREST no: la máscara ya trae su degradado y un resample duro lo perdería.
    model_mask = crop_mask.resize((side, side), Image.BILINEAR)
    if on_prepared is not None:
        on_prepared(model_input, model_mask)

    edited = model(model_input, model_mask, side, side)
    edited_crop = edited.convert("RGB").resize(crop_image.size, Image.LANCZOS)

    stitched = soft_composite(
        np.asarray(edited_crop), np.asarray(crop_image), np.asarray(crop_mask)
    )
    result = base_rgb.copy()
    result.paste(Image.fromarray(stitched), crop_box[:2])
    return result
