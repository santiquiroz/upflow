"""Strings de filtros de ffmpeg para el modo CCTV (spec §4.4, §4.6, §4.7, §4.8).

Recibe pasos ya validados por `cctv_chain.steps_from_request` y arma el texto:
un filtro por paso, una sola `-vf`, el grafo del carril clasico con el OSD
restituido desde el original y las dimensiones que salen de la cadena. Los
valores se vuelven a chequear aca (numeros finitos o identificadores) para que
un `ResolvedStep` armado a mano no pueda inyectar sintaxis de filtergraph.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import PurePath
from typing import Any

from app.services.cctv_chain import (
    INVALID_PARAM,
    MISSING_PARAM,
    CctvChainError,
    FilterSpec,
    ResolvedStep,
    StepSpec,
    in_catalog_order,
    step_spec,
)

NOT_A_FILTER = "cctv.error.notAFilter"
CROP_OUTSIDE_FRAME = "cctv.error.cropOutsideFrame"
OSD_BOX_OUTSIDE_FRAME = "cctv.error.osdBoxOutsideFrame"

EXACT_SCALE_FLAGS: tuple[str, ...] = ("accurate_rnd", "full_chroma_int", "bitexact")
DEFAULT_SCALE_ALGORITHM = "neighbor"
SETSAR_MAX = 1000
OSD_ALIGNMENT = 2
NULL_FILTER = "null"
FILTER_SEPARATOR = ","

# Nivel 1 (valor de opcion) y nivel 2 (descripcion del filtergraph) de la
# documentacion de ffmpeg. Como el argv no pasa por un shell, no hay nivel 3.
_OPTION_LEVEL_SPECIALS = "\\':"
_GRAPH_LEVEL_SPECIALS = "\\'[],;"
_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_LABEL = re.compile(r"[A-Za-z0-9_:]+")
_OSD_BRANCH_STEPS = frozenset({"aspect", "deinterlace", "crop", "gray", "scale"})
_UNBUILDABLE_STEPS = frozenset({"osd_protect"})

Box = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class FrameGeometry:
    width: int
    height: int
    sar: Fraction = Fraction(1)


def _escape_chars(text: str, specials: str) -> str:
    return "".join(f"\\{char}" if char in specials else char for char in text)


def escape_filter_path(path: PurePath | str) -> str:
    posix = str(path).replace("\\", "/")
    return _escape_chars(_escape_chars(posix, _OPTION_LEVEL_SPECIALS), _GRAPH_LEVEL_SPECIALS)


def scale_flags(algorithm: str = DEFAULT_SCALE_ALGORITHM) -> str:
    return "+".join((_checked_token(algorithm), *EXACT_SCALE_FLAGS))


def build_scale_to(width: int, height: int, algorithm: str = DEFAULT_SCALE_ALGORITHM) -> str:
    _require_positive_int("width", width)
    _require_positive_int("height", height)
    return f"scale=w={width}:h={height}:flags={scale_flags(algorithm)}"


def _unsafe(reason: str) -> CctvChainError:
    return CctvChainError(INVALID_PARAM, f"Unsafe filter value: {reason}.")


def _checked_token(value: Any) -> str:
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise _unsafe(f"{value!r} is not a plain identifier")
    return value


def _format_number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _unsafe(f"{value!r} is not a number")
    if isinstance(value, float) and not math.isfinite(value):
        raise _unsafe(f"{value!r} is not finite")
    return str(value) if isinstance(value, int) else repr(value)


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return _checked_token(value)
    return _format_number(value)


def _require_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _unsafe(f"{name} must be a positive integer, got {value!r}")


def _param(step: ResolvedStep, name: str) -> Any:
    if name not in step.params:
        raise CctvChainError(MISSING_PARAM, f"Step {step.id!r} needs parameter {name!r}.")
    return step.params[name]


def _int_param(step: ResolvedStep, name: str) -> int:
    value = _param(step, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise _unsafe(f"{step.id}.{name} must be an integer, got {value!r}")
    return value


def _filter_spec_of(spec: StepSpec, step: ResolvedStep) -> FilterSpec:
    for candidate in spec.filters:
        if candidate.name == step.filter:
            return candidate
    raise CctvChainError(INVALID_PARAM, f"Step {step.id!r} has no filter {step.filter!r}.")


def _not_a_filter(step: ResolvedStep, reason: str) -> CctvChainError:
    return CctvChainError(NOT_A_FILTER, f"Step {step.id!r} is not an ffmpeg filter: {reason}.")


def _buildable_spec(step: ResolvedStep) -> FilterSpec:
    spec = step_spec(step.id)
    if spec.deferred_to is not None:
        raise _not_a_filter(step, f"deferred to {spec.deferred_to}")
    if spec.category != "classic":
        raise _not_a_filter(step, f"it runs in the {spec.category} stage")
    if step.id in _UNBUILDABLE_STEPS:
        raise _not_a_filter(step, "use build_osd_graph")
    return _filter_spec_of(spec, step)


def _options(spec: FilterSpec, step: ResolvedStep) -> str:
    return ":".join(f"{param.option or param.name}={_format_value(_param(step, param.name))}" for param in spec.params)


def _generic_filter(spec: FilterSpec, step: ResolvedStep) -> str:
    name = spec.ffmpeg_filters[0]
    options = _options(spec, step)
    return f"{name}={options}" if options else name


def _trim_filter(spec: FilterSpec, step: ResolvedStep) -> str:
    first_dropped = _int_param(step, "end_frame") + 1
    return f"trim=start_frame={_int_param(step, 'start_frame')}:end_frame={first_dropped}"


def _setsar_filter(spec: FilterSpec, step: ResolvedStep) -> str:
    sar = _step_sar(step)
    return f"setsar=sar={sar.numerator}/{sar.denominator}:max={SETSAR_MAX}"


def _crop_filter(spec: FilterSpec, step: ResolvedStep) -> str:
    return _crop_text(_crop_rect(step))


def _tmedian_filter(spec: FilterSpec, step: ResolvedStep) -> str:
    radius = _int_param(step, "radius")
    # tmedian por si solo saca cuadros con la ventana llena: pierde 2*radius al
    # final y la salida k es la mediana centrada en k+radius. Clonar los bordes
    # conserva el conteo y centra la ventana (PTS exactos solo en CFR).
    pad = f"tpad=start={radius}:start_mode=clone:stop={radius}:stop_mode=clone"
    return _join([pad, _generic_filter(spec, step)])


def _gray_filter(spec: FilterSpec, step: ResolvedStep) -> str:
    return "format=pix_fmts=gray"


def _scale_filter(spec: FilterSpec, step: ResolvedStep) -> str:
    factor = _scale_factor(step)
    return f"scale=w=iw*{factor}:h=ih*{factor}:flags={scale_flags(_param(step, 'flags'))}"


_BUILDERS: Mapping[str, Callable[[FilterSpec, ResolvedStep], str]] = {
    "trim": _trim_filter,
    "setsar": _setsar_filter,
    "crop": _crop_filter,
    "gray": _gray_filter,
    "tmedian": _tmedian_filter,
    "scale": _scale_filter,
}


def build_filter(step: ResolvedStep) -> str:
    spec = _buildable_spec(step)
    return _BUILDERS.get(spec.name, _generic_filter)(spec, step)


def _join(filters: Sequence[str]) -> str:
    return FILTER_SEPARATOR.join(filters)


def compose_vf(steps: Sequence[ResolvedStep]) -> str:
    return _join([build_filter(step) for step in in_catalog_order(steps)])


def vf_args(steps: Sequence[ResolvedStep]) -> list[str]:
    graph = compose_vf(steps)
    return ["-vf", graph] if graph else []


def _step_sar(step: ResolvedStep) -> Fraction:
    return Fraction(_int_param(step, "num"), _int_param(step, "den"))


def _scale_factor(step: ResolvedStep) -> int:
    factor = _int_param(step, "factor")
    _require_positive_int("scale factor", factor)
    return factor


def _crop_rect(step: ResolvedStep) -> Box:
    return (_int_param(step, "x"), _int_param(step, "y"), _int_param(step, "w"), _int_param(step, "h"))


def _check_crop_fits(rect: Box, geometry: FrameGeometry) -> None:
    x, y, w, h = rect
    if min(x, y) < 0 or min(w, h) < 1 or x + w > geometry.width or y + h > geometry.height:
        raise CctvChainError(
            CROP_OUTSIDE_FRAME,
            f"Crop {w}x{h} at ({x}, {y}) doesn't fit in the {geometry.width}x{geometry.height} frame.",
        )


def _cropped(geometry: FrameGeometry, step: ResolvedStep) -> FrameGeometry:
    rect = _crop_rect(step)
    _check_crop_fits(rect, geometry)
    return FrameGeometry(rect[2], rect[3], geometry.sar)


def _scaled(geometry: FrameGeometry, step: ResolvedStep) -> FrameGeometry:
    factor = _scale_factor(step)
    return FrameGeometry(geometry.width * factor, geometry.height * factor, geometry.sar)


def _with_sar(geometry: FrameGeometry, step: ResolvedStep) -> FrameGeometry:
    return FrameGeometry(geometry.width, geometry.height, _step_sar(step))


_GEOMETRY_CHANGES: Mapping[str, Callable[[FrameGeometry, ResolvedStep], FrameGeometry]] = {
    "crop": _cropped,
    "scale": _scaled,
    "aspect": _with_sar,
}


def _check_geometry_step(step: ResolvedStep) -> None:
    spec = step_spec(step.id)
    if spec.deferred_to is not None or step.id == "ai_upscale":
        raise _not_a_filter(step, "its output size isn't known here")


def _apply_geometry(geometry: FrameGeometry, step: ResolvedStep) -> FrameGeometry:
    _check_geometry_step(step)
    change = _GEOMETRY_CHANGES.get(step.id)
    return change(geometry, step) if change else geometry


def output_dims_after(
    steps: Sequence[ResolvedStep], width: int, height: int, sar: Fraction = Fraction(1)
) -> FrameGeometry:
    _require_positive_int("width", width)
    _require_positive_int("height", height)
    geometry = FrameGeometry(width, height, Fraction(sar))
    for step in in_catalog_order(steps):
        geometry = _apply_geometry(geometry, step)
    return geometry


def _check_box(box: Box, width: int, height: int) -> None:
    if len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, int) for v in box):
        raise CctvChainError(OSD_BOX_OUTSIDE_FRAME, f"OSD box {box!r} must be four integers [x, y, w, h].")
    x, y, w, h = box
    if x < 0 or y < 0 or w < 1 or h < 1 or x + w > width or y + h > height:
        raise CctvChainError(
            OSD_BOX_OUTSIDE_FRAME, f"OSD box {w}x{h} at ({x}, {y}) is outside the {width}x{height} frame."
        )


def _crop_box(box: Box | None, step: ResolvedStep) -> Box | None:
    if box is None:
        return None
    cx, cy, cw, ch = _crop_rect(step)
    left, top = max(box[0], cx), max(box[1], cy)
    right, bottom = min(box[0] + box[2], cx + cw), min(box[1] + box[3], cy + ch)
    if right <= left or bottom <= top:
        return None
    return (left - cx, top - cy, right - left, bottom - top)


def _scale_box(box: Box | None, step: ResolvedStep) -> Box | None:
    if box is None:
        return None
    factor = _scale_factor(step)
    return (box[0] * factor, box[1] * factor, box[2] * factor, box[3] * factor)


_BOX_CHANGES: Mapping[str, Callable[[Box | None, ResolvedStep], Box | None]] = {
    "crop": _crop_box,
    "scale": _scale_box,
}


def _aligned_box(box: Box, geometry: FrameGeometry) -> Box:
    # Cajas en coordenadas pares: crop y overlay redondean distinto la croma
    # submuestreada, y una caja impar quedaria corrida medio pixel de croma.
    left = box[0] - box[0] % OSD_ALIGNMENT
    top = box[1] - box[1] % OSD_ALIGNMENT
    right = min(geometry.width, box[0] + box[2] + (box[0] + box[2]) % OSD_ALIGNMENT)
    bottom = min(geometry.height, box[1] + box[3] + (box[1] + box[3]) % OSD_ALIGNMENT)
    return (left, top, right - left, bottom - top)


def _box_after(box: Box, steps: Sequence[ResolvedStep]) -> Box | None:
    moved: Box | None = box
    for step in steps:
        change = _BOX_CHANGES.get(step.id)
        moved = change(moved, step) if change else moved
    return moved


def osd_boxes_after(
    steps: Sequence[ResolvedStep], boxes: Sequence[Box], source: FrameGeometry
) -> tuple[Box, ...]:
    ordered = in_catalog_order(steps)
    final = output_dims_after(ordered, source.width, source.height, source.sar)
    for box in boxes:
        _check_box(tuple(box), source.width, source.height)
    moved = (_box_after(tuple(box), ordered) for box in boxes)
    return tuple(_aligned_box(box, final) for box in moved if box is not None)


def _checked_label(label: Any) -> str:
    if not isinstance(label, str) or not _LABEL.fullmatch(label):
        raise _unsafe(f"{label!r} is not a filtergraph label")
    return label


def _chain_or_null(steps: Sequence[ResolvedStep]) -> str:
    return compose_vf(steps) or NULL_FILTER


def _output_suffix(pix_fmt: str | None) -> str:
    return "" if pix_fmt is None else f"{FILTER_SEPARATOR}format=pix_fmts={_checked_token(pix_fmt)}"


def _crop_text(box: Box) -> str:
    x, y, w, h = box
    # exact=1: sin el, crop redondea x/y impares al submuestreo de croma y la
    # imagen queda corrida respecto de las coordenadas declaradas.
    return f"crop=w={w}:h={h}:x={x}:y={y}:exact=1"


def _overlay_text(box: Box) -> str:
    # shortest=1 y repeatlast=0: si una rama pierde cuadros, el grafo termina
    # antes y el chequeo de conteo lo ve, en vez de rellenar con repetidos.
    return f"overlay=x={box[0]}:y={box[1]}:format=auto:shortest=1:repeatlast=0"


def _split(count: int, labels: Sequence[str]) -> str:
    return f"split={count}" + "".join(f"[{label}]" for label in labels)


def _head_segment(source: str, trim: Sequence[ResolvedStep], outputs: Sequence[str]) -> str:
    filters = [build_filter(step) for step in trim]
    return f"[{source}]" + _join([*filters, _split(len(outputs), outputs)])


def _osd_branch_segments(branch: Sequence[ResolvedStep], boxes: Sequence[Box]) -> list[str]:
    prefix = [build_filter(step) for step in branch]
    if len(boxes) == 1:
        return [f"[o]{_join([*prefix, _crop_text(boxes[0])])}[osd0]"]
    sources = [f"o{index}" for index in range(len(boxes))]
    fan_out = f"[o]{_join([*prefix, _split(len(boxes), sources)])}"
    crops = [f"[o{index}]{_crop_text(box)}[osd{index}]" for index, box in enumerate(boxes)]
    return [fan_out, *crops]


def _overlay_segments(boxes: Sequence[Box], output: str, pix_fmt: str | None) -> list[str]:
    segments = []
    for index, box in enumerate(boxes):
        last = index == len(boxes) - 1
        target = output if last else f"p{index + 1}"
        suffix = _output_suffix(pix_fmt) if last else ""
        segments.append(f"[p{index}][osd{index}]{_overlay_text(box)}{suffix}[{target}]")
    return segments


def _plain_graph(steps: Sequence[ResolvedStep], source: str, output: str, pix_fmt: str | None) -> str:
    return f"[{source}]{_chain_or_null(steps)}{_output_suffix(pix_fmt)}[{output}]"


def build_osd_graph(
    steps: Sequence[ResolvedStep],
    boxes: Sequence[Box],
    source: FrameGeometry,
    *,
    input_label: str = "0:v",
    output_label: str = "v",
    pix_fmt: str | None = None,
) -> str:
    source_label, output = _checked_label(input_label), _checked_label(output_label)
    ordered = tuple(step for step in in_catalog_order(steps) if step.id not in _UNBUILDABLE_STEPS)
    placed = osd_boxes_after(ordered, boxes, source)
    if not placed:
        return _plain_graph(ordered, source_label, output, pix_fmt)
    trim = [step for step in ordered if step.id == "trim"]
    chain = [step for step in ordered if step.id != "trim"]
    branch = [step for step in chain if step.id in _OSD_BRANCH_STEPS]
    segments = [
        _head_segment(source_label, trim, ["m", "o"]),
        f"[m]{_chain_or_null(chain)}[p0]",
        *_osd_branch_segments(branch, placed),
        *_overlay_segments(placed, output, pix_fmt),
    ]
    return ";".join(segments)
