from __future__ import annotations

from dataclasses import dataclass

# Nadie quiere "x4": quiere que el resultado quede en 1080p. El multiplicador ciego
# es la causa de que un 4K con escala 4 pida 15360x8640 sin que nada avise -- 16 veces
# los pixeles de 4K, y casi nunca lo que la persona buscaba.
#
# Los modelos solo hacen escalados ENTEROS (2x, 3x, 4x), asi que llegar a una
# resolucion exacta necesita dos pasos: el entero mas chico que ALCANCE el objetivo, y
# despues un redimensionado a la medida exacta.

# Presets por ALTO, que es la convencion con la que se habla de resolucion de video.
# El ancho sale de la relacion de aspecto de la fuente, nunca de una tabla: recortar o
# estirar para encajar en un ancho fijo deformaria el video.
TARGET_PRESETS: dict[str, int] = {
    "480p": 480,
    "720p": 720,
    "1080p": 1080,
    "1440p": 1440,
    "2160p": 2160,
}


@dataclass(frozen=True, slots=True)
class ResolutionPlan:
    """Como llegar del tamaño de la fuente al objetivo.

    `model_scale` en None significa que NO hace falta correr el modelo: la fuente ya
    llega al objetivo y alcanza un redimensionado de ffmpeg. Es el caso que hoy no
    existe y el que convierte horas en segundos.
    """

    source_width: int
    source_height: int
    target_height: int
    model_scale: int | None
    output_width: int
    output_height: int
    needs_resize: bool
    # El objetivo queda por encima de lo que el escalado maximo puede dar, asi que el
    # redimensionado final AGRANDA sin modelo y no agrega detalle. Se avisa en vez de
    # entregarlo en silencio.
    exceeds_model_reach: bool


def _to_even(value: int) -> int:
    # yuv420p necesita dimensiones pares: un ancho impar hace fallar el encode.
    return value if value % 2 == 0 else value + 1


def _width_for_height(source_width: int, source_height: int, height: int) -> int:
    return _to_even(round(source_width * height / source_height))


def resolve_target_height(target: str | int) -> int:
    """Acepta un preset ("1080p") o un alto explicito."""
    if isinstance(target, int):
        if target <= 0:
            raise ValueError(f"target height must be positive, got {target}")
        return target
    preset = TARGET_PRESETS.get(target)
    if preset is None:
        raise ValueError(
            f"Unknown target resolution {target!r}. Known: {sorted(TARGET_PRESETS)}"
        )
    return preset


def smallest_scale_reaching(
    source_height: int, target_height: int, allowed_scales: tuple[int, ...]
) -> int | None:
    """El escalado entero mas chico que ALCANZA el objetivo.

    None si la fuente ya llega sola. Se elige el mas chico a proposito: correr el
    modelo a 4x para despues bajar a 1080p gasta horas de GPU en pixeles que se van a
    tirar.
    """
    if source_height >= target_height:
        return None
    for scale in sorted(allowed_scales):
        if source_height * scale >= target_height:
            return scale
    return max(allowed_scales) if allowed_scales else None


def plan_for_target(
    source_width: int,
    source_height: int,
    target: str | int,
    allowed_scales: tuple[int, ...] = (2, 3, 4),
) -> ResolutionPlan:
    if source_width <= 0 or source_height <= 0:
        raise ValueError(
            f"source dimensions must be positive, got {source_width}x{source_height}"
        )

    target_height = resolve_target_height(target)
    scale = smallest_scale_reaching(source_height, target_height, allowed_scales)

    reached_height = source_height if scale is None else source_height * scale
    exceeds = scale is not None and reached_height < target_height

    output_height = _to_even(target_height)
    output_width = _width_for_height(source_width, source_height, target_height)

    # Solo hace falta redimensionar si lo que quedo despues del modelo no es ya el
    # tamaño pedido.
    reached_width = (
        source_width if scale is None else source_width * scale
    )
    needs_resize = (reached_width, reached_height) != (output_width, output_height)

    return ResolutionPlan(
        source_width=source_width,
        source_height=source_height,
        target_height=target_height,
        model_scale=scale,
        output_width=output_width,
        output_height=output_height,
        needs_resize=needs_resize,
        exceeds_model_reach=exceeds,
    )


def megapixels_per_frame(width: int, height: int) -> float:
    return width * height / 1_000_000


def plan_for_scale(
    source_width: int, source_height: int, scale: int
) -> ResolutionPlan:
    """El camino viejo: multiplicador ciego, sin objetivo.

    Se expresa como un plan para que el resto del codigo trate los dos caminos igual y
    los avisos puedan comparar el costo de uno contra el otro.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    width = _to_even(source_width * scale)
    height = _to_even(source_height * scale)
    return ResolutionPlan(
        source_width=source_width,
        source_height=source_height,
        target_height=height,
        model_scale=scale,
        output_width=width,
        output_height=height,
        needs_resize=False,
        exceeds_model_reach=False,
    )
