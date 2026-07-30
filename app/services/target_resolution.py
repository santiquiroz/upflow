from __future__ import annotations

import re
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

    None si la fuente ya llega sola. Se elige el mas chico porque todo lo que viene
    DESPUES del modelo escala con el cuadrado: los PNG intermedios, el disco y el
    encode. Ver la nota sobre el costo real mas abajo -- la inferencia en si casi no
    cambia con la escala.
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


# ---------------------------------------------------------------------------
# Elegir el MODELO segun el objetivo
#
# En este catalogo la escala esta pegada al modelo: realesrgan-x4plus solo ofrece
# [4], y los animevideov3-xN son un model_id por escala. Solo realesr-animevideov3
# ofrece [2,3,4]. Asi que para abaratar un pedido no alcanza con bajar la escala --
# hay que elegir OTRO modelo.
#
# DONDE ESTA EL COSTO, medido leyendo la red (scripts/export-realesrgan-onnx.py):
# los bloques RRDB, que son el cuerpo y lo dominante, corren a resolucion de ENTRADA
# y no cambian con la escala. Solo la cola (conv_up1 a 2x, conv_up2 a 4x) trabaja mas
# grande. Y los tres .bin de animevideov3 son IDENTICOS en tamaño: sus grafos x2/x3
# son derivados del x4 con un remuestreo hacia abajo.
#
# Consecuencia: elegir un modelo de 2x en vez de 4x NO divide por cuatro el tiempo de
# inferencia. Lo que si baja con el cuadrado de la escala es todo lo de despues --
# PNG intermedios, disco, encode -- y eso es real pero es otra cosa. Prometer "4 veces
# mas rapido" seria falso.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelChoice:
    model_id: str
    scale: int
    # El objetivo queda por encima de lo que este modelo alcanza. Se devuelve igual
    # porque es lo mejor disponible, pero el que llama tiene que poder avisarlo.
    falls_short: bool


def choose_model_for_target(
    candidates: tuple[tuple[str, int | None], ...],
    source_height: int,
    target_height: int,
) -> ModelChoice | None:
    """El modelo mas barato que ALCANZA el objetivo.

    Los candidatos son (model_id, escala). Una escala None significa desconocida --
    se excluye, porque planificar con un dato que no se tiene es adivinar.

    Si ninguno alcanza se devuelve el de mayor escala con falls_short, en vez de None:
    "lo mejor que puedo" es mas util que "no puedo", siempre que se diga.
    """
    usable = [(mid, scale) for mid, scale in candidates if scale is not None and scale > 0]
    if not usable:
        return None

    reaching = [
        (mid, scale) for mid, scale in usable if source_height * scale >= target_height
    ]
    if reaching:
        # El de menor escala: menos pixeles de salida, o sea menos PNG intermedios,
        # menos disco y menos encode. NO cuatro veces menos inferencia (ver la nota de
        # arriba sobre donde esta el costo).
        model_id, scale = min(reaching, key=lambda item: item[1])
        return ModelChoice(model_id=model_id, scale=scale, falls_short=False)

    model_id, scale = max(usable, key=lambda item: item[1])
    return ModelChoice(model_id=model_id, scale=scale, falls_short=True)


# Convencion de nombres de los repos de upscalers: "2x-AnimeSharpV4",
# "4x_foolhardy_Remacri", "RealESRGAN_x4plus". No es metadata, es una costumbre, asi
# que esto es una PISTA y no un dato: la escala real la detecta el instalador leyendo
# el grafo ONNX. Mostrarla como certeza seria prometer una deteccion que no existe.
# El digito NO puede tener otro digito al lado: sin eso "libx264" daria 2 y
# "ESRGAN-v22" daria 2. Un falso positivo es peor que no mostrar nada, porque haria
# que la UI etiquete "2x" un modelo de 4x.
_SCALE_HINT_PATTERNS = (
    # "2x-AnimeSharp", "4x_Remacri"
    re.compile(r"(?<!\d)([1-8])x(?![\da-z])", re.IGNORECASE),
    # "RealESRGAN_x4plus", "ESRGAN-x2"
    re.compile(r"x([1-8])(?!\d)", re.IGNORECASE),
)


def scale_hint_from_name(name: str) -> int | None:
    """Escala insinuada por el nombre del repo, o None si no se puede decir.

    Deliberadamente conservador: ante la duda devuelve None. Un falso positivo haria
    que la UI muestre "2x" sobre un modelo de 4x, y eso es peor que no mostrar nada.
    """
    for pattern in _SCALE_HINT_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None
