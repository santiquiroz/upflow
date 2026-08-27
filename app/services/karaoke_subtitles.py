"""Subtitulos ASS con la letra encendiendose palabra por palabra.

Un karaoke con la linea entera en blanco es un video con subtitulos. Lo que lo
vuelve karaoke es saber DONDE va la voz ahora, y eso se dice con las etiquetas
`\\k` de ASS: cada palabra lleva cuanto dura, y libass —el que usa ffmpeg— pinta
el relleno a medida que pasa.

Se eligio ASS y no SRT porque SRT no tiene forma de expresar esto: habria que
emitir una linea nueva por palabra, parpadeando, que es peor que no resaltar.
"""

from __future__ import annotations

from dataclasses import dataclass

# ASS mide la duracion de cada `\k` en CENTISEGUNDOS, no en milisegundos ni en
# segundos. Con la unidad equivocada el resaltado corre 10x y se despega en la
# primera linea.
CENTISECONDS_PER_SECOND = 100

# Tamaños en puntos ASS sobre PlayRes 1280x720; libass escala al video real.
FONT_SIZES = {"small": 36, "medium": 48, "large": 64}
# Numpad ASS: 2 = abajo-centro, 8 = arriba-centro.
ALIGNMENT_BY_POSITION = {"bottom": 2, "top": 8}
# La traduccion acompaña, no compite: mas chica que la linea principal.
TRANSLATION_FONT_RATIO = 0.7


@dataclass(frozen=True, slots=True)
class KaraokeStyle:
    """Lo que el usuario puede tocar del subtitulo.

    `base_color` es el texto ANTES de cantarse y `highlight_color` el ya
    cantado. En ASS eso es al reves de lo que suena: el karaoke `\\k` arranca
    en SecondaryColour y se rellena hacia PrimaryColour.
    """

    size: str = "medium"
    position: str = "bottom"
    base_color: str = "#FFFF00"
    highlight_color: str = "#FFFFFF"


def hex_to_ass_color(color: str) -> str:
    """`#RRGGBB` a `&H00BBGGRR`: ASS guarda los canales al reves y con alfa."""
    value = color.strip().lstrip("#")
    if len(value) != 6 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ValueError(f"Color invalido: {color!r}. Se espera #RRGGBB.")
    rr, gg, bb = value[0:2], value[2:4], value[4:6]
    return f"&H00{bb.upper()}{gg.upper()}{rr.upper()}"


def _font_size(style: KaraokeStyle) -> int:
    if style.size not in FONT_SIZES:
        raise ValueError(
            f"Tamaño invalido: {style.size!r}. Validos: {', '.join(FONT_SIZES)}."
        )
    return FONT_SIZES[style.size]


def _alignment(style: KaraokeStyle) -> int:
    if style.position not in ALIGNMENT_BY_POSITION:
        raise ValueError(
            f"Posicion invalida: {style.position!r}. "
            f"Validas: {', '.join(ALIGNMENT_BY_POSITION)}."
        )
    return ALIGNMENT_BY_POSITION[style.position]


def _style_line(
    name: str, font_size: int, primary: str, secondary: str, alignment: int, margin_v: int
) -> str:
    # Borde negro grueso: es lo que se lee sobre cualquier fondo, que es el
    # caso real cuando el video de atras lo puso el usuario.
    return (
        f"Style: {name},Arial,{font_size},{primary},{secondary},&H00000000,"
        f"&H80000000,-1,0,0,0,100,100,0,0,1,3,1,{alignment},40,40,{margin_v},1"
    )


def build_style_lines(style: KaraokeStyle) -> list[str]:
    """El estilo principal y el de la traduccion, coherentes entre si.

    Con alineacion abajo, MarginV mas grande = mas arriba: la principal va a 60
    y la traduccion debajo a 16. Arriba es al reves, asi que la traduccion baja
    sumando el alto de la principal.
    """
    principal = _font_size(style)
    traduccion = int(round(principal * TRANSLATION_FONT_RATIO))
    alignment = _alignment(style)
    primary = hex_to_ass_color(style.highlight_color)
    secondary = hex_to_ass_color(style.base_color)
    if alignment == ALIGNMENT_BY_POSITION["bottom"]:
        margen_principal, margen_traduccion = 60, 16
    else:
        margen_principal = 40
        margen_traduccion = margen_principal + principal + 12
    return [
        _style_line("Karaoke", principal, primary, secondary, alignment, margen_principal),
        _style_line(
            "Translation", traduccion, primary, secondary, alignment, margen_traduccion
        ),
    ]


@dataclass(frozen=True, slots=True)
class KaraokeWord:
    text: str
    start: float
    end: float


@dataclass(frozen=True, slots=True)
class KaraokeLine:
    start: float
    end: float
    words: tuple[KaraokeWord, ...]


def ass_timestamp(seconds: float) -> str:
    """`h:mm:ss.cc`. ASS usa UNA cifra de hora y centesimas, no milesimas."""
    seconds = max(0.0, seconds)
    horas, resto = divmod(int(seconds), 3600)
    minutos, segundos = divmod(resto, 60)
    centesimas = int(round((seconds - int(seconds)) * CENTISECONDS_PER_SECOND))
    if centesimas == CENTISECONDS_PER_SECOND:
        centesimas = 0
        segundos += 1
    return f"{horas}:{minutos:02d}:{segundos:02d}.{centesimas:02d}"


def karaoke_durations(words: list[KaraokeWord], line_start: float) -> list[int]:
    """Cuanto dura cada `\\k`, en centisegundos, incluyendo el hueco previo.

    El hueco ANTES de una palabra se le suma a esa palabra y no se pierde: `\\k`
    no tiene forma de decir "espera y despues empieza", asi que si se descartan
    los silencios el resaltado se adelanta y termina antes que la voz.
    """
    duraciones: list[int] = []
    cursor = line_start
    for palabra in words:
        fin = max(palabra.end, palabra.start)
        centesimas = int(round((fin - cursor) * CENTISECONDS_PER_SECOND))
        # Nunca cero: un `\k0` no avanza y libass pinta la palabra de golpe.
        duraciones.append(max(1, centesimas))
        cursor = fin
    return duraciones


def line_to_dialogue(line: KaraokeLine, style_name: str = "Karaoke") -> str:
    if not line.words:
        return ""
    duraciones = karaoke_durations(list(line.words), line.start)
    texto = "".join(
        f"{{\\k{duracion}}}{palabra.text} "
        for palabra, duracion in zip(line.words, duraciones)
    ).strip()
    return (
        f"Dialogue: 0,{ass_timestamp(line.start)},{ass_timestamp(line.end)},"
        f"{style_name},,0,0,0,,{texto}"
    )


def render_karaoke_ass(
    lines: list[KaraokeLine],
    *,
    width: int = 1280,
    height: int = 720,
    translations: list[str] | None = None,
    style: KaraokeStyle | None = None,
) -> str:
    """El archivo .ass completo.

    `PlayResX/Y` viajan porque libass escala los tamaños de fuente respecto de
    esa resolucion: sin declararla, la letra sale de otro tamaño segun el video.

    `translations` acompaña por INDICE a `lines`: la linea traducida hereda los
    tiempos de la original y se resalta proporcional por letras — los tiempos
    por palabra solo existen de verdad en el idioma que se canta.
    """
    estilo = style or KaraokeStyle()
    cabecera = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "",
        "[V4+ Styles]",
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
        "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding",
        *build_style_lines(estilo),
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    dialogos = [d for d in (line_to_dialogue(linea) for linea in lines) if d]
    if translations:
        for linea, traduccion in zip(lines, translations):
            if not (traduccion or "").strip():
                continue
            traducida = split_line_proportionally(traduccion, linea.start, linea.end)
            dialogo = line_to_dialogue(traducida, style_name="Translation")
            if dialogo:
                dialogos.append(dialogo)
    return "\n".join(cabecera + dialogos) + "\n"


def split_line_proportionally(
    text: str, start: float, end: float
) -> KaraokeLine:
    """Reparte una linea entre sus palabras por cantidad de letras.

    Es una APROXIMACION y hay que decirlo: una nota sostenida sobre una palabra
    corta la corre visiblemente. Existe para que el resaltado funcione con los
    modelos que solo entregan tiempos por linea; cuando hay tiempos por palabra
    de verdad, no se usa.
    """
    palabras = text.split()
    if not palabras:
        return KaraokeLine(start=start, end=end, words=())
    letras = [max(1, len(p)) for p in palabras]
    total = sum(letras)
    duracion = max(0.0, end - start)
    cronometrado: list[KaraokeWord] = []
    cursor = start
    for palabra, peso in zip(palabras, letras):
        siguiente = cursor + duracion * (peso / total)
        cronometrado.append(KaraokeWord(text=palabra, start=cursor, end=siguiente))
        cursor = siguiente
    return KaraokeLine(start=start, end=end, words=tuple(cronometrado))


def line_from_segment(segment) -> KaraokeLine:
    """La linea de karaoke de un segmento, usando lo mejor que tenga.

    Con tiempos por palabra de verdad se usan esos. Sin ellos se reparte por
    cantidad de letras, que es una aproximacion visible en notas sostenidas. La
    decision vive aca y no en el que llama para que no haya dos criterios.
    """
    palabras = getattr(segment, "words", ()) or ()
    if not palabras:
        return split_line_proportionally(segment.text, segment.start, segment.end)
    return KaraokeLine(
        start=segment.start,
        end=segment.end,
        words=tuple(
            KaraokeWord(text=p.word, start=p.start, end=p.end) for p in palabras
        ),
    )
