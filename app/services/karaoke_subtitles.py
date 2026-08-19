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

# Blanco con borde negro grueso: es lo que se lee sobre cualquier fondo, que es
# el caso real cuando el video de atras lo puso el usuario.
DEFAULT_STYLE = (
    "Style: Karaoke,Arial,48,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,"
    "-1,0,0,0,100,100,0,0,1,3,1,2,40,40,60,1"
)


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


def line_to_dialogue(line: KaraokeLine) -> str:
    if not line.words:
        return ""
    duraciones = karaoke_durations(list(line.words), line.start)
    texto = "".join(
        f"{{\\k{duracion}}}{palabra.text} "
        for palabra, duracion in zip(line.words, duraciones)
    ).strip()
    return (
        f"Dialogue: 0,{ass_timestamp(line.start)},{ass_timestamp(line.end)},"
        f"Karaoke,,0,0,0,,{texto}"
    )


def render_karaoke_ass(
    lines: list[KaraokeLine], *, width: int = 1280, height: int = 720
) -> str:
    """El archivo .ass completo.

    `PlayResX/Y` viajan porque libass escala los tamaños de fuente respecto de
    esa resolucion: sin declararla, la letra sale de otro tamaño segun el video.
    """
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
        DEFAULT_STYLE,
        "",
        "[Events]",
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
    ]
    dialogos = [d for d in (line_to_dialogue(linea) for linea in lines) if d]
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
