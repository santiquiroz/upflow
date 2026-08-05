"""Segmentos con tiempo y su serializacion a formatos de subtitulo.

Whisper devuelve los tiempos POR CHUNK y cada chunk arranca en cero, asi que el
corrimiento por posicion en el audio es parte del contrato y no un detalle del
llamador: sin el, todos los subtitulos quedan encimados sobre el primer medio
minuto.

Modulo puro a proposito: nada de esto necesita el modelo cargado, asi que se
prueba entero sin GPU ni descargas.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str = field(default="")

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(
                f"Un segmento no puede terminar antes de empezar: {self.start} > {self.end}"
            )
        object.__setattr__(self, "text", self.text.strip())

    def shifted(self, offset_seconds: float) -> TranscriptSegment:
        return TranscriptSegment(
            start=self.start + offset_seconds,
            end=self.end + offset_seconds,
            text=self.text,
        )


def _clock_parts(seconds: float) -> tuple[int, int, int, int]:
    total_ms = max(0, round(seconds * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return hours, minutes, secs, millis


# SRT separa los milisegundos con COMA y WebVTT con PUNTO. Usar el separador del
# otro formato deja un archivo que el reproductor ignora sin decir nada.
def srt_timestamp(seconds: float) -> str:
    hours, minutes, secs, millis = _clock_parts(seconds)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def vtt_timestamp(seconds: float) -> str:
    hours, minutes, secs, millis = _clock_parts(seconds)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


# El procesador devuelve [{"text": ..., "timestamp": (inicio, fin)}]. Un chunk
# cortado al ras puede traer `fin=None`: el modelo no llego a cerrar el segmento.
# Descartarlo perderia texto, asi que se cierra en el borde del chunk.
def segments_from_offsets(
    offsets: list[dict[str, object]], *, chunk_seconds: int
) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for entry in offsets:
        timestamp = entry.get("timestamp")
        if not isinstance(timestamp, (tuple, list)) or len(timestamp) != 2:
            continue
        start, end = timestamp
        if start is None:
            continue
        closed_end = chunk_seconds if end is None else end
        text = str(entry.get("text") or "")
        segments.append(
            TranscriptSegment(start=float(start), end=float(closed_end), text=text)
        )
    return segments


def merge_chunk_segments(
    per_chunk: list[list[TranscriptSegment]], *, chunk_seconds: int
) -> list[TranscriptSegment]:
    merged: list[TranscriptSegment] = []
    for index, segments in enumerate(per_chunk):
        offset = index * chunk_seconds
        merged.extend(
            segment.shifted(offset) for segment in segments if segment.text
        )
    return merged


def segments_to_srt(segments: list[TranscriptSegment]) -> str:
    blocks = [
        f"{number}\n{srt_timestamp(segment.start)} --> {srt_timestamp(segment.end)}\n{segment.text}\n"
        for number, segment in enumerate(segments, start=1)
    ]
    return "\n".join(blocks) + "\n" if blocks else ""


def segments_to_vtt(segments: list[TranscriptSegment]) -> str:
    blocks = [
        f"{vtt_timestamp(segment.start)} --> {vtt_timestamp(segment.end)}\n{segment.text}\n"
        for segment in segments
    ]
    return "WEBVTT\n\n" + "\n".join(blocks)


def segments_to_text(segments: list[TranscriptSegment]) -> str:
    return " ".join(segment.text for segment in segments if segment.text).strip()


@dataclass(frozen=True, slots=True)
class SubtitleFormat:
    extension: str
    media_type: str


# Un .srt servido como text/plain lo abre el navegador en vez de bajarlo, asi
# que el tipo MIME es parte del contrato y no un detalle.
SUBTITLE_FORMATS: dict[str, SubtitleFormat] = {
    "txt": SubtitleFormat(extension=".txt", media_type="text/plain; charset=utf-8"),
    "srt": SubtitleFormat(extension=".srt", media_type="application/x-subrip"),
    "vtt": SubtitleFormat(extension=".vtt", media_type="text/vtt"),
}

_RENDERERS = {
    "txt": segments_to_text,
    "srt": segments_to_srt,
    "vtt": segments_to_vtt,
}


def render_segments(segments: list[TranscriptSegment], fmt: str) -> str:
    renderer = _RENDERERS.get(fmt)
    if renderer is None:
        raise ValueError(f"Formato de subtitulo desconocido: {fmt!r}")
    return renderer(segments)
