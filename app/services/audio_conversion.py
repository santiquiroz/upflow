from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.models import AudioJob

# Conversion directa: UNA pasada de ffmpeg del archivo original al formato
# destino, sin pasar por el decode a WAV 48 kHz / 16 bits que el pipeline de
# procesamiento necesita para sus motores. Ese decode existe porque
# DeepFilterNet, RNNoise y los separadores estan especificados a una tasa fija;
# aplicarlo a alguien que solo quiere un FLAC 44.1/24 como MP3 le cambiaria la
# tasa y le truncaria la profundidad sin que nada lo diga.

LOSSLESS_OUTPUT_FORMATS = frozenset({"wav", "flac"})
LOSSY_OUTPUT_FORMATS = frozenset({"mp3", "m4a"})

# Tasas que acepta cada codec con perdida. Fuera de estas ffmpeg resamplea SOLO
# y en silencio (medido con el ffmpeg vendorizado: 96 kHz -> MP3 sale a 48 kHz
# sin una linea de aviso). Resolver la tasa aca permite pasarla explicita y
# dejar el cambio registrado en el job.
_LOSSY_SAMPLE_RATES: dict[str, tuple[int, ...]] = {
    "mp3": (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000),
    "m4a": (
        8000, 11025, 12000, 16000, 22050, 24000,
        32000, 44100, 48000, 64000, 88200, 96000,
    ),
}

# libmp3lame es mono/estereo: un 5.1 se downmixea si o si (tambien medido).
_MAX_OUTPUT_CHANNELS: dict[str, int] = {"mp3": 2}

# El encoder FLAC de ffmpeg escribe hasta 24 bits (sample_fmt s32 -> 24 bits
# reales). Un origen de 32 bits pierde profundidad: se registra, no se calla.
_MAX_FLAC_BIT_DEPTH = 24

# Sin profundidad que preservar (origen con perdida) 16 bits es la eleccion
# honesta: la precision extra de un decode a float son artefactos del codec, no
# informacion del original. Sin fijarlo, ffmpeg elige distinto por destino
# (WAV sale de 16 bits y FLAC de 24 desde el MISMO MP3).
_LOSSY_SOURCE_BIT_DEPTH = 16

# Calidad de los destinos con perdida: pocas opciones con nombre, no un campo
# numerico libre. El default entrega MAS que el 192k fijo que habia antes, asi
# que nadie recibe menos calidad que la que ya venia recibiendo.
LOSSY_QUALITY_BITRATES: dict[str, dict[str, str]] = {
    "maximum": {"mp3": "320k", "m4a": "256k"},
    "balanced": {"mp3": "192k", "m4a": "192k"},
    "compact": {"mp3": "128k", "m4a": "128k"},
}
DEFAULT_LOSSY_QUALITY = "maximum"
AUDIO_LOSSY_QUALITIES = frozenset(LOSSY_QUALITY_BITRATES)

# Extensiones cuyo contenedor determina el codec sin ambiguedad. `.m4a` queda
# AFUERA a proposito: puede traer ALAC (sin perdida) o AAC, y pasar un ALAC a
# AAC dentro del mismo .m4a es un pedido legitimo, no un trabajo vacio.
UNAMBIGUOUS_SOURCE_FORMATS: dict[str, str] = {
    ".wav": "wav",
    ".wave": "wav",
    ".flac": "flac",
    ".mp3": "mp3",
}

_LOSSLESS_CODECS = frozenset({"flac", "alac", "wavpack", "tta", "als", "mlp", "truehd"})

# Codec que produce cada destino, para saber cuando el origen YA lo trae y
# alcanza con remuxear en vez de recodificar. "wav" no figura: su unico origen
# copiable seria otro PCM, y ese caso ya lo rechaza el manager como trabajo
# vacio antes de llegar aca.
_TARGET_CODECS: dict[str, str] = {"flac": "flac", "mp3": "mp3", "m4a": "aac"}

_CODEC_DISPLAY_NAMES: dict[str, str] = {
    "aac": "AAC",
    "alac": "ALAC",
    "flac": "FLAC",
    "mp3": "MP3",
    "opus": "Opus",
    "vorbis": "Vorbis",
}


@dataclass(frozen=True, slots=True)
class SourceAudio:
    """Lo que ffprobe dice del archivo ORIGINAL, antes de tocarlo."""

    codec_name: str
    sample_rate: int
    channels: int
    # None cuando el origen tiene perdida: ahi no hay profundidad que preservar.
    bit_depth: int | None
    is_lossless: bool


@dataclass(frozen=True, slots=True)
class ConversionPlan:
    codec_args: list[str] = field(default_factory=list)
    sample_rate: int = 0
    channels: int = 0
    # None en destinos con perdida: la profundidad no es un concepto ahi.
    bit_depth: int | None = None
    bitrate: str | None = None
    resampled_from: int | None = None
    downmixed_from: int | None = None
    bit_depth_reduced_from: int | None = None
    stream_copy: bool = False


def is_conversion_only(job: AudioJob) -> bool:
    """True cuando el job no pide NINGUN procesamiento, solo el formato."""
    return not (
        job.separate
        or job.cleanup_steps
        or job.denoise
        or job.restore
        or job.voice_steps
        or job.master
    )


def unambiguous_source_format(filename: str) -> str | None:
    """El formato de salida equivalente al del archivo de entrada, o None.

    None significa "no se puede afirmar": sin extension conocida, o con una
    extension que admite mas de un codec. En ese caso la conversion se permite;
    la unica alternativa seria adivinar.
    """
    return UNAMBIGUOUS_SOURCE_FORMATS.get(Path(filename).suffix.lower())


def build_probe_command(ffprobe_path: Path, source_path: Path) -> list[str]:
    return [
        str(ffprobe_path),
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels,bits_per_raw_sample,bits_per_sample",
        "-of",
        "json",
        str(source_path),
    ]


def parse_source_audio(probe_json: dict[str, Any]) -> SourceAudio:
    streams = probe_json.get("streams") or []
    if not streams:
        raise RuntimeError(
            "El archivo no tiene ninguna pista de audio que convertir."
        )
    stream = streams[0]
    codec_name = str(stream.get("codec_name") or "unknown")
    is_lossless = _is_lossless_codec(codec_name)
    return SourceAudio(
        codec_name=codec_name,
        sample_rate=_positive_int(stream.get("sample_rate")) or 44100,
        channels=_positive_int(stream.get("channels")) or 2,
        bit_depth=_probe_bit_depth(stream) if is_lossless else None,
        is_lossless=is_lossless,
    )


def resolve_conversion_plan(
    output_format: str, lossy_quality: str, source: SourceAudio
) -> ConversionPlan:
    if output_format in LOSSY_OUTPUT_FORMATS:
        return _lossy_plan(output_format, lossy_quality, source)
    return _lossless_plan(output_format, source)


def build_conversion_command(
    ffmpeg_path: Path, source_path: Path, output_path: Path, plan: ConversionPlan
) -> list[str]:
    command = [
        str(ffmpeg_path),
        "-y",
        "-i",
        str(source_path),
        # Sin video (una caratula embebida es un stream de video) pero CON tags:
        # convertir "sin tocar nada" incluye no perder titulo ni artista.
        "-vn",
        "-map_metadata",
        "0",
        *plan.codec_args,
    ]
    if not plan.stream_copy:
        command += ["-ar", str(plan.sample_rate), "-ac", str(plan.channels)]
    command.append(str(output_path))
    return command


def conversion_metadata(
    source: SourceAudio, plan: ConversionPlan, output_format: str
) -> dict[str, Any]:
    """Lo que la conversion hizo REALMENTE, para el detalle del trabajo.

    Los avisos viajan como texto en el job y no solo en la UI: un agente por MCP
    tiene que poder enterarse de un resample forzado igual que alguien mirando
    la pantalla.
    """
    metadata: dict[str, Any] = {
        "conversionSourceFormat": codec_display_name(source.codec_name),
        "conversionTargetFormat": output_format.upper(),
        "conversionSampleRate": plan.sample_rate,
        "conversionChannels": plan.channels,
    }
    if plan.bit_depth is not None:
        metadata["conversionBitDepth"] = plan.bit_depth
    if plan.bitrate is not None:
        metadata["conversionBitrate"] = plan.bitrate
    if plan.stream_copy:
        metadata["conversionCopied"] = (
            "El audio ya venia en el codec de destino: se copio tal cual, sin "
            "recodificar."
        )
    metadata.update(_conversion_warnings(plan, output_format))
    return metadata


def codec_display_name(codec_name: str) -> str:
    if codec_name.startswith("pcm_"):
        return "WAV (PCM)"
    return _CODEC_DISPLAY_NAMES.get(codec_name, codec_name.upper())


def lossy_bitrate(output_format: str, lossy_quality: str) -> str | None:
    tier = LOSSY_QUALITY_BITRATES.get(lossy_quality, LOSSY_QUALITY_BITRATES[DEFAULT_LOSSY_QUALITY])
    return tier.get(output_format)


def _conversion_warnings(plan: ConversionPlan, output_format: str) -> dict[str, str]:
    warnings: dict[str, str] = {}
    if plan.resampled_from is not None:
        warnings["conversionResampled"] = (
            f"{output_format.upper()} no admite {plan.resampled_from} Hz: la "
            f"salida quedo a {plan.sample_rate} Hz."
        )
    if plan.downmixed_from is not None:
        warnings["conversionDownmixed"] = (
            f"{output_format.upper()} no admite {plan.downmixed_from} canales: "
            f"la salida quedo en {plan.channels}."
        )
    if plan.bit_depth_reduced_from is not None:
        warnings["conversionBitDepthReduced"] = (
            f"{output_format.upper()} escribe hasta {plan.bit_depth} bits: el "
            f"origen de {plan.bit_depth_reduced_from} bits se redujo."
        )
    return warnings


def _lossless_plan(output_format: str, source: SourceAudio) -> ConversionPlan:
    depth = _target_bit_depth(output_format, source)
    stream_copy = _can_stream_copy(output_format, source, source.sample_rate, source.channels)
    reduced_from = (
        source.bit_depth
        if not stream_copy and source.bit_depth is not None and depth < source.bit_depth
        else None
    )
    return ConversionPlan(
        codec_args=["-c:a", "copy"] if stream_copy else _lossless_codec_args(output_format, depth),
        sample_rate=source.sample_rate,
        channels=source.channels,
        bit_depth=source.bit_depth if stream_copy else depth,
        bit_depth_reduced_from=reduced_from,
        stream_copy=stream_copy,
    )


def _lossy_plan(output_format: str, lossy_quality: str, source: SourceAudio) -> ConversionPlan:
    sample_rate = _nearest_supported_rate(output_format, source.sample_rate)
    channels = min(source.channels, _MAX_OUTPUT_CHANNELS.get(output_format, source.channels))
    bitrate = lossy_bitrate(output_format, lossy_quality)
    stream_copy = _can_stream_copy(output_format, source, sample_rate, channels)
    codec_args = (
        ["-c:a", "copy"]
        if stream_copy
        else ["-c:a", _lossy_encoder(output_format), "-b:a", str(bitrate)]
    )
    return ConversionPlan(
        codec_args=codec_args,
        sample_rate=sample_rate,
        channels=channels,
        bitrate=None if stream_copy else bitrate,
        resampled_from=source.sample_rate if sample_rate != source.sample_rate else None,
        downmixed_from=source.channels if channels != source.channels else None,
        stream_copy=stream_copy,
    )


def _can_stream_copy(
    output_format: str, source: SourceAudio, sample_rate: int, channels: int
) -> bool:
    """Remux en vez de recodificar cuando el origen YA trae el codec destino.

    Solo alcanzable con `.m4a` (el manager rechaza antes wav->wav, flac->flac y
    mp3->mp3 como trabajo vacio). Sin esto, alguien que pide "m4a" sobre un AAC
    se llevaria una recodificacion con perdida encima de otra con perdida.
    """
    if _TARGET_CODECS.get(output_format) != source.codec_name:
        return False
    return sample_rate == source.sample_rate and channels == source.channels


def _lossy_encoder(output_format: str) -> str:
    return "libmp3lame" if output_format == "mp3" else "aac"


def _target_bit_depth(output_format: str, source: SourceAudio) -> int:
    if source.bit_depth is None:
        return _LOSSY_SOURCE_BIT_DEPTH
    if output_format == "flac":
        return _MAX_FLAC_BIT_DEPTH if source.bit_depth > _MAX_FLAC_BIT_DEPTH else source.bit_depth
    return source.bit_depth


def _lossless_codec_args(output_format: str, depth: int) -> list[str]:
    return _wav_codec_args(depth) if output_format == "wav" else _flac_codec_args(depth)


def _wav_codec_args(depth: int) -> list[str]:
    if depth <= 16:
        return ["-c:a", "pcm_s16le"]
    if depth <= 24:
        return ["-c:a", "pcm_s24le"]
    return ["-c:a", "pcm_s32le"]


def _flac_codec_args(depth: int) -> list[str]:
    # s16 escribe 16 bits y s32 escribe 24 (verificado con el ffmpeg del repo).
    return ["-c:a", "flac", "-sample_fmt", "s16" if depth <= 16 else "s32"]


def _nearest_supported_rate(output_format: str, sample_rate: int) -> int:
    supported = _LOSSY_SAMPLE_RATES.get(output_format)
    if supported is None or sample_rate in supported:
        return sample_rate
    # Empate a favor de la tasa MAS ALTA: ante dos candidatos igual de lejos,
    # el que conserva mas ancho de banda.
    return min(supported, key=lambda candidate: (abs(candidate - sample_rate), -candidate))


def _is_lossless_codec(codec_name: str) -> bool:
    return codec_name.startswith("pcm_") or codec_name in _LOSSLESS_CODECS


def _probe_bit_depth(stream: dict[str, Any]) -> int | None:
    # FLAC reporta la profundidad en bits_per_raw_sample y deja bits_per_sample
    # en 0; WAV hace lo contrario segun el codec. Se toma el primero positivo.
    for key in ("bits_per_raw_sample", "bits_per_sample"):
        depth = _positive_int(stream.get(key))
        if depth is not None:
            return depth
    return None


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
