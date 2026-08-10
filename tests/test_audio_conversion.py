from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from app.config import Settings
from app.models import AudioJob
from app.services.audio_conversion import (
    SourceAudio,
    build_conversion_command,
    conversion_metadata,
    is_conversion_only,
    parse_source_audio,
    resolve_conversion_plan,
    unambiguous_source_format,
)
from app.services.audio_job_manager import AudioJobManager
from app.services.audio_pipeline import AudioPipeline
from app.services.device_semaphores import DeviceSemaphores
from app.services.progress import build_audio_stages
from app.services.storage import StorageService

# ---------------------------------------------------------------------------
# Conversion directa: un job SIN ningun paso de procesamiento convierte de
# formato en UNA pasada de ffmpeg, conservando tasa de muestreo y profundidad
# de bits hasta donde el destino lo permita. El camino de PROCESAMIENTO no
# cambia: sigue decodificando a 48 kHz / 16 bits porque sus motores lo exigen
# (eso lo cubre test_audio_pipeline.py).
# ---------------------------------------------------------------------------


def make_settings(tmp_path: Path) -> Settings:
    settings = Settings(RUNTIME_DIR=str(tmp_path))
    StorageService(settings)
    return settings


def flac_24bit_44100() -> SourceAudio:
    return SourceAudio(
        codec_name="flac", sample_rate=44100, channels=2, bit_depth=24, is_lossless=True
    )


def mp3_source(sample_rate: int = 44100, channels: int = 2) -> SourceAudio:
    return SourceAudio(
        codec_name="mp3",
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=None,
        is_lossless=False,
    )


# ---------------------------------------------------------------------------
# is_conversion_only -- que cuenta como "no se pidio ningun paso"
# ---------------------------------------------------------------------------


def test_a_job_without_any_step_is_conversion_only() -> None:
    job = AudioJob(source_path=Path("song.flac"), original_filename="song.flac")

    assert is_conversion_only(job) is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"denoise": "rnnoise"},
        {"restore": "apollo"},
        {"master": "streaming"},
        {"voice_steps": ["deesser"]},
        {"cleanup_steps": ["denoise"]},
        {"separate": True},
    ],
)
def test_any_requested_step_takes_the_job_out_of_the_conversion_path(
    overrides: dict[str, object],
) -> None:
    job = AudioJob(
        source_path=Path("song.flac"), original_filename="song.flac", **overrides
    )

    assert is_conversion_only(job) is False


# ---------------------------------------------------------------------------
# Plan: destinos SIN perdida conservan tasa y profundidad
# ---------------------------------------------------------------------------


def test_flac_to_wav_keeps_sample_rate_and_bit_depth() -> None:
    plan = resolve_conversion_plan("wav", "maximum", flac_24bit_44100())

    assert plan.sample_rate == 44100
    assert plan.bit_depth == 24
    assert plan.codec_args == ["-c:a", "pcm_s24le"]
    assert plan.resampled_from is None


def test_16_bit_source_to_wav_stays_16_bit() -> None:
    source = SourceAudio("flac", 44100, 2, 16, True)

    plan = resolve_conversion_plan("wav", "maximum", source)

    assert plan.codec_args == ["-c:a", "pcm_s16le"]
    assert plan.bit_depth == 16


def test_24_bit_source_to_flac_uses_the_sample_format_that_writes_24_bits() -> None:
    # Origen WAV y no FLAC: un flac->flac se resuelve copiando el stream, que
    # no ejercita el mapeo de profundidad a sample_fmt.
    source = SourceAudio("pcm_s24le", 44100, 2, 24, True)

    plan = resolve_conversion_plan("flac", "maximum", source)

    # s32 es como el encoder FLAC de ffmpeg escribe 24 bits reales (verificado
    # con el binario vendorizado); s16 escribiria 16.
    assert plan.codec_args == ["-c:a", "flac", "-sample_fmt", "s32"]
    assert plan.bit_depth == 24


def test_high_sample_rate_survives_a_lossless_target() -> None:
    source = SourceAudio("flac", 96000, 2, 24, True)

    plan = resolve_conversion_plan("wav", "maximum", source)

    assert plan.sample_rate == 96000
    assert plan.resampled_from is None


def test_32_bit_source_to_flac_records_the_depth_it_could_not_keep() -> None:
    source = SourceAudio("pcm_s32le", 44100, 2, 32, True)

    plan = resolve_conversion_plan("flac", "maximum", source)

    assert plan.bit_depth == 24
    assert plan.bit_depth_reduced_from == 32


def test_a_lossy_source_to_a_lossless_target_settles_on_16_bits() -> None:
    # Sin profundidad de origen que preservar, los dos destinos sin perdida
    # tienen que coincidir: sin fijarlo, ffmpeg escribe WAV de 16 y FLAC de 24
    # desde el MISMO mp3.
    wav_plan = resolve_conversion_plan("wav", "maximum", mp3_source())
    flac_plan = resolve_conversion_plan("flac", "maximum", mp3_source())

    assert wav_plan.bit_depth == 16
    assert flac_plan.bit_depth == 16
    assert flac_plan.codec_args == ["-c:a", "flac", "-sample_fmt", "s16"]
    assert wav_plan.bit_depth_reduced_from is None


# ---------------------------------------------------------------------------
# Plan: destinos CON perdida -- tasa soportada, resample registrado
# ---------------------------------------------------------------------------


def test_mp3_keeps_a_sample_rate_its_codec_supports() -> None:
    plan = resolve_conversion_plan("mp3", "maximum", flac_24bit_44100())

    assert plan.sample_rate == 44100
    assert plan.resampled_from is None
    assert plan.bit_depth is None


def test_96k_to_mp3_resamples_to_the_nearest_supported_rate_and_says_so() -> None:
    source = SourceAudio("flac", 96000, 2, 24, True)

    plan = resolve_conversion_plan("mp3", "maximum", source)

    assert plan.sample_rate == 48000
    assert plan.resampled_from == 96000
    metadata = conversion_metadata(source, plan, "mp3")
    assert "96000" in metadata["conversionResampled"]
    assert "48000" in metadata["conversionResampled"]


def test_96k_survives_m4a_because_aac_supports_it() -> None:
    source = SourceAudio("flac", 96000, 2, 24, True)

    plan = resolve_conversion_plan("m4a", "maximum", source)

    assert plan.sample_rate == 96000
    assert plan.resampled_from is None
    assert "conversionResampled" not in conversion_metadata(source, plan, "m4a")


def test_a_surround_source_to_mp3_downmixes_and_says_so() -> None:
    source = SourceAudio("flac", 48000, 6, 24, True)

    plan = resolve_conversion_plan("mp3", "maximum", source)

    assert plan.channels == 2
    assert plan.downmixed_from == 6
    assert "6" in conversion_metadata(source, plan, "mp3")["conversionDownmixed"]


def test_a_surround_source_keeps_its_channels_on_m4a() -> None:
    source = SourceAudio("flac", 48000, 6, 24, True)

    plan = resolve_conversion_plan("m4a", "maximum", source)

    assert plan.channels == 6
    assert plan.downmixed_from is None


# ---------------------------------------------------------------------------
# Calidad de los destinos con perdida
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("quality", "mp3_bitrate", "m4a_bitrate"),
    [("maximum", "320k", "256k"), ("balanced", "192k", "192k"), ("compact", "128k", "128k")],
)
def test_each_quality_tier_maps_to_its_bitrate(
    quality: str, mp3_bitrate: str, m4a_bitrate: str
) -> None:
    mp3_plan = resolve_conversion_plan("mp3", quality, flac_24bit_44100())
    m4a_plan = resolve_conversion_plan("m4a", quality, flac_24bit_44100())

    assert mp3_plan.codec_args == ["-c:a", "libmp3lame", "-b:a", mp3_bitrate]
    assert m4a_plan.codec_args == ["-c:a", "aac", "-b:a", m4a_bitrate]


def test_the_default_quality_is_never_below_the_fixed_192k_it_replaced() -> None:
    plan = resolve_conversion_plan("mp3", "maximum", flac_24bit_44100())

    assert plan.bitrate == "320k"


def test_an_unknown_quality_falls_back_to_the_default_instead_of_crashing() -> None:
    # El manager valida antes; esto cubre llamadas directas al planificador.
    plan = resolve_conversion_plan("mp3", "no-existe", flac_24bit_44100())

    assert plan.bitrate == "320k"


# ---------------------------------------------------------------------------
# Remux en vez de recodificar cuando el origen YA trae el codec destino
# ---------------------------------------------------------------------------


def test_an_aac_source_to_m4a_is_copied_instead_of_re_encoded() -> None:
    source = SourceAudio("aac", 44100, 2, None, False)

    plan = resolve_conversion_plan("m4a", "maximum", source)

    assert plan.stream_copy is True
    assert plan.codec_args == ["-c:a", "copy"]
    assert "conversionCopied" in conversion_metadata(source, plan, "m4a")


def test_an_alac_source_to_m4a_is_a_real_conversion_not_a_copy() -> None:
    source = SourceAudio("alac", 44100, 2, 24, True)

    plan = resolve_conversion_plan("m4a", "maximum", source)

    assert plan.stream_copy is False
    assert plan.codec_args == ["-c:a", "aac", "-b:a", "256k"]


def test_a_copy_skips_the_rate_and_channel_flags_that_would_force_a_decode() -> None:
    source = SourceAudio("aac", 44100, 2, None, False)
    plan = resolve_conversion_plan("m4a", "maximum", source)

    command = build_conversion_command(
        Path("ffmpeg"), Path("in.m4a"), Path("out.m4a"), plan
    )

    assert "-ar" not in command
    assert "-ac" not in command


# ---------------------------------------------------------------------------
# El comando
# ---------------------------------------------------------------------------


def test_the_command_is_a_single_pass_from_the_original_file() -> None:
    plan = resolve_conversion_plan("mp3", "maximum", flac_24bit_44100())

    command = build_conversion_command(
        Path("ffmpeg.exe"), Path("song.flac"), Path("out.mp3"), plan
    )

    assert command[:5] == ["ffmpeg.exe", "-y", "-i", "song.flac", "-vn"]
    assert command[-1] == "out.mp3"
    # Tags conservados: convertir "sin tocar nada" incluye no perder el titulo.
    assert command[command.index("-map_metadata") + 1] == "0"
    assert command[command.index("-ar") + 1] == "44100"


# ---------------------------------------------------------------------------
# Lectura del ffprobe
# ---------------------------------------------------------------------------


def test_flac_reports_its_depth_through_bits_per_raw_sample() -> None:
    probe = {
        "streams": [
            {
                "codec_name": "flac",
                "sample_rate": "44100",
                "channels": 2,
                "bits_per_sample": 0,
                "bits_per_raw_sample": "24",
            }
        ]
    }

    source = parse_source_audio(probe)

    assert source.bit_depth == 24
    assert source.is_lossless is True


def test_wav_reports_its_depth_through_bits_per_sample() -> None:
    probe = {
        "streams": [
            {"codec_name": "pcm_s16le", "sample_rate": "48000", "channels": 2, "bits_per_sample": 16}
        ]
    }

    source = parse_source_audio(probe)

    assert source.bit_depth == 16
    assert source.is_lossless is True


def test_a_lossy_codec_has_no_bit_depth_even_if_the_probe_reports_one() -> None:
    probe = {
        "streams": [
            {"codec_name": "mp3", "sample_rate": "44100", "channels": 2, "bits_per_sample": 16}
        ]
    }

    source = parse_source_audio(probe)

    assert source.bit_depth is None
    assert source.is_lossless is False


def test_a_file_without_audio_streams_fails_with_a_readable_message() -> None:
    with pytest.raises(RuntimeError, match="pista de audio"):
        parse_source_audio({"streams": []})


# ---------------------------------------------------------------------------
# Deteccion del formato de origen por extension
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("song.flac", "flac"),
        ("SONG.FLAC", "flac"),
        ("song.wav", "wav"),
        ("song.mp3", "mp3"),
        # Ambiguas o desconocidas: no se puede afirmar el codec del contenedor.
        ("song.m4a", None),
        ("song.opus", None),
        ("song", None),
    ],
)
def test_only_unambiguous_extensions_claim_a_source_format(
    filename: str, expected: str | None
) -> None:
    assert unambiguous_source_format(filename) == expected


# ---------------------------------------------------------------------------
# Validacion en el manager
# ---------------------------------------------------------------------------


def make_manager(settings: Settings) -> AudioJobManager:
    pipeline = AudioPipeline(settings, {}, {})
    return AudioJobManager(settings, pipeline, DeviceSemaphores(settings))


def write_upload(settings: Settings, name: str = "song.flac") -> Path:
    source_path = settings.uploads_path / name
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"fake-audio-bytes")
    return source_path


async def test_a_job_with_no_steps_but_a_different_format_is_accepted(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)

    job = await manager.create_job(
        source_path=write_upload(settings),
        original_filename="song.flac",
        output_format="mp3",
    )

    assert is_conversion_only(job) is True
    assert job.output_format == "mp3"


async def test_a_job_with_no_steps_and_the_same_format_is_rejected(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)

    with pytest.raises(ValueError, match="ya esta en FLAC"):
        await manager.create_job(
            source_path=write_upload(settings),
            original_filename="song.flac",
            output_format="flac",
        )


async def test_the_same_format_is_fine_as_long_as_a_step_was_requested(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)

    job = await manager.create_job(
        source_path=write_upload(settings),
        original_filename="song.flac",
        output_format="flac",
        master="streaming",
    )

    assert job.master == "streaming"


async def test_m4a_to_m4a_is_allowed_because_the_container_may_hold_alac(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)

    job = await manager.create_job(
        source_path=write_upload(settings, "song.m4a"),
        original_filename="song.m4a",
        output_format="m4a",
    )

    assert job.output_format == "m4a"


async def test_m4a_is_a_valid_output_format(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)

    job = await manager.create_job(
        source_path=write_upload(settings),
        original_filename="song.flac",
        output_format="m4a",
    )

    assert job.output_format == "m4a"


async def test_an_unknown_output_format_is_rejected(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)

    with pytest.raises(ValueError, match="output_format must be one of"):
        await manager.create_job(
            source_path=write_upload(settings),
            original_filename="song.flac",
            output_format="aiff",
        )


async def test_an_unknown_quality_tier_is_rejected(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)

    with pytest.raises(ValueError, match="lossy_quality must be one of"):
        await manager.create_job(
            source_path=write_upload(settings),
            original_filename="song.flac",
            output_format="mp3",
            lossy_quality="insane",
        )


async def test_the_default_quality_tier_is_maximum(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    manager = make_manager(settings)

    job = await manager.create_job(
        source_path=write_upload(settings),
        original_filename="song.flac",
        output_format="mp3",
    )

    assert job.lossy_quality == "maximum"


# ---------------------------------------------------------------------------
# Etapas: la conversion tiene la suya, y NO reusa decoding/finalizing
# ---------------------------------------------------------------------------


def test_a_conversion_only_job_shows_one_stage_of_its_own() -> None:
    job = AudioJob(source_path=Path("song.flac"), original_filename="song.flac")

    stages = build_audio_stages(job)

    assert [stage.key for stage in stages] == ["converting"]


def test_a_processing_job_never_shows_the_conversion_stage() -> None:
    job = AudioJob(
        source_path=Path("song.flac"), original_filename="song.flac", denoise="rnnoise"
    )

    keys = [stage.key for stage in build_audio_stages(job)]

    assert "converting" not in keys
    assert keys == ["decoding", "denoising", "finalizing"]


# ---------------------------------------------------------------------------
# Pipeline: una sola pasada, sin decode a 48 kHz
# ---------------------------------------------------------------------------


class RecordingPipeline(AudioPipeline):
    """Sin binarios reales: registra los comandos y devuelve un probe fijo."""

    def __init__(self, *args: object, source: SourceAudio | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.commands: list[list[str]] = []
        self.source = source or flac_24bit_44100()

    async def _run_process(self, command: list[str], failure_message: str) -> None:
        self.commands.append(list(command))
        output_path = Path(command[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-converted-audio")

    async def _probe_source(self, source_path: Path) -> SourceAudio:
        return self.source


async def test_the_conversion_path_runs_exactly_one_ffmpeg_pass(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = RecordingPipeline(settings, {}, {})
    job = AudioJob(
        source_path=write_upload(settings),
        original_filename="song.flac",
        output_format="wav",
    )

    output_path = await pipeline.run(job)

    assert len(pipeline.commands) == 1
    assert output_path.suffix == ".wav"


async def test_the_conversion_path_never_decodes_to_the_engine_sample_rate(
    tmp_path: Path,
) -> None:
    # Este es el bug que la conversion existe para evitar: el decode del camino
    # de procesamiento fija 48000 y pcm_s16le, y aplicarselo a un FLAC 44.1/24
    # le cambiaria la tasa y le truncaria la profundidad en silencio.
    settings = make_settings(tmp_path)
    pipeline = RecordingPipeline(settings, {}, {})
    job = AudioJob(
        source_path=write_upload(settings),
        original_filename="song.flac",
        output_format="wav",
    )

    await pipeline.run(job)

    command = pipeline.commands[0]
    assert "48000" not in command
    assert command[command.index("-ar") + 1] == "44100"
    assert command[command.index("-c:a") + 1] == "pcm_s24le"


async def test_the_conversion_records_what_it_produced_in_the_job(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = RecordingPipeline(settings, {}, {})
    job = AudioJob(
        source_path=write_upload(settings),
        original_filename="song.flac",
        output_format="mp3",
    )

    await pipeline.run(job)

    assert job.metadata["conversionSourceFormat"] == "FLAC"
    assert job.metadata["conversionTargetFormat"] == "MP3"
    assert job.metadata["conversionSampleRate"] == 44100
    assert job.metadata["conversionBitrate"] == "320k"
    assert job.metadata["stage"] == "completed"
    assert job.metadata["progress"] == 1.0


async def test_a_forced_resample_reaches_the_job_metadata(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = RecordingPipeline(
        settings, {}, {}, source=SourceAudio("flac", 96000, 2, 24, True)
    )
    job = AudioJob(
        source_path=write_upload(settings),
        original_filename="song.flac",
        output_format="mp3",
    )

    await pipeline.run(job)

    assert "conversionResampled" in job.metadata
    assert job.metadata["conversionSampleRate"] == 48000


# ---------------------------------------------------------------------------
# Smoke con ffmpeg REAL: lo unico que prueba que la tasa y la profundidad
# sobreviven de verdad. Se saltea sin los binarios vendorizados.
# ---------------------------------------------------------------------------

_FFMPEG = Settings(RUNTIME_DIR="runtime").ffmpeg_binary_path
_FFPROBE = Settings(RUNTIME_DIR="runtime").ffprobe_binary_path

needs_ffmpeg = pytest.mark.skipif(
    not (_FFMPEG.exists() and _FFPROBE.exists()),
    reason="vendored ffmpeg/ffprobe not present",
)


def write_real_flac(path: Path, sample_rate: int = 44100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(_FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={sample_rate}:duration=1",
            "-c:a", "flac", "-sample_fmt", "s32", "-bits_per_raw_sample", "24", str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def write_real_wav_24bit(path: Path, sample_rate: int = 44100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(_FFMPEG), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={sample_rate}:duration=1",
            "-c:a", "pcm_s24le", str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def probe_stream(path: Path) -> dict:
    completed = subprocess.run(
        [
            str(_FFPROBE), "-v", "error", "-select_streams", "a:0",
            "-show_entries",
            "stream=codec_name,sample_rate,channels,bits_per_raw_sample,bits_per_sample",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)["streams"][0]


@needs_ffmpeg
async def test_real_flac_44100_24bit_to_wav_keeps_both(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = AudioPipeline(settings, {}, {})
    source_path = write_real_flac(settings.uploads_path / "song.flac")
    job = AudioJob(
        source_path=source_path, original_filename="song.flac", output_format="wav"
    )

    output_path = await pipeline.run(job)

    stream = probe_stream(output_path)
    assert stream["sample_rate"] == "44100"
    assert stream["codec_name"] == "pcm_s24le"
    assert int(stream["bits_per_sample"]) == 24


@needs_ffmpeg
async def test_real_flac_44100_to_mp3_does_not_end_up_at_48k(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = AudioPipeline(settings, {}, {})
    source_path = write_real_flac(settings.uploads_path / "song.flac")
    job = AudioJob(
        source_path=source_path, original_filename="song.flac", output_format="mp3"
    )

    output_path = await pipeline.run(job)

    stream = probe_stream(output_path)
    assert stream["sample_rate"] == "44100"
    assert stream["codec_name"] == "mp3"


@needs_ffmpeg
async def test_real_96k_flac_to_mp3_resamples_and_records_it(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = AudioPipeline(settings, {}, {})
    source_path = write_real_flac(settings.uploads_path / "hires.flac", sample_rate=96000)
    job = AudioJob(
        source_path=source_path, original_filename="hires.flac", output_format="mp3"
    )

    output_path = await pipeline.run(job)

    assert probe_stream(output_path)["sample_rate"] == "48000"
    assert "96000" in job.metadata["conversionResampled"]


@needs_ffmpeg
async def test_real_wav_24bit_to_flac_keeps_24_bits(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = AudioPipeline(settings, {}, {})
    source_path = write_real_wav_24bit(settings.uploads_path / "song.wav")
    job = AudioJob(
        source_path=source_path, original_filename="song.wav", output_format="flac"
    )

    output_path = await pipeline.run(job)

    stream = probe_stream(output_path)
    assert stream["codec_name"] == "flac"
    assert stream["sample_rate"] == "44100"
    assert int(stream["bits_per_raw_sample"]) == 24


@needs_ffmpeg
async def test_real_flac_to_m4a_produces_playable_aac(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pipeline = AudioPipeline(settings, {}, {})
    source_path = write_real_flac(settings.uploads_path / "song.flac")
    job = AudioJob(
        source_path=source_path, original_filename="song.flac", output_format="m4a"
    )

    output_path = await pipeline.run(job)

    stream = probe_stream(output_path)
    assert stream["codec_name"] == "aac"
    assert stream["sample_rate"] == "44100"


# ---------------------------------------------------------------------------
# Capabilities: formatos y escalones viajan como DATO, no como docstring. Un
# agente por MCP tiene que poder descubrirlos sin leer la documentacion.
# ---------------------------------------------------------------------------


async def test_capabilities_expose_the_output_formats_and_quality_tiers(
    tmp_path: Path,
) -> None:
    from app.api.routes import audio_capabilities

    capabilities = await audio_capabilities(settings=make_settings(tmp_path))

    assert capabilities.output_formats == ["flac", "m4a", "mp3", "wav"]
    assert capabilities.lossy_formats == ["m4a", "mp3"]
    assert capabilities.default_lossy_quality == "maximum"
    # En orden de calidad DESCENDENTE, no alfabetico: la UI lo pinta tal cual.
    assert [tier.id for tier in capabilities.lossy_qualities] == [
        "maximum",
        "balanced",
        "compact",
    ]
    assert capabilities.lossy_qualities[0].bitrates == {"mp3": "320k", "m4a": "256k"}


async def test_the_api_rejects_a_conversion_with_nothing_to_convert(tmp_path: Path) -> None:
    import io

    from fastapi import HTTPException
    from starlette.datastructures import UploadFile

    from app.api.routes import create_audio_job

    settings = make_settings(tmp_path)
    manager = make_manager(settings)
    upload = UploadFile(filename="song.flac", file=io.BytesIO(b"flac-bytes"))

    with pytest.raises(HTTPException) as excinfo:
        await create_audio_job(
            request=None,  # type: ignore[arg-type]
            file=upload,
            denoise=None,
            restore=None,
            device=None,
            output_format="flac",
            audio_jobs=manager,
            storage=StorageService(settings),
            settings=settings,
        )

    assert excinfo.value.status_code == 400
    assert "ya esta en FLAC" in excinfo.value.detail


async def test_the_api_accepts_a_conversion_with_no_steps(tmp_path: Path) -> None:
    import io

    from starlette.datastructures import UploadFile

    from app.api.routes import create_audio_job

    settings = make_settings(tmp_path)
    manager = make_manager(settings)
    upload = UploadFile(filename="song.flac", file=io.BytesIO(b"flac-bytes"))

    response = await create_audio_job(
        request=None,  # type: ignore[arg-type]
        file=upload,
        denoise=None,
        restore=None,
        device=None,
        output_format="m4a",
        lossy_quality="balanced",
        audio_jobs=manager,
        storage=StorageService(settings),
        settings=settings,
    )

    job = manager.get_job(response.job_id)
    assert job is not None
    assert job.output_format == "m4a"
    assert job.lossy_quality == "balanced"
    assert is_conversion_only(job) is True
