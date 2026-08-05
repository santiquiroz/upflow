from __future__ import annotations

from pathlib import Path

import pytest

from app.services.media_decode import build_decode_to_wav_command, needs_decoding

# ---------------------------------------------------------------------------
# El motor de transcripcion lee el audio con `soundfile.read()`, que entiende
# WAV/FLAC/OGG/MP3 pero NO contenedores de video. Hoy la ruta acepta un .mp4 sin
# chistar, lo encola, toma el semaforo del dispositivo y recien ahi revienta con
# un error crudo de libsndfile.
#
# ffmpeg ya viaja con la app y `audio_pipeline._decode_to_wav` ya prueba este
# camino para el modulo de audio: se reusa el mismo patron.
# ---------------------------------------------------------------------------


class TestNeedsDecoding:
    @pytest.mark.parametrize("name", ["clip.mp4", "clip.MKV", "clip.mov", "clip.webm", "clip.m4a", "clip.aac"])
    def test_a_container_soundfile_cannot_read_needs_ffmpeg(self, name: str) -> None:
        assert needs_decoding(Path(name)) is True

    @pytest.mark.parametrize("name", ["voz.wav", "voz.FLAC", "voz.ogg", "voz.mp3"])
    def test_a_format_soundfile_reads_is_left_alone(self, name: str) -> None:
        # Decodificar lo que ya se puede leer seria una copia de mas por archivo.
        assert needs_decoding(Path(name)) is False

    def test_an_unknown_extension_is_decoded_rather_than_refused(self) -> None:
        # Mandarlo a ffmpeg da una chance de que ande; rechazarlo de plano no.
        assert needs_decoding(Path("grabacion.sinextension")) is True


class TestDecodeCommand:
    def test_drops_the_video_stream_and_targets_the_rate_whisper_wants(self) -> None:
        command = build_decode_to_wav_command(
            ffmpeg="ffmpeg.exe",
            source=Path("in.mp4"),
            destination=Path("out.wav"),
            sample_rate=16000,
        )
        assert command[0] == "ffmpeg.exe"
        assert "-vn" in command
        assert command[command.index("-ar") + 1] == "16000"
        assert command[-1] == "out.wav"

    def test_forces_mono_because_the_model_takes_one_channel(self) -> None:
        command = build_decode_to_wav_command(
            ffmpeg="ffmpeg.exe",
            source=Path("in.mkv"),
            destination=Path("out.wav"),
            sample_rate=16000,
        )
        assert command[command.index("-ac") + 1] == "1"

    def test_overwrites_so_a_retry_does_not_hang_on_a_prompt(self) -> None:
        command = build_decode_to_wav_command(
            ffmpeg="ffmpeg.exe",
            source=Path("in.mp4"),
            destination=Path("out.wav"),
            sample_rate=16000,
        )
        assert "-y" in command
