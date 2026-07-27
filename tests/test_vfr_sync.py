from __future__ import annotations

from pathlib import Path

import pytest

from app.services.engines.ffmpeg_frame_source import FfmpegFrameSource


# ---------------------------------------------------------------------------
# Sincronía en fuentes VFR. Un BDrip de anime con cadencia mixta (tramos a
# 23.976 y a 29.97) declara avg_frame_rate=29.97 pero tiene MENOS frames de los
# que esa tasa implica. Decodificar con -fps_mode passthrough conserva esos
# frames reales, pero después se encodea a fps CONSTANTE derivado del valor
# declarado: el video sale más corto que el audio y los subtítulos, y la deriva
# crece hasta minutos. Medido sobre material real: 39.925 frames reales contra
# 45.195 que implica la tasa declarada, o sea 177 s de desfase en 25 minutos.
#
# El arreglo es normalizar a CFR en el decode: ffmpeg duplica/descarta según los
# timestamps reales y a partir de ahí frames == duración × fps por construcción.
# ---------------------------------------------------------------------------


def _flag_value(command: list[str], flag: str) -> str | None:
    return command[command.index(flag) + 1] if flag in command else None


def make_source(fps: str = "30000/1001") -> FfmpegFrameSource:
    return FfmpegFrameSource(
        Path("ffmpeg.exe"), Path("clip.mkv"), 960, 720, decode_threads=2, fps=fps
    )


def test_stream_source_normalises_to_cfr_at_the_declared_rate() -> None:
    command = make_source().build_command()

    assert _flag_value(command, "-fps_mode") == "cfr", "passthrough deja pasar la cadencia VFR"
    assert _flag_value(command, "-r") == "30000/1001"


def test_stream_source_never_uses_the_deprecated_vsync_flag() -> None:
    assert "-vsync" not in make_source().build_command()


def test_stream_source_keeps_decode_threads_before_the_input() -> None:
    # -threads después de -i configuraría el encoder de salida, no el decoder.
    command = make_source().build_command()
    assert command.index("-threads") < command.index("-i")


def test_stream_source_still_emits_rawvideo_rgb24_to_stdout() -> None:
    command = make_source().build_command()
    assert _flag_value(command, "-f") == "rawvideo"
    assert _flag_value(command, "-pix_fmt") == "rgb24"
    assert command[-1] == "pipe:1"


@pytest.mark.parametrize("fps", ["24000/1001", "30000/1001", "25/1", "60/1"])
def test_stream_source_passes_through_any_valid_rate(fps: str) -> None:
    assert _flag_value(make_source(fps).build_command(), "-r") == fps


def test_png_extraction_normalises_to_cfr(tmp_path: Path) -> None:
    # El camino clásico tiene el MISMO problema: extrae passthrough y luego
    # encodea a fps constante.
    from tests.test_video_upscaler import make_stream_upscaler

    upscaler = make_stream_upscaler(tmp_path)
    command = upscaler._build_extract_frames_command(
        Path("src.mkv"), tmp_path / "frames-in", "30000/1001"
    )

    assert _flag_value(command, "-fps_mode") == "cfr"
    assert _flag_value(command, "-r") == "30000/1001"
    assert "-vsync" not in command
    assert command[-1].endswith("%08d.png")
