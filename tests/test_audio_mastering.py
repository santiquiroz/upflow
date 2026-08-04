from __future__ import annotations

from pathlib import Path

import pytest

from app.services.audio_mastering import (
    MASTERING_PRESETS,
    LoudnessMeasurement,
    build_master_command,
    build_measure_command,
    mastering_preset,
    parse_loudness_measurement,
)

# ---------------------------------------------------------------------------
# Acabado profesional: normalización de sonoridad a EBU R128, que es el estándar
# real de entrega (streaming -14 LUFS, broadcast -23 LUFS). Se hace en DOS
# pasadas — medir y después aplicar con esa medición — porque la de una sola
# pasada es adaptativa y bombea. Todo con ffmpeg, que ya viene con la app.
# ---------------------------------------------------------------------------

SALIDA_REAL = """
[Parsed_loudnorm_0 @ 000001]
{
	"input_i" : "-27.61",
	"input_tp" : "-9.20",
	"input_lra" : "5.30",
	"input_thresh" : "-37.94",
	"output_i" : "-14.02",
	"output_tp" : "-1.49",
	"output_lra" : "5.20",
	"output_thresh" : "-24.35",
	"normalization_type" : "dynamic",
	"target_offset" : "0.02"
}
"""


def test_every_preset_targets_a_real_standard() -> None:
    por_id = {p.id: p for p in MASTERING_PRESETS}
    # -14 LUFS es lo que piden Spotify/YouTube; -23 es EBU R128 de broadcast.
    assert por_id["streaming"].target_lufs == -14.0
    assert por_id["broadcast"].target_lufs == -23.0
    for preset in MASTERING_PRESETS:
        assert preset.true_peak <= -1.0, "un pico real sobre -1 dBTP clipea al recodificar"
        assert preset.label_key and preset.description_key


def test_an_unknown_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="streaming"):
        mastering_preset("no-existe")


# --- primera pasada: medir ------------------------------------------------


def test_the_measure_pass_asks_for_json_and_writes_no_audio(tmp_path: Path) -> None:
    cmd = build_measure_command(Path("ffmpeg.exe"), tmp_path / "in.wav", "streaming")
    unidos = " ".join(cmd)
    assert "print_format=json" in unidos
    # Medir no puede producir un archivo: si escribiera, pagaríamos el encode dos veces.
    assert "-f" in cmd and "null" in cmd


def test_the_measure_pass_uses_the_targets_of_the_preset(tmp_path: Path) -> None:
    cmd = build_measure_command(Path("ffmpeg.exe"), tmp_path / "in.wav", "broadcast")
    unidos = " ".join(cmd)
    assert "I=-23.0" in unidos
    assert "TP=-1.0" in unidos


def test_reading_the_measurement(tmp_path: Path) -> None:
    medida = parse_loudness_measurement(SALIDA_REAL)
    assert medida is not None
    assert medida.input_i == -27.61
    assert medida.input_tp == -9.20
    assert medida.target_offset == 0.02


def test_noise_around_the_json_does_not_break_the_reading() -> None:
    ruido = "algo antes\n" + SALIDA_REAL + "\nalgo despues"
    assert parse_loudness_measurement(ruido) is not None


def test_a_missing_measurement_is_none_not_a_crash() -> None:
    """Si ffmpeg no imprimió el JSON se sigue sin masterizar, no se rompe el trabajo."""
    assert parse_loudness_measurement("ffmpeg dijo cualquier otra cosa") is None


# --- segunda pasada: aplicar ----------------------------------------------


def medida() -> LoudnessMeasurement:
    return LoudnessMeasurement(
        input_i=-27.61, input_tp=-9.20, input_lra=5.30, input_thresh=-37.94, target_offset=0.02
    )


def test_the_second_pass_feeds_back_what_was_measured(tmp_path: Path) -> None:
    """Sin measured_*, loudnorm corre en modo adaptativo de una pasada y bombea."""
    cmd = build_master_command(
        Path("ffmpeg.exe"), tmp_path / "in.wav", tmp_path / "out.wav", "streaming", medida()
    )
    unidos = " ".join(cmd)
    assert "measured_I=-27.61" in unidos
    assert "measured_TP=-9.2" in unidos
    assert "measured_LRA=5.3" in unidos
    assert "measured_thresh=-37.94" in unidos
    assert "offset=0.02" in unidos
    assert "linear=true" in unidos


def test_the_voice_preset_adds_de_essing_and_gentle_compression(tmp_path: Path) -> None:
    cmd = build_master_command(
        Path("ffmpeg.exe"), tmp_path / "in.wav", tmp_path / "out.wav", "voice", medida()
    )
    unidos = " ".join(cmd)
    assert "deesser" in unidos
    assert "acompressor" in unidos
    # El orden importa: se de-esea y comprime ANTES de fijar la sonoridad final.
    assert unidos.index("deesser") < unidos.index("loudnorm")


def test_the_music_presets_do_not_touch_the_dynamics(tmp_path: Path) -> None:
    """Comprimir música que no lo pidió es exactamente lo que arruina un master."""
    for preset in ("streaming", "broadcast"):
        cmd = build_master_command(
            Path("ffmpeg.exe"), tmp_path / "in.wav", tmp_path / "out.wav", preset, medida()
        )
        unidos = " ".join(cmd)
        assert "acompressor" not in unidos
        assert "deesser" not in unidos


def test_the_output_keeps_full_resolution(tmp_path: Path) -> None:
    """loudnorm remuestrea a 192k internamente; sin fijar el formato de salida se
    escribiría un wav de 32 bits float que después pesa el doble."""
    cmd = build_master_command(
        Path("ffmpeg.exe"), tmp_path / "in.wav", tmp_path / "out.wav", "streaming", medida()
    )
    assert "-c:a" in cmd
    assert "pcm_s24le" in cmd


def test_the_input_and_output_paths_are_wired(tmp_path: Path) -> None:
    entrada, salida = tmp_path / "in.wav", tmp_path / "out.wav"
    cmd = build_master_command(Path("ffmpeg.exe"), entrada, salida, "streaming", medida())
    assert str(entrada) in cmd
    assert cmd[-1] == str(salida)


# --- integracion con la cadena -------------------------------------------


@pytest.mark.asyncio
async def test_the_chain_masters_and_records_what_it_measured(tmp_path: Path, monkeypatch) -> None:
    from app.config import Settings
    from app.models import AudioJob
    from app.services.audio_pipeline import AudioPipeline

    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))
    pipeline = AudioPipeline(settings, audio_enhancers={}, restorers={})

    job = AudioJob(source_path=tmp_path / "in.wav", original_filename="in.wav", master="streaming")
    destino = tmp_path / "out.wav"

    async def fake_capturing(command):
        return SALIDA_REAL

    corridos: list = []

    async def fake_run(command, failure_message):
        corridos.append(command)
        destino.write_bytes(b"wav")

    monkeypatch.setattr(pipeline, "_run_process_capturing", fake_capturing)
    monkeypatch.setattr(pipeline, "_run_process", fake_run)

    assert await pipeline._master(job, tmp_path / "in.wav", destino) is True
    assert job.metadata["loudnessBefore"] == -27.61
    assert job.metadata["loudnessTarget"] == -14.0
    assert "measured_I=-27.61" in " ".join(corridos[0])


@pytest.mark.asyncio
async def test_a_failed_measurement_does_not_kill_the_job(tmp_path: Path, monkeypatch) -> None:
    """El acabado es el ultimo paso: si falla, se entrega el audio ya procesado
    sin masterizar en vez de perder todo el trabajo."""
    from app.config import Settings
    from app.models import AudioJob
    from app.services.audio_pipeline import AudioPipeline

    settings = Settings(_env_file=None, RUNTIME_DIR=str(tmp_path / "runtime"))
    pipeline = AudioPipeline(settings, audio_enhancers={}, restorers={})
    job = AudioJob(source_path=tmp_path / "in.wav", original_filename="in.wav", master="streaming")

    async def sin_medicion(command):
        return "ffmpeg no imprimio nada util"

    monkeypatch.setattr(pipeline, "_run_process_capturing", sin_medicion)

    assert await pipeline._master(job, tmp_path / "in.wav", tmp_path / "out.wav") is False
    assert "masteringSkipped" in job.metadata
