from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Acabado profesional del audio: normalización de sonoridad a EBU R128.
#
# Es lo que separa un audio "procesado" de uno entregable. Quitar ruido y
# restaurar agudos arregla el material, pero deja el volumen donde estaba: dos
# archivos tratados igual suenan a distinto volumen, y ninguna plataforma los
# acepta así. Los objetivos no son gusto nuestro, son estándares publicados:
# -14 LUFS para streaming (Spotify/YouTube), -23 LUFS para broadcast (EBU R128).
#
# Se hace en DOS pasadas — medir y después aplicar con esa medición — porque
# loudnorm en una sola pasada trabaja adaptativo y bombea el volumen. Todo con
# ffmpeg, que ya viene con la app: cero dependencias nuevas.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MasteringPreset:
    id: str
    # Claves, no copia: ver realtime_service.RealtimePreset.
    label_key: str
    description_key: str
    target_lufs: float
    # Techo de pico real. Sobre -1 dBTP, recodificar a un formato con pérdida
    # clipea: el códec reconstruye picos por encima de la muestra original.
    true_peak: float
    loudness_range: float
    # Cadena previa a la sonoridad. Vacía en música: comprimir algo que no lo
    # pidió es exactamente lo que arruina un master.
    pre_filters: tuple[str, ...] = ()


MASTERING_PRESETS: tuple[MasteringPreset, ...] = (
    MasteringPreset(
        id="streaming",
        label_key="audio.mastering.preset.streaming.label",
        description_key="audio.mastering.preset.streaming.description",
        target_lufs=-14.0,
        true_peak=-1.0,
        loudness_range=11.0,
    ),
    MasteringPreset(
        id="broadcast",
        label_key="audio.mastering.preset.broadcast.label",
        description_key="audio.mastering.preset.broadcast.description",
        target_lufs=-23.0,
        true_peak=-1.0,
        loudness_range=7.0,
    ),
    MasteringPreset(
        id="voice",
        label_key="audio.mastering.preset.voice.label",
        description_key="audio.mastering.preset.voice.description",
        target_lufs=-16.0,
        true_peak=-1.5,
        loudness_range=7.0,
        # De-eser y compresión suave ANTES de fijar la sonoridad: si se hiciera
        # después, cambiarían el nivel que loudnorm acaba de dejar clavado.
        pre_filters=(
            "deesser=i=0.4",
            "acompressor=threshold=-18dB:ratio=2.5:attack=20:release=250",
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class LoudnessMeasurement:
    input_i: float
    input_tp: float
    input_lra: float
    input_thresh: float
    target_offset: float


def mastering_preset(preset_id: str) -> MasteringPreset:
    for preset in MASTERING_PRESETS:
        if preset.id == preset_id:
            return preset
    conocidos = ", ".join(p.id for p in MASTERING_PRESETS)
    raise ValueError(f"Preset de masterizado desconocido: {preset_id!r}. Conocidos: {conocidos}")


def _loudnorm_targets(preset: MasteringPreset) -> str:
    return f"I={preset.target_lufs}:TP={preset.true_peak}:LRA={preset.loudness_range}"


def build_measure_command(ffmpeg: Path, source: Path, preset_id: str) -> list[str]:
    """Primera pasada: medir. No escribe audio (-f null), solo imprime el JSON."""
    preset = mastering_preset(preset_id)
    return [
        str(ffmpeg), "-hide_banner", "-nostats", "-y",
        "-i", str(source),
        "-af", f"loudnorm={_loudnorm_targets(preset)}:print_format=json",
        "-f", "null", "-",
    ]


def parse_loudness_measurement(output: str) -> LoudnessMeasurement | None:
    """Lee el JSON que loudnorm imprime en stderr, entre el resto del ruido.

    Devuelve None si no está: sin medición se sigue sin masterizar, que es mejor
    que fallar el trabajo entero por el paso de acabado.
    """
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", output, re.DOTALL)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
        return LoudnessMeasurement(
            input_i=float(data["input_i"]),
            input_tp=float(data["input_tp"]),
            input_lra=float(data["input_lra"]),
            input_thresh=float(data["input_thresh"]),
            target_offset=float(data["target_offset"]),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def build_master_command(
    ffmpeg: Path,
    source: Path,
    destination: Path,
    preset_id: str,
    measurement: LoudnessMeasurement,
) -> list[str]:
    """Segunda pasada: aplicar, realimentando lo medido.

    Sin los `measured_*`, loudnorm vuelve al modo adaptativo de una pasada y el
    volumen bombea. `linear=true` le pide una corrección de ganancia constante,
    que es lo que uno quiere de un master.
    """
    preset = mastering_preset(preset_id)
    loudnorm = (
        f"loudnorm={_loudnorm_targets(preset)}"
        f":measured_I={measurement.input_i}"
        f":measured_TP={measurement.input_tp}"
        f":measured_LRA={measurement.input_lra}"
        f":measured_thresh={measurement.input_thresh}"
        f":offset={measurement.target_offset}"
        ":linear=true:print_format=summary"
    )
    chain = ",".join([*preset.pre_filters, loudnorm])
    return [
        str(ffmpeg), "-hide_banner", "-nostats", "-y",
        "-i", str(source),
        "-af", chain,
        # loudnorm remuestrea a 192 kHz internamente y sin fijar formato ffmpeg
        # escribe float de 32 bits, que pesa el doble sin ganar nada audible.
        "-c:a", "pcm_s24le",
        "-ar", "48000",
        str(destination),
    ]
