from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import Settings

# ---------------------------------------------------------------------------
# Fase 7.1 - overlay de reescalado en tiempo real.
#
# Upflow es el PLANO DE CONTROL, no el motor: la captura, la inferencia y la
# presentación viven en Magpie, que corre como proceso aparte. Eso no es un
# atajo, es obligatorio: Magpie es GPL-3.0 y linkearlo volvería GPL a Upflow.
# Tampoco se redistribuye — se baja en la máquina del usuario, igual que el
# acelerador de NVIDIA, así que no hay redistribución nuestra que justificar.
#
# NO hace falta ningún driver. Magpie presenta en una ventana overlay sin
# enganchar el swapchain del juego, que es justamente por qué no tiene riesgo
# de anti-cheat. Los drivers harían falta solo para frame generation, que hoy
# no tiene camino open-source en Windows (ver docs/REALTIME_MODULE.md).
#
# La superficie de control es el propio config.json de Magpie. Verificado
# corriendo el binario real v0.12.1: deja
# %LOCALAPPDATA%\Magpie\config\v4\config.json con `scalingModes` (modos con
# nombre) y `profiles[0].scalingMode` como índice del modo activo.
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# Se AGREGA un modo con nombre propio en vez de pisar los del usuario: el
# archivo es de Magpie y alguien puede usarlo a mano fuera de Upflow.
UPFLOW_MODE_NAME = "Upflow"

MAGPIE_CONFIG_RELATIVE = Path("Magpie") / "config" / "v4" / "config.json"


@dataclass(frozen=True, slots=True)
class RealtimePreset:
    id: str
    # Claves, no copia: el texto que se ve lo traduce el frontend. Mandar la
    # frase desde aca dejaba la pantalla sin traducir en el otro idioma.
    label_key: str
    description_key: str
    # Efectos tal como los nombra Magpie dentro de su carpeta effects/.
    effects: tuple[dict[str, Any], ...]


# Solo presets que caben en el presupuesto de un frame (~16 ms a 60 fps). Los
# modelos de transformer quedan afuera a propósito: son de la ruta offline.
PRESETS: tuple[RealtimePreset, ...] = (
    RealtimePreset(
        id="anime4k",
        label_key="realtime.preset.anime4k.label",
        description_key="realtime.preset.anime4k.description",
        effects=({"name": "Anime4K\\Anime4K_Upscale_Denoise_L"},),
    ),
    RealtimePreset(
        id="cunny",
        label_key="realtime.preset.cunny.label",
        description_key="realtime.preset.cunny.description",
        effects=({"name": "CuNNy2\\CuNNy-4x12-NVL"},),
    ),
    RealtimePreset(
        id="fsr",
        label_key="realtime.preset.fsr.label",
        description_key="realtime.preset.fsr.description",
        effects=(
            {"name": "FSR\\FSR_EASU", "scalingType": 1, "scale": {"x": 1.0, "y": 1.0}},
            {"name": "FSR\\FSR_RCAS", "parameters": {"sharpness": 0.87}},
        ),
    ),
    RealtimePreset(
        id="lanczos",
        label_key="realtime.preset.lanczos.label",
        description_key="realtime.preset.lanczos.description",
        effects=({"name": "Lanczos", "scalingType": 1, "scale": {"x": 1.0, "y": 1.0}},),
    ),
)


def available_presets() -> tuple[RealtimePreset, ...]:
    return PRESETS


def _preset(preset_id: str) -> RealtimePreset:
    for candidate in PRESETS:
        if candidate.id == preset_id:
            return candidate
    conocidos = ", ".join(p.id for p in PRESETS)
    raise ValueError(f"Preset de tiempo real desconocido: {preset_id!r}. Conocidos: {conocidos}")


def magpie_config_path(settings: Settings) -> Path:
    if settings.magpie_config:
        return Path(settings.magpie_config)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / MAGPIE_CONFIG_RELATIVE


def apply_upflow_mode(
    config: dict[str, Any], *, preset: str, max_frame_rate: int | None = None
) -> dict[str, Any]:
    """Deja el modo de Upflow activo SIN tocar el resto del config.

    Se muta el dict recibido en vez de armar uno nuevo: el archivo es de Magpie
    y una versión más nueva puede traer claves que no conocemos. Reconstruirlo
    desde cero le borraría los ajustes al usuario en silencio.
    """
    chosen = _preset(preset)

    modes: list[dict[str, Any]] = config.setdefault("scalingModes", [])
    nuestro = {"name": UPFLOW_MODE_NAME, "effects": [dict(e) for e in chosen.effects]}
    for index, mode in enumerate(modes):
        if mode.get("name") == UPFLOW_MODE_NAME:
            modes[index] = nuestro
            break
    else:
        index = len(modes)
        modes.append(nuestro)

    profiles: list[dict[str, Any]] = config.setdefault("profiles", [])
    if not profiles:
        profiles.append({})
    profiles[0]["scalingMode"] = index
    if max_frame_rate is not None:
        profiles[0]["frameRateLimiterEnabled"] = True
        profiles[0]["maxFrameRate"] = float(max_frame_rate)
    return config


class RealtimeService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def available(self) -> bool:
        return self.settings.magpie_binary_path.is_file()

    def _load_config(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            # No se pisa: reescribirlo le borraría los modos que tenga hechos.
            raise RuntimeError(
                f"El config de Magpie no se pudo leer y no se va a sobrescribir: {path} ({exc})"
            ) from exc
        if not isinstance(loaded, dict):
            raise RuntimeError(f"El config de Magpie no es un objeto JSON: {path}")
        return loaded

    def _spawn(self, command: list[str]) -> int:
        # Sin ventana de consola y desligado: es una app de escritorio que vive
        # mientras el usuario la use, no un trabajo de la cola.
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        return subprocess.Popen(command, creationflags=flags).pid

    def start(self, *, preset: str, max_frame_rate: int | None = None) -> int:
        if not self.available():
            raise RuntimeError(
                "El overlay de tiempo real no está instalado: corré "
                "scripts/download-magpie.ps1 o instalá el componente desde el instalador."
            )
        path = magpie_config_path(self.settings)
        config = apply_upflow_mode(
            self._load_config(path), preset=preset, max_frame_rate=max_frame_rate
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config, indent=4), encoding="utf-8")
        pid = self._spawn([str(self.settings.magpie_binary_path)])
        logger.info("realtime: overlay lanzado (pid %s, preset %s)", pid, preset)
        return pid
