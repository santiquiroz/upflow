from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.config import APOLLO_MODE, AUDIOSR_MODE, Settings
from app.services.capabilities import resolve_one
from app.services.gpu_session_coordinator import GpuSessionCoordinator
from app.services.missing_pack import missing_pack_message


class AudioRestorer(Protocol):
    """Shared contract of ApolloRestorer and AudioSrRestorer."""

    def available(self) -> bool: ...

    async def run(self, input_wav: Path, output_wav: Path, device: str) -> None: ...


# Que capacidad del CATALOG respalda cada modo: la fuente de "¿esta instalado el
# pack?" es la resolucion de capabilities, no una re-lectura a mano de settings.
_CAPABILITY_BY_MODE = {APOLLO_MODE: "audio.restore", AUDIOSR_MODE: "audio.restoreSr"}


def _missing_pack_for_mode(settings: Settings, mode: str) -> str | None:
    resolved = resolve_one(_CAPABILITY_BY_MODE[mode], settings, None)
    return resolved.missing_packs[0] if resolved.missing_packs else None


def _raise_missing_restore_pack(pack: str, mode: str) -> None:
    raise ValueError(
        missing_pack_message(pack, detail=f"Lo pide el modo de restauracion {mode!r}.")
    )


def validate_restore_mode_ready(settings: Settings, mode: str) -> None:
    """Raises ValueError with distinct disabled vs not-installed messages,
    same split as the video audio_enhance validation."""
    if mode == APOLLO_MODE:
        if not settings.enable_audio_restore:
            raise ValueError(
                "Audio restoration is disabled by configuration (set ENABLE_AUDIO_RESTORE=true)"
            )
        missing = _missing_pack_for_mode(settings, mode)
        if missing is not None:
            _raise_missing_restore_pack(missing, mode)
        return
    if mode == AUDIOSR_MODE:
        if not settings.enable_audiosr:
            raise ValueError(
                "AudioSR restoration is disabled by configuration (set ENABLE_AUDIOSR=true)"
            )
        missing = _missing_pack_for_mode(settings, mode)
        if missing is not None:
            _raise_missing_restore_pack(missing, mode)
        if not settings.audiosr_available():
            # Mas fuerte que el PathRequirement del catalogo a proposito: una
            # descarga a medias deja la carpeta existiendo sin que el motor
            # pueda cargar (la tarjeta puede decir "disponible" con la carpeta
            # sola; validar un job no).
            _raise_missing_restore_pack("audiosr", mode)
        return
    raise ValueError(f"Unknown restore mode: {mode!r}")


def build_restorers(settings: Settings, gpu_coordinator: GpuSessionCoordinator) -> dict[str, AudioRestorer]:
    from app.services.engines.apollo_restore import ApolloRestorer
    from app.services.engines.audiosr_restore import AudioSrRestorer

    return {
        APOLLO_MODE: ApolloRestorer(settings, gpu_coordinator),
        AUDIOSR_MODE: AudioSrRestorer(settings, gpu_coordinator),
    }
