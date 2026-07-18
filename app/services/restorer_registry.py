from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.config import APOLLO_MODE, AUDIOSR_MODE, Settings


class AudioRestorer(Protocol):
    """Shared contract of ApolloRestorer and AudioSrRestorer."""

    def available(self) -> bool: ...

    async def run(self, input_wav: Path, output_wav: Path, device: str) -> None: ...


def validate_restore_mode_ready(settings: Settings, mode: str) -> None:
    """Raises ValueError with distinct disabled vs not-installed messages,
    same split as the video audio_enhance validation."""
    if mode == APOLLO_MODE:
        if not settings.enable_audio_restore:
            raise ValueError(
                "Audio restoration is disabled by configuration (set ENABLE_AUDIO_RESTORE=true)"
            )
        if not settings.apollo_restore_model_path.exists():
            raise ValueError(
                f"restore mode {mode!r} requested but the Apollo model is not installed "
                "(run scripts/download-apollo.ps1)"
            )
        return
    if mode == AUDIOSR_MODE:
        if not settings.enable_audiosr:
            raise ValueError(
                "AudioSR restoration is disabled by configuration (set ENABLE_AUDIOSR=true)"
            )
        if not settings.audiosr_available():
            raise ValueError(
                f"restore mode {mode!r} requested but the AudioSR models are not installed "
                "(run scripts/download-audiosr-onnx.ps1)"
            )
        return
    raise ValueError(f"Unknown restore mode: {mode!r}")


def build_restorers(settings: Settings) -> dict[str, AudioRestorer]:
    from app.services.engines.apollo_restore import ApolloRestorer
    from app.services.engines.audiosr_restore import AudioSrRestorer

    return {
        APOLLO_MODE: ApolloRestorer(settings),
        AUDIOSR_MODE: AudioSrRestorer(settings),
    }
