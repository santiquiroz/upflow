from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings
from app.services.capabilities import HostProbes, ResolvedCapability, resolve_one
from app.services.ffmpeg_capabilities import (
    FfmpegCapabilities,
    FfmpegProbeError,
    cached_capabilities,
    unavailable_steps,
)

CapabilitiesLoader = Callable[[Path], FfmpegCapabilities]


@dataclass(frozen=True, slots=True)
class CctvCapabilityView:
    available: bool
    reason_key: str | None
    ai_available: bool
    ai_reason_key: str | None
    unavailable_steps: tuple[str, ...]


def _reason_key(resolved: ResolvedCapability) -> str | None:
    return None if resolved.status == "available" else resolved.setup_reason_key


def build_unavailable_steps(settings: Settings, load: CapabilitiesLoader) -> tuple[str, ...]:
    try:
        return unavailable_steps(load(settings.ffmpeg_binary_path))
    except FfmpegProbeError:
        return ()


def cctv_capability_view(
    settings: Settings, probes: HostProbes, load: CapabilitiesLoader = cached_capabilities
) -> CctvCapabilityView:
    classic = resolve_one("video.cctv", settings, None, probes)
    ai = resolve_one("video.cctvAi", settings, None, probes)
    return CctvCapabilityView(
        available=classic.status == "available",
        reason_key=_reason_key(classic),
        ai_available=ai.status == "available",
        ai_reason_key=_reason_key(ai),
        unavailable_steps=build_unavailable_steps(settings, load),
    )
