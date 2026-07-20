from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AudioTrackInfo:
    index: int
    codec: str
    channels: int
    is_default: bool
    language: str | None


@dataclass(frozen=True)
class SubtitleTrackInfo:
    index: int
    codec: str
    language: str | None


def _language_tag(stream: dict[str, Any]) -> str | None:
    return stream.get("tags", {}).get("language")


def _is_default(stream: dict[str, Any]) -> bool:
    return bool(stream.get("disposition", {}).get("default"))


def parse_audio_tracks(probe: dict[str, Any]) -> list[AudioTrackInfo]:
    return [
        AudioTrackInfo(
            index=stream["index"],
            codec=stream.get("codec_name", "unknown"),
            channels=int(stream.get("channels", 1)),
            is_default=_is_default(stream),
            language=_language_tag(stream),
        )
        for stream in probe.get("streams", [])
        if stream.get("codec_type") == "audio"
    ]


def parse_subtitle_tracks(probe: dict[str, Any]) -> list[SubtitleTrackInfo]:
    return [
        SubtitleTrackInfo(
            index=stream["index"],
            codec=stream.get("codec_name", "unknown"),
            language=_language_tag(stream),
        )
        for stream in probe.get("streams", [])
        if stream.get("codec_type") == "subtitle"
    ]
