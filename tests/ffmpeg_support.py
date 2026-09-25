"""`needs_ffmpeg` compartido: sin los binarios el test se saltea, pero con
UPFLOW_REQUIRE_FFMPEG=1 falla. Asi una corrida que exige ffmpeg real no puede
terminar en verde con los tests de ffmpeg salteados en silencio.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from app.config import Settings

REQUIRE_FFMPEG_ENV = "UPFLOW_REQUIRE_FFMPEG"
FFMPEG_MARKER = "ffmpeg"

needs_ffmpeg = pytest.mark.ffmpeg


def ffmpeg_binaries(settings: Settings) -> tuple[Path, Path]:
    return settings.ffmpeg_binary_path, settings.ffprobe_binary_path


def missing_binaries(binaries: tuple[Path, ...]) -> tuple[Path, ...]:
    return tuple(path for path in binaries if not path.is_file())


def ffmpeg_is_required(environ: Mapping[str, str]) -> bool:
    return environ.get(REQUIRE_FFMPEG_ENV) == "1"


def missing_ffmpeg_reason(missing: tuple[Path, ...]) -> str:
    return "ffmpeg/ffprobe not found: " + ", ".join(str(path) for path in missing)


def enforce_ffmpeg(missing: tuple[Path, ...], required: bool) -> None:
    if not missing:
        return
    reason = missing_ffmpeg_reason(missing)
    if required:
        pytest.fail(f"{REQUIRE_FFMPEG_ENV}=1 but {reason}", pytrace=False)
    pytest.skip(reason)


def check_ffmpeg_for(item: pytest.Item, environ: Mapping[str, str]) -> None:
    if item.get_closest_marker(FFMPEG_MARKER) is None:
        return
    missing = missing_binaries(ffmpeg_binaries(Settings()))
    enforce_ffmpeg(missing, ffmpeg_is_required(environ))
