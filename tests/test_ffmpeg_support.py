from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.config import Settings
from ffmpeg_support import (
    REQUIRE_FFMPEG_ENV,
    enforce_ffmpeg,
    ffmpeg_binaries,
    ffmpeg_is_required,
    missing_binaries,
    needs_ffmpeg,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_NODE = "tests/test_ffmpeg_support.py::test_the_configured_ffmpeg_runs"


def touch(path: Path) -> Path:
    path.write_bytes(b"")
    return path


def env_without_ffmpeg(tmp_path: Path, require_flag: str) -> dict[str, str]:
    return {
        **os.environ,
        "FFMPEG_BINARY": str(tmp_path / "no-ffmpeg.exe"),
        "FFPROBE_BINARY": str(tmp_path / "no-ffprobe.exe"),
        REQUIRE_FFMPEG_ENV: require_flag,
    }


def run_probe_without_ffmpeg(tmp_path: Path, require_flag: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", PROBE_NODE, "-q", "-rs", "-p", "no:cacheprovider"],
        cwd=REPO_ROOT,
        env=env_without_ffmpeg(tmp_path, require_flag),
        capture_output=True,
        text=True,
        timeout=120,
    )


@needs_ffmpeg
def test_the_configured_ffmpeg_runs() -> None:
    ffmpeg, ffprobe = ffmpeg_binaries(Settings())

    for binary in (ffmpeg, ffprobe):
        completed = subprocess.run([str(binary), "-version"], capture_output=True, timeout=30)
        assert completed.returncode == 0


def test_the_binaries_come_from_the_settings(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        RUNTIME_DIR=str(tmp_path / "runtime"),
        FFMPEG_BINARY=str(tmp_path / "ff.exe"),
        FFPROBE_BINARY=str(tmp_path / "fp.exe"),
    )

    assert ffmpeg_binaries(settings) == (tmp_path / "ff.exe", tmp_path / "fp.exe")


def test_only_the_absent_binaries_are_missing(tmp_path: Path) -> None:
    present = touch(tmp_path / "ffmpeg.exe")
    absent = tmp_path / "ffprobe.exe"

    assert missing_binaries((present, absent)) == (absent,)


def test_a_directory_is_not_a_binary(tmp_path: Path) -> None:
    assert missing_binaries((tmp_path,)) == (tmp_path,)


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({REQUIRE_FFMPEG_ENV: "1"}, True),
        ({REQUIRE_FFMPEG_ENV: "0"}, False),
        ({REQUIRE_FFMPEG_ENV: ""}, False),
        ({}, False),
    ],
)
def test_ffmpeg_is_required_only_with_the_flag_set_to_one(environ: dict[str, str], expected: bool) -> None:
    assert ffmpeg_is_required(environ) is expected


@pytest.mark.parametrize("required", [True, False])
def test_nothing_happens_when_no_binary_is_missing(required: bool) -> None:
    enforce_ffmpeg((), required)


def test_a_missing_binary_skips_by_default(tmp_path: Path) -> None:
    absent = tmp_path / "ffmpeg.exe"

    with pytest.raises(pytest.skip.Exception, match="ffmpeg.exe"):
        enforce_ffmpeg((absent,), required=False)


def test_a_missing_binary_fails_when_ffmpeg_is_required(tmp_path: Path) -> None:
    absent = tmp_path / "ffmpeg.exe"

    with pytest.raises(pytest.fail.Exception, match=f"{REQUIRE_FFMPEG_ENV}=1.*ffmpeg.exe"):
        enforce_ffmpeg((absent,), required=True)


def test_the_marked_test_is_skipped_without_ffmpeg(tmp_path: Path) -> None:
    completed = run_probe_without_ffmpeg(tmp_path, "")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "1 skipped" in completed.stdout


def test_the_marked_test_fails_without_ffmpeg_when_it_is_required(tmp_path: Path) -> None:
    completed = run_probe_without_ffmpeg(tmp_path, "1")

    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "1 error" in completed.stdout
    assert f"{REQUIRE_FFMPEG_ENV}=1 but" in completed.stdout
    assert "no-ffmpeg.exe" in completed.stdout
