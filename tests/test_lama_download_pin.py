from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download-lama.ps1"
MODEL_SIZE = 205669692
MODEL_SHA256 = "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9"
MODEL_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"


def run_installer(root: Path, stub: str) -> subprocess.CompletedProcess:
    script_dir = root / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    copied = script_dir / SCRIPT.name
    shutil.copy2(SCRIPT, copied)
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", f"{stub}\n& '{copied}'"],
        capture_output=True, timeout=60,
    )


def test_model_download_is_pinned_to_the_measured_artifact():
    script = SCRIPT.read_text(encoding="utf-8-sig")

    assert MODEL_URL in script
    assert MODEL_SHA256 in script
    assert str(MODEL_SIZE) in script
    assert "vendor\\lama" in script
    assert "SHA-256" in script


requires_powershell = pytest.mark.skipif(
    sys.platform != "win32" or shutil.which("powershell") is None,
    reason="El instalador requiere Windows PowerShell",
)


@requires_powershell
def test_same_size_corrupt_download_fails_hash_and_leaves_no_model(tmp_path):
    stub = f"""
function Invoke-WebRequest {{
    param([string]$Uri, [string]$OutFile, [switch]$UseBasicParsing)
    $stream = [System.IO.File]::Create($OutFile)
    try {{ $stream.SetLength({MODEL_SIZE}) }} finally {{ $stream.Dispose() }}
}}
"""

    result = run_installer(tmp_path, stub)

    assert result.returncode != 0
    assert "SHA-256" in (result.stderr + result.stdout).decode("utf-8", "replace")
    assert not (tmp_path / "vendor/lama/big-lama.pt").exists()
    assert not (tmp_path / "vendor/lama/big-lama.pt.download").exists()


@requires_powershell
def test_existing_corrupt_model_is_removed_when_redownload_fails(tmp_path):
    model = tmp_path / "vendor/lama/big-lama.pt"
    model.parent.mkdir(parents=True)
    with model.open("wb") as stream:
        stream.truncate(MODEL_SIZE)

    result = run_installer(tmp_path, "function Invoke-WebRequest { throw 'sin red' }")

    assert result.returncode != 0
    assert not model.exists()
    assert not model.with_name(model.name + ".download").exists()
