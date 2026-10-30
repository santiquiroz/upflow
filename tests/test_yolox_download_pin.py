from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download-yolox.ps1"
MODEL_SIZE = 35858002
MODEL_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
MODEL_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"


def run_installer(root: Path, stub: str, destination: Path) -> subprocess.CompletedProcess:
    script_dir = root / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    copied = script_dir / SCRIPT.name
    shutil.copy2(SCRIPT, copied)
    path = str(copied).replace("'", "''")
    target = str(destination).replace("'", "''")
    return subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            f"{stub}\n& '{path}' -Destino '{target}'",
        ],
        capture_output=True,
        timeout=60,
    )


def test_model_download_is_pinned_to_the_measured_artifact():
    script = SCRIPT.read_text(encoding="utf-8-sig")

    assert MODEL_URL in script
    assert MODEL_SHA256 in script
    assert str(MODEL_SIZE) in script
    assert "vendor\\yolox" in script
    assert "SHA-256" in script


requires_powershell = pytest.mark.skipif(
    sys.platform != "win32" or shutil.which("powershell") is None,
    reason="El instalador requiere Windows PowerShell",
)


@requires_powershell
def test_same_size_corrupt_download_fails_hash_and_leaves_no_model(tmp_path):
    destination = tmp_path / "temporary-destination"
    stub = f"""
function Invoke-WebRequest {{
    param([string]$Uri, [string]$OutFile, [switch]$UseBasicParsing)
    $stream = [System.IO.File]::Create($OutFile)
    try {{ $stream.SetLength({MODEL_SIZE}) }} finally {{ $stream.Dispose() }}
}}
"""

    result = run_installer(tmp_path, stub, destination)

    assert result.returncode != 0
    assert "SHA-256" in (result.stderr + result.stdout).decode("utf-8", "replace")
    assert list(destination.iterdir()) == []
