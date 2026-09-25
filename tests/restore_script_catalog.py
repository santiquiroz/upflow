"""La tabla de scripts/download-restore.ps1, leida por PowerShell y no por regex.

Leer el texto no prueba que PowerShell lo entienda como uno cree: la tabla se
evalua con Windows PowerShell 5.1, que es el que corre en la instalacion.
"""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from app.services.restore_models import (
    RestoreArtifact,
    RestoreBundle,
    RestoreLicenseFile,
    RestoreModelSpec,
)

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "download-restore.ps1"
CATALOG_START = "$catalogo = @{"
CATALOG_END = "}"

_VALIDATE_SET = re.compile(r"\[ValidateSet\(([^)]*)\)\]")
_QUOTED = re.compile(r"'([^']*)'")
_SHA256 = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{64}(?![0-9a-fA-F])")

requires_powershell = pytest.mark.skipif(
    sys.platform != "win32" or shutil.which("powershell") is None,
    reason="Hace falta Windows PowerShell, que es el que corre en la instalacion",
)


def script_text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def validate_set(text: str) -> tuple[str, ...]:
    match = _VALIDATE_SET.search(text)
    assert match is not None, "download-restore.ps1 no declara ValidateSet"
    return tuple(_QUOTED.findall(match.group(1)))


def declared_sha256(text: str) -> set[str]:
    return set(_SHA256.findall(text))


def _catalog_bounds(lines: list[str]) -> tuple[int, int]:
    start = lines.index(CATALOG_START)
    return start, lines.index(CATALOG_END, start + 1)


def catalog_block(text: str) -> str:
    lines = text.splitlines()
    start, end = _catalog_bounds(lines)
    return "\n".join(lines[start : end + 1])


def with_catalog(text: str, block: str) -> str:
    lines = text.splitlines()
    start, end = _catalog_bounds(lines)
    return "\n".join(lines[:start] + block.splitlines() + lines[end + 1 :]) + "\n"


def run_powershell(command: str, *, timeout: float = 120) -> subprocess.CompletedProcess:
    # -EncodedCommand evita que las comillas del bloque pasen por el quoting de la linea de comandos.
    encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-EncodedCommand", encoded],
        capture_output=True,
        timeout=timeout,
    )


def evaluate_catalog(block: str) -> dict:
    command = f"{block}\n[Console]::Out.Write((ConvertTo-Json -InputObject $catalogo -Depth 6 -Compress))"
    result = run_powershell(command)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    return json.loads(result.stdout.decode("utf-8"))


def file_row(artifact: RestoreArtifact, models: Mapping[str, RestoreModelSpec]) -> dict:
    return {
        "Model": artifact.model_id,
        "Precision": artifact.precision,
        "File": artifact.filename,
        "Url": artifact.url,
        "Sha256": artifact.sha256,
        "Size": artifact.size,
        "License": models[artifact.model_id].license_spdx,
    }


def license_row(license_file: RestoreLicenseFile) -> dict:
    return {
        "Model": license_file.model_id,
        "File": license_file.filename,
        "Url": license_file.url,
        "Sha256": license_file.sha256,
        "Size": license_file.size,
    }


def expected_catalog(
    bundles: Mapping[str, RestoreBundle], models: Mapping[str, RestoreModelSpec]
) -> dict:
    return {
        name: {
            "Files": sorted((file_row(a, models) for a in bundle.artifacts), key=_row_key),
            "LicenseFiles": sorted((license_row(f) for f in bundle.license_files), key=_row_key),
        }
        for name, bundle in bundles.items()
    }


def _row_key(row: dict) -> tuple[str, str]:
    return row["Model"], row["File"]


def comparable(catalog: dict) -> dict:
    return {
        name: {
            "Files": sorted((_without_label(row) for row in tables["Files"]), key=_row_key),
            "LicenseFiles": sorted(tables["LicenseFiles"], key=_row_key),
        }
        for name, tables in catalog.items()
    }


def _without_label(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "Label"}


def _ps_literal(value: str | int) -> str:
    if isinstance(value, int):
        return str(value)
    return "'" + value.replace("'", "''") + "'"


def _ps_row(row: Mapping[str, str | int]) -> str:
    return "@{ " + "; ".join(f"{key} = {_ps_literal(value)}" for key, value in row.items()) + " }"


def _ps_rows(rows: list[Mapping[str, str | int]]) -> list[str]:
    return ["@(", *(f"    {_ps_row(row)}" for row in rows), ")"]


def _labelled(row: dict) -> dict:
    return {**row, "Label": f"{row['Model']} {row['Precision']}"}


def render_catalog(
    bundles: Mapping[str, RestoreBundle], models: Mapping[str, RestoreModelSpec]
) -> str:
    lines = [CATALOG_START]
    for name, bundle in bundles.items():
        files = [_labelled(file_row(a, models)) for a in bundle.artifacts]
        licenses = [license_row(f) for f in bundle.license_files]
        lines.append(f"    '{name}' = @{{")
        lines += _indented("Files = ", _ps_rows(files))
        lines += _indented("LicenseFiles = ", _ps_rows(licenses))
        lines.append("    }")
    lines.append(CATALOG_END)
    return "\n".join(lines)


def _indented(prefix: str, rows: list[str]) -> list[str]:
    head, *rest = rows
    return [f"        {prefix}{head}", *(f"        {line}" for line in rest)]
