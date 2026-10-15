from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


LAUNCHER = Path(__file__).resolve().parent.parent / "scripts" / "upflow-launcher.ps1"
POWERSHELL = shutil.which("powershell.exe")
LEGACY_DEFAULT = (
    "DEFAULT_DEVICE=dml:0                 # id preferido (cpu|dml:N); "
    "si no hay GPU DirectML disponible, cae automaticamente a cpu"
)
EXAMPLE_DEFAULT = "DEFAULT_DEVICE=gpu                   # GPU preferida de Windows"

pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or POWERSHELL is None,
    reason="Hace falta Windows PowerShell, que es el que corre en la instalacion",
)


def powershell_literal(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def run_migration(env_path: Path, example_path: Path) -> subprocess.CompletedProcess:
    script = (
        "$ErrorActionPreference = 'Stop'\n"
        f"$envPath = {powershell_literal(env_path)}\n"
        f"$envExamplePath = {powershell_literal(example_path)}\n"
        "$tokens = $null\n"
        "$parseErrors = $null\n"
        "$ast = [System.Management.Automation.Language.Parser]::ParseFile("
        f"{powershell_literal(LAUNCHER)}, [ref]$tokens, [ref]$parseErrors)\n"
        "if ($parseErrors.Count -gt 0) { throw 'Launcher parse failed' }\n"
        "$functions = @($ast.FindAll({ param($node)\n"
        "    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] "
        "-and $node.Name -eq 'Update-LegacyDefaultDevice'\n"
        "}, $true))\n"
        "if ($functions.Count -ne 1) { throw 'Expected one migration function' }\n"
        ". ([scriptblock]::Create($functions[0].Extent.Text))\n"
        "Update-LegacyDefaultDevice\n"
    )
    process = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr.decode("utf-8", "replace")
    return process


@pytest.fixture
def env_files(tmp_path: Path) -> tuple[Path, Path]:
    env_path = tmp_path / ".env"
    example_path = tmp_path / ".env.example"
    example_path.write_text(
        f"# Plantilla\n{EXAMPLE_DEFAULT}\nENABLE_FEATURE=true\n", encoding="utf-8"
    )
    return env_path, example_path


def test_untouched_legacy_default_is_replaced_preserving_other_lines(
    env_files: tuple[Path, Path],
) -> None:
    env_path, example_path = env_files
    before = ["# Configuracion local", "PORT=8090", ""]
    after = ["", "ENABLE_FEATURE=false", "# Fin"]
    env_path.write_text(
        "\n".join([*before, LEGACY_DEFAULT, *after]) + "\n", encoding="utf-8"
    )

    process = run_migration(env_path, example_path)

    assert env_path.read_text(encoding="utf-8-sig").splitlines() == [
        *before,
        EXAMPLE_DEFAULT,
        *after,
    ]
    assert "gpu" in process.stdout.decode("utf-8", "replace").lower()


@pytest.mark.parametrize(
    "default_line",
    [
        "DEFAULT_DEVICE=dml:1",
        "DEFAULT_DEVICE=dml:0",
        "DEFAULT_DEVICE=cpu",
        "DEFAULT_DEVICE=gpu",
        "DEFAULT_DEVICE=dml:0 # Eleccion del usuario",
    ],
)
def test_user_default_is_unchanged_byte_for_byte(
    env_files: tuple[Path, Path], default_line: str
) -> None:
    env_path, example_path = env_files
    original = f"# Mi configuracion\r\n{default_line}\r\nPORT=8090\r\n".encode("utf-8")
    env_path.write_bytes(original)

    run_migration(env_path, example_path)

    assert env_path.read_bytes() == original


def test_missing_env_is_not_created(env_files: tuple[Path, Path]) -> None:
    env_path, example_path = env_files

    run_migration(env_path, example_path)

    assert not env_path.exists()
