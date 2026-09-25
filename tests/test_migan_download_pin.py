"""download-migan.ps1: revision fijada, sha256 que corta y las dos licencias MIT.

El script decia "tamano y sha256 verificados" pero solo comparaba el tamano, y
bajaba directo al destino: una descarga corrupta del mismo tamano quedaba en
vendor/migan y la app la daba por instalada (`migan_available` solo mira que el
archivo exista). Ademas la revision fijada (2023) es anterior al LICENSE que el
repo de HF agrego el 2026-09-14, asi que el pack se distribuia sin licencia.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
MIGAN = RAIZ / "scripts" / "download-migan.ps1"

REVISION_HF = "406830d0fa60666da0071c342ad2fbc8f30c5c64"
SHA256_ONNX = "6f1f3530a1a2324b19752018ce756088b07973cda8d7d890034ace5c8a48c40b"
BYTES_ONNX = 28079181
COMMIT_GITHUB = "680b0c6149c51e272412f0e4686c0d6668dba15c"
SHA256_LICENSE_HF = "13348eaaebc0b07b0faf3a92db04e0858737713a917d064b599c87cac05b571f"
SHA256_LICENSE_WEIGHTS = "674e33456b8d03693b84ab305aa7372d93c91e68b234651b0adad106d518e6eb"


def sha256_de(contenido: bytes) -> str:
    return hashlib.sha256(contenido).hexdigest()


def texto_del_script() -> str:
    return MIGAN.read_text(encoding="utf-8")


class TestLoQueDeclaraElScript:
    def test_fija_la_revision_de_hf_que_trae_la_licencia(self) -> None:
        texto = texto_del_script()
        assert re.findall(r"\$revision\s*=\s*'([0-9a-f]{40})'", texto) == [REVISION_HF]

    def test_declara_el_sha256_y_el_tamano_del_onnx(self) -> None:
        texto = texto_del_script()
        assert SHA256_ONNX in texto
        assert str(BYTES_ONNX) in texto

    def test_baja_el_license_de_hf_en_la_misma_revision(self) -> None:
        texto = texto_del_script()
        assert "huggingface.co/andraniksargsyan/migan/resolve/$revision/LICENSE" in texto

    def test_baja_el_license_weights_de_github_en_un_commit_fijo(self) -> None:
        texto = texto_del_script()
        assert f"$weightsCommit = '{COMMIT_GITHUB}'" in texto
        assert (
            "raw.githubusercontent.com/Picsart-AI-Research/MI-GAN/$weightsCommit/LICENSE-WEIGHTS"
            in texto
        )
        assert "/main/LICENSE-WEIGHTS" not in texto

    def test_cada_archivo_lleva_un_sha256_propio(self) -> None:
        sha256s = set(re.findall(r"Sha256\s*=\s*'([0-9a-f]{64})'", texto_del_script()))
        assert len(sha256s) == 3

    def test_no_avisa_en_vez_de_fallar(self) -> None:
        assert not re.search(r"^\s*Write-Warning\b", texto_del_script(), re.MULTILINE)


# ---------------------------------------------------------------------------
# Corrido de verdad, con Invoke-WebRequest sustituido por una funcion de la
# sesion que invoca al script (en PowerShell la funcion gana al cmdlet).
# ---------------------------------------------------------------------------

requiere_powershell = pytest.mark.skipif(
    sys.platform != "win32" or shutil.which("powershell") is None,
    reason="Hace falta Windows PowerShell, que es el que corre en la instalacion",
)

STUB_ONNX_DEL_TAMANO_JUSTO = f"""
function Invoke-WebRequest {{
    param([string]$Uri, [string]$OutFile, [switch]$UseBasicParsing)
    if ($Uri -like '*.onnx') {{
        $stream = [System.IO.File]::Create($OutFile)
        try {{ $stream.SetLength({BYTES_ONNX}) }} finally {{ $stream.Dispose() }}
        return
    }}
    throw "sin red para $Uri"
}}
"""


def correr_migan(raiz: Path, preparacion: str) -> subprocess.CompletedProcess:
    (raiz / "scripts").mkdir(parents=True, exist_ok=True)
    copia = raiz / "scripts" / MIGAN.name
    shutil.copy2(MIGAN, copia)
    return subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            f"{preparacion}\n& '{copia}'",
        ],
        capture_output=True,
        timeout=180,
    )


def salida_de(resultado: subprocess.CompletedProcess) -> str:
    return (resultado.stderr + resultado.stdout).decode("utf-8", "replace")


def ignorar_licencias(raiz: Path) -> None:
    # Las licencias se instalan antes que el ONNX; para ejercitar el chequeo del
    # ONNX solo hace falta que las licencias ya esten y verifiquen.
    destino = raiz / "vendor" / "migan"
    destino.mkdir(parents=True, exist_ok=True)
    (destino / "LICENSE").write_bytes(LICENCIA_MIT_PAIR)
    (destino / "LICENSE-WEIGHTS").write_bytes(LICENCIA_WEIGHTS_PAIR)


LICENCIA_MIT_PAIR = (
    b"MIT License\n\nCopyright (c) 2024 Picsart AI Research (PAIR)\n\n"
    b"Permission is hereby granted, free of charge, to any person obtaining a copy\n"
    b"of this software and associated documentation files (the \"Software\"), to deal\n"
    b"in the Software without restriction, including without limitation the rights\n"
    b"to use, copy, modify, merge, publish, distribute, sublicense, and/or sell\n"
    b"copies of the Software, and to permit persons to whom the Software is\n"
    b"furnished to do so, subject to the following conditions:\n\n"
    b"The above copyright notice and this permission notice shall be included in all\n"
    b"copies or substantial portions of the Software.\n\n"
    b"THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\n"
    b"IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\n"
    b"FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\n"
    b"AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\n"
    b"LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\n"
    b"OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE\n"
    b"SOFTWARE."
)
# El LICENSE-WEIGHTS de GitHub es el mismo texto con salto de linea final.
LICENCIA_WEIGHTS_PAIR = LICENCIA_MIT_PAIR + b"\n"


def test_las_licencias_de_la_prueba_son_las_que_fija_el_script() -> None:
    assert sha256_de(LICENCIA_MIT_PAIR) == SHA256_LICENSE_HF
    assert sha256_de(LICENCIA_WEIGHTS_PAIR) == SHA256_LICENSE_WEIGHTS
    texto = texto_del_script()
    assert SHA256_LICENSE_HF in texto
    assert SHA256_LICENSE_WEIGHTS in texto


@requiere_powershell
class TestElScriptCorridoDeVerdad:

    def test_un_onnx_del_tamano_justo_pero_otro_contenido_falla_por_sha256(
        self, tmp_path: Path
    ) -> None:
        # Exactamente el agujero de antes: el chequeo por tamano lo dejaba pasar.
        ignorar_licencias(tmp_path)

        resultado = correr_migan(tmp_path, STUB_ONNX_DEL_TAMANO_JUSTO)

        assert resultado.returncode != 0
        assert "SHA-256" in salida_de(resultado)
        migan = tmp_path / "vendor" / "migan"
        assert not (migan / "migan_pipeline_v2.onnx").exists()
        assert not (migan / "migan_pipeline_v2.onnx.download").exists()

    def test_un_onnx_corrupto_ya_instalado_no_cuenta_como_instalado(
        self, tmp_path: Path
    ) -> None:
        # Lo que pudo dejar el script viejo: se borra para que `migan_available`
        # no lo de por bueno, y como no hay red, el script falla.
        ignorar_licencias(tmp_path)
        onnx = tmp_path / "vendor" / "migan" / "migan_pipeline_v2.onnx"
        with onnx.open("wb") as archivo:
            archivo.truncate(BYTES_ONNX)

        resultado = correr_migan(tmp_path, "function Invoke-WebRequest { throw 'sin red' }")

        assert resultado.returncode != 0
        assert not onnx.exists()

    def test_sin_poder_bajar_la_licencia_falla_y_no_instala_el_onnx(
        self, tmp_path: Path
    ) -> None:
        resultado = correr_migan(tmp_path, STUB_ONNX_DEL_TAMANO_JUSTO)

        assert resultado.returncode != 0
        assert "LICENSE" in salida_de(resultado)
        migan = tmp_path / "vendor" / "migan"
        assert not (migan / "migan_pipeline_v2.onnx").exists()
        assert not (migan / "LICENSE").exists()

    def test_una_licencia_alterada_se_vuelve_a_bajar(self, tmp_path: Path) -> None:
        ignorar_licencias(tmp_path)
        alterada = tmp_path / "vendor" / "migan" / "LICENSE-WEIGHTS"
        alterada.write_bytes(b"otra licencia")

        resultado = correr_migan(tmp_path, "function Invoke-WebRequest { throw 'sin red' }")

        assert resultado.returncode != 0
        assert "LICENSE-WEIGHTS" in salida_de(resultado)
        assert not alterada.exists()
