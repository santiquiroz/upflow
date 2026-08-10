"""Un script de descarga no puede terminar en 0 sin dejar el pack usable.

Pasado real reportado por el usuario (2026-08-10): la pantalla de Voz decia
"falta el modelo de conversion de voz" DESPUES de que la app le dijera que ya
estaba bajado. `download-voice-conversion.ps1` bajaba dos de las tres piezas,
avisaba de la tercera con un `Write-Warning` y terminaba en 0. `PackProvisioner`
solo mira el codigo de salida, asi que daba el pack por instalado.

Son dos invariantes distintas y las dos hacen falta:

  * el script FALLA si no produjo lo que la capacidad necesita (aca);
  * el CATALOG declara todo lo que el motor exige (test_capabilities.py).

Con una sola, el agujero sigue abierto por el otro lado.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from app.services.capabilities import CATALOG, PathRequirement
from app.services.pack_provisioner import PACK_SCRIPTS

RAIZ = Path(__file__).resolve().parent.parent
SCRIPTS = RAIZ / "scripts"
CONVERSION_DE_VOZ = SCRIPTS / "download-voice-conversion.ps1"


# ---------------------------------------------------------------------------
# El barrido: todo script que respalda una tarjeta tiene que poder fallar
# ---------------------------------------------------------------------------


def scripts_de_capacidades() -> list[Path]:
    packs = {
        requirement.pack
        for capability in CATALOG
        for requirement in capability.requirements
        if isinstance(requirement, PathRequirement)
    }
    return sorted({SCRIPTS / PACK_SCRIPTS[pack] for pack in packs if pack in PACK_SCRIPTS})


SCRIPTS_DE_CAPACIDADES = scripts_de_capacidades()


def test_hay_scripts_que_revisar() -> None:
    # Sin esto, un cambio en el catalogo dejaria la parametrizacion vacia y todo
    # el barrido en verde sin haber mirado nada.
    assert SCRIPTS_DE_CAPACIDADES


@pytest.mark.parametrize(
    "script", SCRIPTS_DE_CAPACIDADES, ids=[s.name for s in SCRIPTS_DE_CAPACIDADES]
)
def test_todo_script_de_una_tarjeta_puede_fallar(script: Path) -> None:
    texto = script.read_text(encoding="utf-8", errors="replace")
    assert re.search(r"\b(throw|Write-Error)\b", texto), (
        f"{script.name} no tiene ninguna forma de fallar. Un script que siempre "
        "termina en 0 le dice al provisioner que el pack quedo instalado, aunque "
        "no haya bajado nada."
    )


@pytest.mark.parametrize(
    "script", SCRIPTS_DE_CAPACIDADES, ids=[s.name for s in SCRIPTS_DE_CAPACIDADES]
)
def test_ningun_script_avisa_de_una_pieza_faltante_en_vez_de_fallar(script: Path) -> None:
    # `Write-Warning` seguido de terminar normalmente es EXACTAMENTE el bug: el
    # usuario no ve esa salida y el provisioner no la mira. Si algo falta, se
    # lanza; si es de verdad opcional, se dice con Write-Host.
    ofensas = [
        f"{numero}: {linea.strip()}"
        for numero, linea in enumerate(
            script.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        )
        if re.match(r"^\s*Write-Warning\b", linea)
    ]
    assert ofensas == [], (
        f"{script.name} avisa en vez de fallar:\n" + "\n".join(ofensas) + "\n"
        "Un aviso no cambia el codigo de salida, asi que el pack queda dado por "
        "instalado igual."
    )


# ---------------------------------------------------------------------------
# La conversion de voz, en detalle: son TRES piezas
# ---------------------------------------------------------------------------


class TestElScriptDeConversionDeVoz:
    def test_nombra_las_tres_piezas(self) -> None:
        texto = CONVERSION_DE_VOZ.read_text(encoding="utf-8")
        for pieza in ("speecht5-vc", "speecht5-hifigan", "tdnn.onnx"):
            assert pieza in texto, pieza

    def test_baja_el_xvector_en_vez_de_pedirlo_a_mano(self) -> None:
        # Antes mandaba a crear un venv con torch+speechbrain (~2 GB). Ahora sale
        # de un release publicado, como cualquier otro pack.
        texto = CONVERSION_DE_VOZ.read_text(encoding="utf-8")
        assert "port-xvector-onnx/releases/download/" in texto
        assert "speechbrain==" not in texto

    def test_verifica_la_integridad_de_lo_que_baja(self) -> None:
        # Un x-vector corrupto no revienta: devuelve embeddings fuera del espacio
        # y la conversion sale en NaN. Por eso se chequea sha256 y tamano.
        texto = CONVERSION_DE_VOZ.read_text(encoding="utf-8")
        assert re.search(r"\$xvectorSha\s*=\s*'[0-9a-f]{64}'", texto)
        assert re.search(r"\$xvectorBytes\s*=\s*\d+", texto)

    def test_falla_si_al_final_falta_cualquiera_de_las_tres(self) -> None:
        texto = CONVERSION_DE_VOZ.read_text(encoding="utf-8")
        bloque = texto[texto.index("$requeridos") :]
        for pieza in ("$vcDir", "$vocoderDir", "$xvector"):
            assert pieza in bloque, pieza
        assert re.search(r"^\s*throw ", bloque, re.MULTILINE)


# ---------------------------------------------------------------------------
# Y ademas se corre de verdad. Leer el .ps1 no prueba que PowerShell lo ejecute
# como uno cree -- en este repo ya paso que el texto se veia perfecto y el
# script fallaba en la instalacion real.
# ---------------------------------------------------------------------------

pytestmark_ps = pytest.mark.skipif(
    sys.platform != "win32" or shutil.which("powershell") is None,
    reason="Hace falta Windows PowerShell, que es el que corre en la instalacion",
)


def preparar_raiz(tmp_path: Path, *, vc=True, vocoder=True, xvector=True) -> Path:
    (tmp_path / "scripts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONVERSION_DE_VOZ, tmp_path / "scripts" / CONVERSION_DE_VOZ.name)
    vendor = tmp_path / "vendor"
    if vc:
        (vendor / "speecht5-vc").mkdir(parents=True, exist_ok=True)
        (vendor / "speecht5-vc" / "config.json").write_text("{}", encoding="utf-8")
    if vocoder:
        (vendor / "speecht5-hifigan").mkdir(parents=True, exist_ok=True)
        (vendor / "speecht5-hifigan" / "config.json").write_text("{}", encoding="utf-8")
    if xvector:
        (vendor / "xvector").mkdir(parents=True, exist_ok=True)
        (vendor / "xvector" / "tdnn.onnx").write_bytes(b"x")
    return tmp_path


def entorno_con_python_del_repo() -> dict[str, str] | None:
    """PATH con el venv del repo adelante, para que `python` sea ESE python.

    La raiz temporal no tiene .venv, asi que el script cae a `python` del PATH.
    Si ahi no hay ninguno, la prueba pasaria por el motivo equivocado (comando no
    encontrado) en vez de por el chequeo del codigo de salida.

    Se prepara el PATH en vez de enlazar el venv: un junction dentro de tmp_path
    lo borra despues la limpieza de pytest, y ahi el riesgo es que se lleve
    puesto el venv de verdad.
    """
    scripts = RAIZ / ".venv" / "Scripts"
    if not (scripts / "python.exe").exists():
        return None
    entorno = dict(os.environ)
    entorno["PATH"] = f"{scripts}{os.pathsep}{entorno.get('PATH', '')}"
    entorno["HF_HUB_OFFLINE"] = "1"
    return entorno


def correr(raiz: Path, *, stub_descarga: str = "") -> subprocess.CompletedProcess:
    script = raiz / "scripts" / CONVERSION_DE_VOZ.name
    # La descarga se sustituye definiendo una funcion con el nombre del cmdlet en
    # la sesion que invoca al script: en PowerShell la funcion gana. Asi se
    # ejercita el script REAL sin tocar la red.
    comando = f"{stub_descarga}\n& '{script}'"
    return subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", comando],
        capture_output=True,
        timeout=180,
    )


STUB_QUE_BAJA_BASURA = (
    "function Invoke-WebRequest { param([string]$Uri,[string]$OutFile,[switch]$UseBasicParsing) "
    "Set-Content -LiteralPath $OutFile -Value 'no soy un modelo' }"
)


@pytestmark_ps
class TestElScriptCorridoDeVerdad:
    def test_con_las_tres_piezas_termina_en_cero(self, tmp_path: Path) -> None:
        resultado = correr(preparar_raiz(tmp_path))

        assert resultado.returncode == 0, resultado.stderr.decode("utf-8", "replace")

    def test_una_descarga_corrupta_del_xvector_falla_y_no_deja_el_archivo(
        self, tmp_path: Path
    ) -> None:
        # El fallo caro: un x-vector corrupto carga igual y devuelve embeddings
        # fuera del espacio, con lo que la conversion sale en NaN. Dejarlo en su
        # sitio seria peor que no bajarlo.
        raiz = preparar_raiz(tmp_path, xvector=False)

        resultado = correr(raiz, stub_descarga=STUB_QUE_BAJA_BASURA)

        assert resultado.returncode != 0
        assert not (raiz / "vendor" / "xvector" / "tdnn.onnx").exists()
        salida = (resultado.stderr + resultado.stdout).decode("utf-8", "replace")
        assert "bytes" in salida or "sha256" in salida

    @pytest.mark.parametrize("falta", ["vc", "vocoder"])
    def test_sin_speecht5_y_sin_poder_bajarlo_falla(self, tmp_path: Path, falta: str) -> None:
        # El script no miraba el codigo de salida del python incrustado: una
        # descarga fallida de SpeechT5 seguia de largo. Se le corta la red a
        # huggingface_hub para que falle de verdad.
        pytest.importorskip("huggingface_hub")
        entorno = entorno_con_python_del_repo()
        if entorno is None:
            pytest.skip("No hay venv del repo del que tomar python")
        raiz = preparar_raiz(tmp_path, **{falta: False})

        script = raiz / "scripts" / CONVERSION_DE_VOZ.name
        resultado = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
            capture_output=True,
            timeout=300,
            env=entorno,
        )

        assert resultado.returncode != 0
        salida = (resultado.stderr + resultado.stdout).decode("utf-8", "replace")
        assert "SpeechT5" in salida, salida[-800:]
