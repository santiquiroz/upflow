from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi import HTTPException

from app.api.routes import synthesize_speech, tts_capabilities
from app.config import Settings
from app.schemas import SynthesizeSpeechRequest
from app.services.missing_pack import PACK_LABELS

# ---------------------------------------------------------------------------
# La pantalla de Voz mostraba "Instalalo con scripts/download-kokoro-tts.ps1" y
# nada mas: ni boton, ni forma de saber QUE bajar.
#
# Una frase no le alcanza al frontend. Para poder ofrecer la descarga necesita el
# NOMBRE del paquete, asi que viaja en la respuesta como un campo aparte y no
# escondido dentro del texto.
# ---------------------------------------------------------------------------


class MotorSinModelo:
    def available(self, model_dir: Path) -> bool:
        return False

    def synthesize(self, *, model_dir: Path, phonemes: str, voice: str):
        return np.zeros(8, dtype=np.float32)


def settings_de(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, RUNTIME_DIR=str(tmp_path))


@pytest.mark.asyncio
async def test_las_capacidades_nombran_el_paquete_que_falta(tmp_path: Path):
    respuesta = await tts_capabilities(
        engine=MotorSinModelo(),
        settings_dep=settings_de(tmp_path),
        model_dir=tmp_path / "no-esta",
    )

    assert respuesta.available is False
    assert respuesta.missing_pack == "kokoro"


@pytest.mark.asyncio
async def test_el_motivo_explica_que_falta_sin_dictar_un_comando(tmp_path: Path):
    respuesta = await tts_capabilities(
        engine=MotorSinModelo(),
        settings_dep=settings_de(tmp_path),
        model_dir=tmp_path / "no-esta",
    )

    assert PACK_LABELS["kokoro"] in respuesta.reason
    assert ".ps1" not in respuesta.reason


@pytest.mark.asyncio
async def test_con_el_modelo_puesto_no_sobra_ningun_paquete(tmp_path: Path):
    # El campo tiene que estar vacio cuando no falta nada: si quedara pegado, la
    # pantalla ofreceria descargar algo que ya esta.
    modelo = tmp_path / "kokoro"
    (modelo / "voices").mkdir(parents=True)
    (modelo / "voices" / "af_heart.bin").write_bytes(b"")

    class MotorListo(MotorSinModelo):
        def available(self, model_dir: Path) -> bool:
            return True

    respuesta = await tts_capabilities(
        engine=MotorListo(), settings_dep=settings_de(tmp_path), model_dir=modelo
    )

    assert respuesta.available is True
    assert respuesta.missing_pack is None


@pytest.mark.asyncio
async def test_sintetizar_sin_modelo_tambien_dice_que_paquete_falta(tmp_path: Path):
    # El 409 llega al banner de error de la pantalla. Sin el paquete ahi, la
    # pantalla puede explicar el problema pero no ofrecer la solucion.
    with pytest.raises(HTTPException) as capturado:
        await synthesize_speech(
            payload=SynthesizeSpeechRequest(text="hola", voice="af_heart"),
            engine=MotorSinModelo(),
            settings_dep=settings_de(tmp_path),
            model_dir=tmp_path / "no-esta",
        )

    assert capturado.value.status_code == 409
    assert capturado.value.detail["missingPack"] == "kokoro"
    assert ".ps1" not in capturado.value.detail["reason"]
