from __future__ import annotations

import pytest

from app.services.engines.transcribe_onnx import is_english_only

from tests.test_transcribe_job_manager import make_manager, register_asr_model

# ---------------------------------------------------------------------------
# Los modelos Whisper que terminan en `.en` entienden UN idioma: ingles. Pedirles
# otro no es una preferencia que se pueda ignorar — la libreria lo rechaza:
#
#   Cannot specify `task` or `language` for an English-only model.
#
# Reportado el 2026-08-06 desde la instalacion real, con un audio en castellano y
# `whisper-tiny.en_timestamped` elegido. Y la app EMPUJABA a ese error: su propio
# consejo de "modelo probado" recomendaba justamente un modelo solo-ingles,
# mientras el desplegable de idioma venia en espanol.
#
# Tragarse el idioma en vez de fallar seria peor: el modelo devolveria fonemas
# castellanos escritos como si fueran ingles, y el usuario recibiria basura sin
# saber por que.
# ---------------------------------------------------------------------------


class TestReconocerLosSoloIngles:
    @pytest.mark.parametrize(
        "model_id",
        [
            "openai/whisper-tiny.en",
            "onnx-community/whisper-tiny.en_timestamped",
            "openai/whisper-small.en",
            "openai/whisper-base.en",
            "WHISPER-TINY.EN",
        ],
    )
    def test_los_que_terminan_en_punto_en_son_solo_ingles(self, model_id: str) -> None:
        assert is_english_only(model_id) is True

    @pytest.mark.parametrize(
        "model_id",
        [
            "openai/whisper-large-v3",
            "onnx-community/whisper-tiny",
            "openai/whisper-large-v3-turbo",
            "onnx-community/whisper-base_timestamped",
        ],
    )
    def test_los_multilingues_no_lo_son(self, model_id: str) -> None:
        assert is_english_only(model_id) is False

    def test_una_palabra_que_contiene_en_no_confunde(self) -> None:
        # `.en` tiene que ser el sufijo del nombre, no aparecer en cualquier lado:
        # marcar de mas dejaria a un modelo multilingue sin poder elegir idioma.
        assert is_english_only("empresa/whisper-entrenado") is False
        assert is_english_only("openai/whisper-tiny.english-ish") is False


class TestElTrabajoSeRechazaAntesDeEmpezar:
    """Mejor rechazar al crear el trabajo que a mitad de la transcripcion.

    El usuario ya subio el archivo y eligio todo; enterarse recien cuando el
    motor explota, con un mensaje en ingles de una libreria, no le dice que
    hacer.
    """

    @pytest.mark.asyncio
    async def test_pedir_espanol_a_un_modelo_solo_ingles_se_rechaza(self, tmp_path):
        manager, settings, registry = make_manager(tmp_path)
        register_asr_model(registry, settings, "whisper-tiny.en_timestamped")
        fuente = tmp_path / "audio.wav"
        fuente.write_bytes(b"RIFF")

        with pytest.raises(ValueError, match="ingl"):
            await manager.create_job(
                source_path=fuente,
                original_filename="audio.wav",
                model_id="whisper-tiny.en_timestamped",
                language="es",
            )

    @pytest.mark.asyncio
    async def test_el_mismo_modelo_en_ingles_se_acepta(self, tmp_path):
        manager, settings, registry = make_manager(tmp_path)
        register_asr_model(registry, settings, "whisper-tiny.en_timestamped")
        fuente = tmp_path / "audio.wav"
        fuente.write_bytes(b"RIFF")

        job = await manager.create_job(
            source_path=fuente,
            original_filename="audio.wav",
            model_id="whisper-tiny.en_timestamped",
            language="en",
        )

        assert job is not None

    @pytest.mark.asyncio
    async def test_sin_idioma_elegido_no_se_rechaza_nada(self, tmp_path):
        # Sin idioma el modelo lo detecta solo, asi que no hay conflicto que
        # denunciar: rechazarlo seria inventar un problema.
        manager, settings, registry = make_manager(tmp_path)
        register_asr_model(registry, settings, "whisper-tiny.en_timestamped")
        fuente = tmp_path / "audio.wav"
        fuente.write_bytes(b"RIFF")

        job = await manager.create_job(
            source_path=fuente,
            original_filename="audio.wav",
            model_id="whisper-tiny.en_timestamped",
        )

        assert job is not None
