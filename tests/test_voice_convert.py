from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services.engines.voice_convert import (
    MAX_SECONDS,
    SAMPLE_RATE,
    VoiceConversionEngine,
    VoiceConversionUnavailable,
    assert_convertible,
    assert_usable_audio,
)
from app.services.missing_pack import PACK_LABELS

# ---------------------------------------------------------------------------
# El fallo caro de estos modelos no es una excepcion: es audio con la duracion
# correcta que no sirve. Ya paso dos veces en este repo (la variante fp16 de
# Kokoro devuelve NaN, y un export de x-vector de terceros devolvia 52 s de
# ruido). Por eso el audio se revisa antes de entregarlo.
# ---------------------------------------------------------------------------


def make_models(tmp_path: Path, *, completo: bool = True) -> Path:
    (tmp_path / "speecht5-vc").mkdir(parents=True)
    (tmp_path / "speecht5-hifigan").mkdir(parents=True)
    (tmp_path / "xvector").mkdir(parents=True)
    if completo:
        (tmp_path / "xvector" / "tdnn.onnx").write_bytes(b"no-es-un-modelo")
    return tmp_path


class TestOutputGuard:
    def test_nan_audio_is_refused(self) -> None:
        with pytest.raises(VoiceConversionUnavailable, match="NaN"):
            assert_usable_audio(np.array([0.1, np.nan], dtype=np.float32))

    def test_empty_output_is_refused(self) -> None:
        with pytest.raises(VoiceConversionUnavailable, match="vac"):
            assert_usable_audio(np.array([], dtype=np.float32))

    def test_real_audio_passes(self) -> None:
        assert_usable_audio(np.array([0.1, -0.2], dtype=np.float32))


class TestInputGuard:
    def test_empty_input_is_refused_before_loading_the_model(self) -> None:
        with pytest.raises(VoiceConversionUnavailable):
            assert_convertible(np.array([], dtype=np.float32))

    def test_audio_over_the_ceiling_says_how_long_it_is(self) -> None:
        largo = np.zeros(SAMPLE_RATE * (MAX_SECONDS + 5), dtype=np.float32)
        with pytest.raises(VoiceConversionUnavailable, match=str(MAX_SECONDS)):
            assert_convertible(largo)

    def test_audio_within_the_ceiling_passes(self) -> None:
        assert_convertible(np.zeros(SAMPLE_RATE * 5, dtype=np.float32))


class TestAvailability:
    def test_needs_the_xvector_encoder_too(self, tmp_path: Path) -> None:
        # Sin el encoder no se puede sacar la voz de la muestra, asi que la
        # capacidad NO esta disponible aunque el modelo de conversion este.
        assert VoiceConversionEngine(make_models(tmp_path, completo=False)).available() is False

    def test_available_with_every_piece_in_place(self, tmp_path: Path) -> None:
        assert VoiceConversionEngine(make_models(tmp_path)).available() is True

    def test_dice_que_paquete_falta_sin_dictar_comandos(self, tmp_path: Path) -> None:
        # Identificar QUE falta es el contrato; la redaccion exacta no lo es.
        # Nombrar un script es la regresion que hay que evitar: el usuario no
        # tiene por que ir a una terminal, la app baja el paquete sola.
        engine = VoiceConversionEngine(tmp_path / "vacio")
        with pytest.raises(VoiceConversionUnavailable) as error:
            engine._load()
        assert PACK_LABELS["voice-conversion"] in str(error.value)
        assert ".ps1" not in str(error.value)


# --- API -------------------------------------------------------------------


def test_capabilities_name_the_missing_pack() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        body = client.get("/api/v1/voice/conversion/capabilities").json()

    # En una maquina sin el modelo bajado, la respuesta tiene que decir QUE
    # paquete falta — no solo negar, y tampoco dictar un comando: con el nombre
    # del paquete la pantalla puede ofrecer el boton de descarga.
    assert "maxSeconds" in body
    if not body["available"]:
        assert body["missingPack"] == "voice-conversion"
        assert ".ps1" not in (body["reason"] or "")


def test_converting_without_the_model_is_a_409_naming_the_pack(tmp_path) -> None:
    import io as _io

    import soundfile
    from fastapi.testclient import TestClient

    from app.api.routes import get_voice_conversion
    from app.main import app
    from app.services.engines.voice_convert import VoiceConversionEngine

    app.dependency_overrides[get_voice_conversion] = lambda: VoiceConversionEngine(tmp_path)
    try:
        buffer = _io.BytesIO()
        soundfile.write(buffer, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/voice/conversion",
                files={
                    "source": ("a.wav", buffer.getvalue(), "audio/wav"),
                    "reference": ("b.wav", buffer.getvalue(), "audio/wav"),
                },
            )
        assert response.status_code == 409
        detalle = response.json()["detail"]
        assert detalle["missingPack"] == "voice-conversion"
        assert ".ps1" not in detalle["reason"]
    finally:
        app.dependency_overrides.pop(get_voice_conversion, None)
