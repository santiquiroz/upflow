"""Conversión de timbre con OpenVoice: el contrato y lo que no falla ruidosamente.

Casi todo lo delicado de este motor son detalles de señal que NO tiran error: una
ventana simétrica en vez de periódica, o un relleno con ceros, devuelven audio con
la voz apenas corrida o metálica, y eso se le atribuye al modelo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services.engines.openvoice_convert import (
    DEFAULT_TAU,
    LATENT_CHANNELS,
    N_FFT,
    SAMPLE_RATE,
    OpenVoiceConversionEngine,
    hann_window,
    spectrogram,
)
from app.services.engines.voice_convert import (
    VoiceConversionUnavailable,
    assert_convertible,
)


def audio(muestras: int = SAMPLE_RATE) -> np.ndarray:
    return (np.random.default_rng(5).standard_normal(muestras) * 0.2).astype(np.float32)


# ---------------------------------------------------------------------------
# La señal
# ---------------------------------------------------------------------------


def test_la_ventana_es_periodica_y_no_la_de_numpy() -> None:
    ventana = hann_window(8)

    # `np.hanning` da la simétrica, que cierra en cero. La periódica —la que usa
    # torch, y con la que se entrenó el modelo— no. Esa muestra de diferencia
    # basta para que la voz salga metálica, y las dos son "una ventana de Hann".
    assert ventana[-1] != pytest.approx(0.0)
    assert not np.allclose(ventana, np.hanning(8))


def test_el_espectrograma_sale_en_la_forma_que_come_el_grafo() -> None:
    spec = spectrogram(audio())

    assert spec.shape[0] == 1
    assert spec.shape[1] == N_FFT // 2 + 1
    assert spec.dtype == np.float32


def test_el_estereo_se_rechaza_en_vez_de_mezclarse_solo() -> None:
    # Mezclar en silencio es tomar una decisión sobre el audio de otro.
    with pytest.raises(ValueError):
        spectrogram(np.zeros((1000, 2), dtype=np.float32))


def test_un_clip_mas_corto_que_un_cuadro_lo_dice() -> None:
    with pytest.raises(VoiceConversionUnavailable) as error:
        spectrogram(np.zeros(50, dtype=np.float32))

    assert "corto" in str(error.value)


def test_la_magnitud_nunca_es_exactamente_cero() -> None:
    # El +1e-6 viene del modelo original; se conserva para que el número sea el
    # mismo que vio el entrenamiento.
    assert (spectrogram(np.zeros(N_FFT * 3, dtype=np.float32)) > 0).all()


# ---------------------------------------------------------------------------
# La duración máxima se mide con la frecuencia del motor
# ---------------------------------------------------------------------------


def test_el_limite_de_duracion_usa_la_frecuencia_de_este_motor() -> None:
    # OpenVoice trabaja a 22050 y SpeechT5 a 16000. Con el valor fijo del motor
    # viejo, un audio de 22050 contaba como si durara 1.38x más y se rechazaban
    # clips que entraban de sobra.
    un_segundo = np.zeros(SAMPLE_RATE, dtype=np.float32)

    assert_convertible(un_segundo, sample_rate=SAMPLE_RATE)


def test_un_audio_vacio_se_rechaza() -> None:
    with pytest.raises(VoiceConversionUnavailable):
        assert_convertible(np.zeros(0, dtype=np.float32), sample_rate=SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Instalación
# ---------------------------------------------------------------------------


def test_sin_los_dos_grafos_no_esta_disponible(tmp_path: Path) -> None:
    motor = OpenVoiceConversionEngine(tmp_path)
    assert not motor.available()

    base = tmp_path / "openvoice"
    base.mkdir()
    (base / "openvoice_converter.onnx").write_bytes(b"x")

    # Con uno solo tampoco: decir "disponible" acá manda a diagnosticar la
    # conversión en vez de la instalación.
    assert not motor.available()
    (base / "openvoice_speaker.onnx").write_bytes(b"x")
    assert motor.available()


def test_convertir_sin_instalar_nombra_el_pack(tmp_path: Path) -> None:
    motor = OpenVoiceConversionEngine(tmp_path)

    with pytest.raises(VoiceConversionUnavailable) as error:
        motor.convert(source=audio(), reference=audio())

    assert "openvoice" in str(error.value).lower()


# ---------------------------------------------------------------------------
# Constantes que tienen que coincidir con el grafo
# ---------------------------------------------------------------------------


def test_los_canales_del_ruido_son_los_del_latente() -> None:
    # Si no coincide con `inter_channels` del modelo, el grafo rechaza la entrada.
    assert LATENT_CHANNELS == 192


def test_el_tau_por_defecto_es_el_de_openvoice() -> None:
    assert DEFAULT_TAU == 0.3


def test_la_frecuencia_es_la_del_modelo_y_no_la_del_motor_viejo() -> None:
    from app.services.engines.voice_convert import SAMPLE_RATE as SPEECHT5_RATE

    assert SAMPLE_RATE == 22050
    assert SAMPLE_RATE != SPEECHT5_RATE
