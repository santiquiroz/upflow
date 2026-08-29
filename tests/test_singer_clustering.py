"""F2a — cantantes por linea: embeddings, clustering y mute por tiempo.

El fallo silencioso que guarda este archivo: etiquetas INESTABLES. Si el mismo
audio sale a veces con s1 y s2 intercambiados (segun el orden interno de scipy),
los colores y el mute elegidos en review le pegan al cantante equivocado y nada
revienta — el video sale, mal. Y un gate sin crossfade tampoco revienta: solo
mete clicks en cada corte.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.engines.singer_clustering import (
    CROSSFADE_SECONDS,
    SINGER_COUNT_MAX,
    SINGER_COUNT_MIN,
    cluster_singers,
    mute_time_spans,
    singer_label,
)
from app.services.engines.singer_embedding import (
    EMBEDDING_SAMPLE_RATE,
    MIN_LINE_SECONDS,
    line_embeddings,
    resample_for_embedding,
    to_mono,
)
from app.services.xvector import XvectorUnavailable

SR = 44100


def versor(*componentes: float) -> np.ndarray:
    v = np.asarray(componentes, dtype=np.float32)
    return v / np.linalg.norm(v)


def spans_secuenciales(n: int) -> list[tuple[float, float]]:
    return [(float(i), float(i) + 1.0) for i in range(n)]


A = versor(1.0, 0.0, 0.0)
B = versor(0.0, 1.0, 0.0)
C = versor(0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Clustering: etiquetas estables s1..sN
# ---------------------------------------------------------------------------


def test_lineas_alternadas_alternan_etiquetas():
    etiquetas = cluster_singers([A, B, A, B], spans_secuenciales(4), 2)

    assert etiquetas == ["s1", "s2", "s1", "s2"]


def test_s1_es_siempre_el_cantante_de_la_primera_linea():
    # La estabilidad ES el contrato: el id no depende del orden interno del
    # clustering sino de quien aparece primero en la cancion.
    etiquetas = cluster_singers([B, A, B], spans_secuenciales(3), 2)

    assert etiquetas == ["s1", "s2", "s1"]


def test_el_ruido_pequeno_no_parte_un_cantante_en_dos():
    rng = np.random.default_rng(7)
    ruidosas = [versor(*(base + rng.normal(0, 0.02, 3))) for base in (A, B, A, A, B, A)]

    etiquetas = cluster_singers(ruidosas, spans_secuenciales(6), 2)

    assert etiquetas == ["s1", "s2", "s1", "s1", "s2", "s1"]


def test_una_linea_sin_embedding_hereda_el_vecino_mas_cercano_en_tiempo():
    spans = [(0.0, 1.0), (1.1, 1.4), (5.0, 6.0)]

    etiquetas = cluster_singers([A, None, B], spans, 2)

    # La del medio esta pegada a la primera, no a la de los 5 segundos.
    assert etiquetas == ["s1", "s1", "s2"]


def test_sin_ningun_embedding_todo_es_el_primer_cantante():
    etiquetas = cluster_singers([None, None], spans_secuenciales(2), 2)

    assert etiquetas == ["s1", "s1"]


def test_un_solo_embedding_valido_no_intenta_clusterizar():
    etiquetas = cluster_singers([None, A], spans_secuenciales(2), 2)

    assert etiquetas == ["s1", "s1"]


def test_tres_cantantes_dan_tres_etiquetas_en_orden_de_aparicion():
    etiquetas = cluster_singers([C, A, B, C], spans_secuenciales(4), 3)

    assert etiquetas == ["s1", "s2", "s3", "s1"]


def test_pedir_mas_cantantes_que_lineas_no_inventa_clusters():
    etiquetas = cluster_singers([A, B], spans_secuenciales(2), SINGER_COUNT_MAX)

    assert etiquetas == ["s1", "s2"]


def test_largos_desparejos_se_rechazan():
    with pytest.raises(ValueError):
        cluster_singers([A, B], spans_secuenciales(3), 2)


def test_sin_lineas_no_hay_etiquetas():
    assert cluster_singers([], [], 2) == []


def test_singer_label_arranca_en_s1():
    assert singer_label(0) == "s1"
    assert SINGER_COUNT_MIN == 2 and SINGER_COUNT_MAX == 4


# ---------------------------------------------------------------------------
# Mute por tiempo con crossfade
# ---------------------------------------------------------------------------


def senal(segundos: float = 3.0) -> np.ndarray:
    return np.ones(int(SR * segundos), dtype=np.float32)


def test_el_tramo_muteado_queda_en_silencio_completo():
    silenciado = mute_time_spans(senal(), SR, [(1.0, 2.0)])

    assert np.all(silenciado[int(1.0 * SR) : int(2.0 * SR)] == 0.0)


def test_lejos_del_tramo_el_audio_no_se_toca():
    silenciado = mute_time_spans(senal(), SR, [(1.0, 2.0)])

    assert silenciado[int(0.5 * SR)] == 1.0
    assert silenciado[int(2.5 * SR)] == 1.0


def test_el_crossfade_no_deja_escalones():
    silenciado = mute_time_spans(senal(), SR, [(1.0, 2.0)])

    # Un salto mayor al paso de la rampa es un click audible en cada corte.
    paso_maximo = 1.0 / (CROSSFADE_SECONDS * SR)
    assert float(np.max(np.abs(np.diff(silenciado)))) <= paso_maximo + 1e-9


def test_estereo_conserva_forma_y_mutea_los_dos_canales():
    estereo = np.stack([senal(), senal() * 0.5], axis=1)

    silenciado = mute_time_spans(estereo, SR, [(1.0, 2.0)])

    assert silenciado.shape == estereo.shape
    assert np.all(silenciado[int(1.2 * SR), :] == 0.0)


def test_tramos_al_borde_del_archivo_no_revientan():
    silenciado = mute_time_spans(senal(), SR, [(0.0, 0.5), (2.9, 5.0)])

    assert silenciado[0] == 0.0
    assert silenciado[-1] == 0.0
    assert silenciado[int(1.5 * SR)] == 1.0


def test_tramos_solapados_se_combinan():
    silenciado = mute_time_spans(senal(), SR, [(1.0, 2.0), (1.5, 2.5)])

    assert np.all(silenciado[int(1.0 * SR) : int(2.5 * SR)] == 0.0)


def test_sin_tramos_devuelve_una_copia_intacta():
    original = senal()

    silenciado = mute_time_spans(original, SR, [])

    np.testing.assert_array_equal(silenciado, original)
    assert silenciado is not original  # inmutable: el wav de voces no se pisa


# ---------------------------------------------------------------------------
# Embeddings por linea (sin ONNX: el encoder es un fake)
# ---------------------------------------------------------------------------


class EncoderQueRegistra:
    def __init__(self) -> None:
        self.ventanas: list[int] = []

    def available(self) -> bool:
        return True

    def encode(self, audio: np.ndarray) -> np.ndarray:
        self.ventanas.append(len(audio))
        return versor(1.0, 0.0)


class EncoderQueFalla:
    def available(self) -> bool:
        return True

    def encode(self, audio: np.ndarray) -> np.ndarray:
        raise XvectorUnavailable("no hay voz ahi")


def test_line_embeddings_remuestrea_y_corta_cada_linea():
    encoder = EncoderQueRegistra()
    audio = np.zeros((SR * 2, 2), dtype=np.float32)

    resultados = line_embeddings(encoder, audio, SR, [(0.0, 1.0), (1.0, 2.0)])

    assert all(r is not None for r in resultados)
    # Cada ventana llega en la frecuencia del encoder, no en la del separador.
    for muestras in encoder.ventanas:
        assert muestras == pytest.approx(EMBEDDING_SAMPLE_RATE, rel=0.01)


def test_una_linea_demasiado_corta_no_se_embebe():
    encoder = EncoderQueRegistra()
    audio = np.zeros(SR * 2, dtype=np.float32)
    corta = (0.0, MIN_LINE_SECONDS / 2)

    resultados = line_embeddings(encoder, audio, SR, [corta, (1.0, 2.0)])

    assert resultados[0] is None
    assert resultados[1] is not None


def test_si_el_encoder_no_saca_voz_la_linea_queda_sin_embedding():
    # Una linea de puro silencio no tiene voz que extraer: eso NO es un error
    # del trabajo, es una linea que hereda cantante del vecino.
    audio = np.zeros(SR * 2, dtype=np.float32)

    resultados = line_embeddings(EncoderQueFalla(), audio, SR, [(0.0, 1.0)])

    assert resultados == [None]


def test_to_mono_promedia_canales_y_resample_conserva_duracion():
    estereo = np.stack([np.ones(SR), np.zeros(SR)], axis=1).astype(np.float32)

    mono = to_mono(estereo)
    remuestreado = resample_for_embedding(mono, SR)

    assert mono.ndim == 1
    assert np.allclose(mono, 0.5)
    assert len(remuestreado) == EMBEDDING_SAMPLE_RATE
