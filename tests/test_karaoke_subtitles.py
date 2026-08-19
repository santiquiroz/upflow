"""El resaltado palabra por palabra: lo que separa un karaoke de un video con subtitulos.

Casi todo lo que se prueba aca son unidades y bordes que NO fallan ruidosamente:
si la duracion va en la unidad equivocada, el archivo es valido y el resaltado
corre 10x. Nada revienta, solo queda mal.
"""

from __future__ import annotations

import pytest

from app.services.karaoke_subtitles import (
    KaraokeLine,
    KaraokeWord,
    ass_timestamp,
    karaoke_durations,
    line_to_dialogue,
    render_karaoke_ass,
    split_line_proportionally,
)


def palabra(texto: str, start: float, end: float) -> KaraokeWord:
    return KaraokeWord(text=texto, start=start, end=end)


# ---------------------------------------------------------------------------
# Formato de tiempo
# ---------------------------------------------------------------------------


def test_ass_usa_centesimas_y_una_cifra_de_hora() -> None:
    # `0:00:01.50`, no `00:00:01,500` (eso es SRT). Un formato de mas cifras no
    # falla: libass lo lee mal y los tiempos quedan corridos.
    assert ass_timestamp(1.5) == "0:00:01.50"


def test_las_horas_no_se_truncan() -> None:
    assert ass_timestamp(3661.25) == "1:01:01.25"


def test_un_tiempo_negativo_se_recorta_a_cero() -> None:
    # Puede salir de restarle un offset a un segmento del primer chunk.
    assert ass_timestamp(-2.0) == "0:00:00.00"


def test_el_redondeo_no_produce_cien_centesimas() -> None:
    # 1.999 redondea a 100 centesimas, que no existe: seria "0:00:01.100".
    assert ass_timestamp(1.999) == "0:00:02.00"


# ---------------------------------------------------------------------------
# Duraciones de \k
# ---------------------------------------------------------------------------


def test_la_duracion_va_en_centisegundos() -> None:
    duraciones = karaoke_durations([palabra("hola", 0.0, 1.0)], line_start=0.0)

    # Si fueran milisegundos daria 1000 y el resaltado tardaria 10 s en cruzar
    # una palabra de 1 s. El archivo seria valido igual.
    assert duraciones == [100]


def test_el_silencio_previo_se_le_suma_a_la_palabra() -> None:
    # `\k` no sabe esperar: si el hueco se descarta, el resaltado se adelanta y
    # termina antes que la voz.
    duraciones = karaoke_durations([palabra("hola", 0.5, 1.0)], line_start=0.0)

    assert duraciones == [100]


def test_ninguna_duracion_es_cero() -> None:
    duraciones = karaoke_durations(
        [palabra("a", 0.0, 0.0), palabra("b", 0.0, 0.001)], line_start=0.0
    )

    # `\k0` no avanza: libass pinta esa palabra de golpe.
    assert all(d >= 1 for d in duraciones)


def test_las_duraciones_siguen_el_orden_de_las_palabras() -> None:
    duraciones = karaoke_durations(
        [palabra("uno", 0.0, 0.2), palabra("dos", 0.2, 1.0)], line_start=0.0
    )

    assert duraciones == [20, 80]


# ---------------------------------------------------------------------------
# La linea de dialogo
# ---------------------------------------------------------------------------


def test_cada_palabra_lleva_su_etiqueta() -> None:
    linea = KaraokeLine(
        start=0.0, end=1.0, words=(palabra("hola", 0.0, 0.5), palabra("mundo", 0.5, 1.0))
    )

    dialogo = line_to_dialogue(linea)

    assert "{\\k50}hola" in dialogo
    assert "{\\k50}mundo" in dialogo


def test_una_linea_sin_palabras_no_emite_dialogo() -> None:
    # Un Dialogue vacio es una linea en blanco parpadeando en pantalla.
    assert line_to_dialogue(KaraokeLine(start=0.0, end=1.0, words=())) == ""


def test_el_dialogo_lleva_los_tiempos_de_la_linea() -> None:
    linea = KaraokeLine(start=2.0, end=4.0, words=(palabra("x", 2.0, 4.0),))

    dialogo = line_to_dialogue(linea)

    assert "0:00:02.00" in dialogo
    assert "0:00:04.00" in dialogo


# ---------------------------------------------------------------------------
# El archivo entero
# ---------------------------------------------------------------------------


def test_el_archivo_declara_la_resolucion() -> None:
    ass = render_karaoke_ass([], width=1920, height=1080)

    # libass escala la fuente respecto de PlayResX/Y: sin declararla, la letra
    # sale de otro tamaño segun el video.
    assert "PlayResX: 1920" in ass
    assert "PlayResY: 1080" in ass


def test_el_archivo_tiene_las_tres_secciones_que_pide_ass() -> None:
    ass = render_karaoke_ass([KaraokeLine(0.0, 1.0, (palabra("a", 0.0, 1.0),))])

    for seccion in ("[Script Info]", "[V4+ Styles]", "[Events]"):
        assert seccion in ass


def test_el_estilo_que_usan_los_dialogos_existe() -> None:
    ass = render_karaoke_ass([KaraokeLine(0.0, 1.0, (palabra("a", 0.0, 1.0),))])

    # Un Dialogue que nombra un estilo inexistente se dibuja con el default y se
    # pierde el borde: ilegible sobre fondo claro.
    assert "Style: Karaoke," in ass
    assert ",Karaoke,,0,0,0,," in ass


# ---------------------------------------------------------------------------
# El reparto aproximado
# ---------------------------------------------------------------------------


def test_reparte_la_linea_entre_sus_palabras() -> None:
    linea = split_line_proportionally("hola mundo", 0.0, 2.0)

    assert [p.text for p in linea.words] == ["hola", "mundo"]
    assert linea.words[0].start == 0.0
    assert linea.words[-1].end == pytest.approx(2.0)


def test_una_palabra_mas_larga_recibe_mas_tiempo() -> None:
    linea = split_line_proportionally("a considerable", 0.0, 2.0)

    corta, larga = linea.words
    assert (larga.end - larga.start) > (corta.end - corta.start)


def test_las_palabras_no_dejan_huecos_entre_si() -> None:
    linea = split_line_proportionally("uno dos tres", 1.0, 4.0)

    for anterior, siguiente in zip(linea.words, linea.words[1:]):
        assert siguiente.start == pytest.approx(anterior.end)


def test_una_linea_vacia_no_inventa_palabras() -> None:
    assert split_line_proportionally("   ", 0.0, 1.0).words == ()
