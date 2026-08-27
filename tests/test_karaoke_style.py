import pytest

from app.services.karaoke_subtitles import (
    KaraokeLine,
    KaraokeStyle,
    KaraokeWord,
    build_style_lines,
    hex_to_ass_color,
    render_karaoke_ass,
)


def linea(texto: str = "hola mundo") -> KaraokeLine:
    return KaraokeLine(
        start=0.0,
        end=2.0,
        words=(
            KaraokeWord(text=texto.split()[0], start=0.0, end=1.0),
            KaraokeWord(text=texto.split()[1], start=1.0, end=2.0),
        ),
    )


def test_hex_a_ass_invierte_los_canales_y_suma_alfa():
    assert hex_to_ass_color("#FF8000") == "&H000080FF"


def test_hex_invalido_se_rechaza():
    with pytest.raises(ValueError):
        hex_to_ass_color("rojo")


def test_estilo_default_replica_el_historico():
    principal, traduccion = build_style_lines(KaraokeStyle())

    assert principal.startswith("Style: Karaoke,Arial,48,&H00FFFFFF,&H0000FFFF,")
    assert principal.endswith(",2,40,40,60,1")
    assert traduccion.startswith("Style: Translation,Arial,34,")


def test_tamano_y_posicion_cambian_fuente_y_alineacion():
    principal, _ = build_style_lines(KaraokeStyle(size="large", position="top"))

    assert ",Arial,64," in principal
    assert ",8,40,40,40,1" in principal


def test_arriba_la_traduccion_baja_debajo_de_la_principal():
    _, traduccion = build_style_lines(KaraokeStyle(size="medium", position="top"))

    # 40 de margen + 48 de fuente principal + 12 de aire.
    assert traduccion.endswith(",8,40,40,100,1")


def test_tamano_desconocido_se_rechaza():
    with pytest.raises(ValueError):
        build_style_lines(KaraokeStyle(size="giant"))


def test_posicion_desconocida_se_rechaza():
    with pytest.raises(ValueError):
        build_style_lines(KaraokeStyle(position="center"))


def test_render_sin_traducciones_no_emite_estilo_translation_en_eventos():
    ass = render_karaoke_ass([linea()])

    assert "Style: Translation," in ass
    assert ",Translation,,0,0,0,," not in ass


def test_traduccion_acompana_con_los_tiempos_de_la_linea_original():
    ass = render_karaoke_ass([linea()], translations=["hello world"])

    eventos = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(eventos) == 2
    original, traducida = eventos
    assert ",Karaoke,,0" in original
    assert ",Translation,,0" in traducida
    # Mismos tiempos de entrada y salida que la linea original.
    assert traducida.split(",")[1:3] == original.split(",")[1:3]
    assert "{\\k" in traducida


def test_traduccion_vacia_no_emite_evento():
    ass = render_karaoke_ass([linea()], translations=["  "])

    eventos = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(eventos) == 1
