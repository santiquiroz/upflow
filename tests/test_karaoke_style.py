import pytest

from app.services.karaoke_subtitles import (
    KaraokeLine,
    KaraokeStyle,
    KaraokeWord,
    build_singer_style_lines,
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


# ---------------------------------------------------------------------------
# F2a: estilos por cantante
# ---------------------------------------------------------------------------


def test_el_estilo_del_cantante_usa_su_color_como_base():
    estilos = build_singer_style_lines(KaraokeStyle(), {"s2": "#FF0000"})

    assert len(estilos) == 1
    # Cambia el color BASE (lo no cantado); el resaltado sigue siendo el global,
    # asi el relleno se lee igual en todas las lineas.
    assert estilos[0].startswith("Style: Singer_s2,Arial,48,&H00FFFFFF,&H000000FF,")
    # Misma alineacion y margen que el estilo principal.
    assert estilos[0].endswith(",2,40,40,60,1")


def test_un_color_de_cantante_invalido_se_rechaza():
    with pytest.raises(ValueError):
        build_singer_style_lines(KaraokeStyle(), {"s1": "rojo"})


def test_la_linea_de_cada_cantante_usa_su_estilo():
    ass = render_karaoke_ass(
        [linea(), linea()],
        singers=["s1", "s2"],
        singer_colors={"s2": "#FF0000"},
    )

    eventos = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    # s1 no tiene color propio: cae al estilo principal, no a uno inventado.
    assert ",Karaoke,,0" in eventos[0]
    assert ",Singer_s2,,0" in eventos[1]
    assert "Style: Singer_s2," in ass


def test_sin_colores_los_cantantes_no_cambian_el_archivo():
    con = render_karaoke_ass([linea()], singers=["s1"])
    sin = render_karaoke_ass([linea()])

    assert con == sin


def test_la_traduccion_no_hereda_el_estilo_del_cantante():
    ass = render_karaoke_ass(
        [linea()],
        translations=["hello world"],
        singers=["s1"],
        singer_colors={"s1": "#FF0000"},
    )

    eventos = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert ",Singer_s1,,0" in eventos[0]
    # La traduccion acompaña con su propio estilo, como siempre.
    assert ",Translation,,0" in eventos[1]


# ---------------------------------------------------------------------------
# F2a: estilos por cantante — el color BASE identifica quien canta la linea
# ---------------------------------------------------------------------------


def test_cada_cantante_con_color_recibe_su_estilo_y_sus_lineas_lo_usan():
    ass = render_karaoke_ass(
        [linea(), linea("chau amigos")],
        singers=["s1", "s2"],
        singer_colors={"s1": "#FF0000", "s2": "#00FF00"},
    )

    # El color del cantante va en el BASE (SecondaryColour, lo no cantado); el
    # resaltado sigue siendo el global, asi el relleno se lee igual en todas.
    assert "Style: Singer_s1,Arial,48,&H00FFFFFF,&H000000FF," in ass
    assert "Style: Singer_s2,Arial,48,&H00FFFFFF,&H0000FF00," in ass
    eventos = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert ",Singer_s1,,0" in eventos[0]
    assert ",Singer_s2,,0" in eventos[1]


def test_el_estilo_del_cantante_hereda_tamano_posicion_y_margen_del_principal():
    ass = render_karaoke_ass(
        [linea()],
        style=KaraokeStyle(size="large", position="top"),
        singers=["s1"],
        singer_colors={"s1": "#FF0000"},
    )

    estilo = next(l for l in ass.splitlines() if l.startswith("Style: Singer_s1,"))
    assert ",Arial,64," in estilo
    assert estilo.endswith(",8,40,40,40,1")


def test_linea_de_cantante_sin_color_cae_al_estilo_principal():
    ass = render_karaoke_ass([linea()], singers=["s1"], singer_colors={})

    assert "Style: Singer_" not in ass
    assert ",Karaoke,,0" in ass


def test_sin_lista_de_cantantes_los_colores_no_cambian_nada():
    con_colores = render_karaoke_ass([linea()], singer_colors={"s1": "#FF0000"})

    eventos = [l for l in con_colores.splitlines() if l.startswith("Dialogue:")]
    assert ",Karaoke,,0" in eventos[0]


def test_la_traduccion_no_cambia_de_estilo_por_cantante():
    ass = render_karaoke_ass(
        [linea()],
        translations=["hello world"],
        singers=["s1"],
        singer_colors={"s1": "#FF0000"},
    )

    eventos = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert ",Singer_s1,,0" in eventos[0]
    # La traduccion acompaña, no compite: conserva su estilo chico global.
    assert ",Translation,,0" in eventos[1]


def test_color_de_cantante_invalido_se_rechaza():
    with pytest.raises(ValueError):
        render_karaoke_ass(
            [linea()], singers=["s1"], singer_colors={"s1": "verde"}
        )
