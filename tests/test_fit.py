from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.services import fit


LIENZO = 400
OVALO = (100, 120, 300, 280)


def _silueta(path: Path, dibujar) -> Path:
    """Como sale del render: fondo transparente, silueta en el alfa."""
    imagen = Image.new("RGBA", (LIENZO, LIENZO), (0, 0, 0, 0))
    dibujar(ImageDraw.Draw(imagen), (255, 255, 255, 255))
    imagen.save(path)
    return path


def _dibujo(path: Path, dibujar) -> Path:
    """Como viene de la hoja: tinta oscura sobre blanco."""
    imagen = Image.new("RGB", (LIENZO, LIENZO), "white")
    dibujar(ImageDraw.Draw(imagen), (10, 10, 10))
    imagen.save(path)
    return path


def _ovalo(draw, color):
    draw.ellipse(OVALO, fill=color)


def _ovalo_grande(draw, color):
    draw.ellipse((60, 90, 340, 310), fill=color)


def _rectangulo(draw, color):
    draw.rectangle(OVALO, fill=color)


def _ovalo_con_apendice(draw, color):
    draw.ellipse(OVALO, fill=color)
    draw.ellipse((290, 190, 330, 215), fill=color)


UN_MILIMETRO = 0.001


def _comparar(tmp_path: Path, silueta, dibujo, *, mpp_dibujo: float = UN_MILIMETRO) -> fit.Ajuste:
    return fit.comparar_vista(
        "front",
        _silueta(tmp_path / "s.png", silueta),
        _dibujo(tmp_path / "d.png", dibujo),
        UN_MILIMETRO,
        mpp_dibujo,
    )


def test_la_misma_forma_calza_perfecto(tmp_path: Path) -> None:
    ajuste = _comparar(tmp_path, _ovalo, _ovalo)

    assert ajuste.anclado == 1.0
    assert ajuste.corrimiento_cm == (0.0, 0.0)


def test_dos_formas_distintas_del_mismo_tamano_culpan_a_la_forma(tmp_path: Path) -> None:
    ajuste = _comparar(tmp_path, _rectangulo, _ovalo)

    assert ajuste.culpa == "forma"
    assert ajuste.anclado < 1.0
    # Mismo alto y mismo ancho: lo unico que difiere es el contorno.
    assert ajuste.ancho.crecimiento == 0.0
    assert ajuste.alto.crecimiento == 0.0


def test_un_modelo_uniformemente_mas_grande_culpa_a_la_escala(tmp_path: Path) -> None:
    ajuste = _comparar(tmp_path, _ovalo_grande, _ovalo)

    assert ajuste.culpa == "escala"
    assert ajuste.ancho.crecimiento > 0
    assert ajuste.alto.crecimiento > 0


def test_una_sola_dimension_pasada_no_es_escala_sino_reparto_de_partes(tmp_path: Path) -> None:
    """Un apendice ensancha la caja sin ser un problema de escala.

    Es el caso que hizo cambiar la regla: mandar a reescalar por esto achica
    todo lo que ya estaba bien.
    """
    ajuste = _comparar(tmp_path, _ovalo_con_apendice, _ovalo)

    assert ajuste.ancho.crecimiento >= fit.DESVIO_QUE_DELATA_ESCALA
    assert ajuste.alto.crecimiento == 0.0
    assert ajuste.culpa == "partes"
    assert ajuste.mejor > ajuste.anclado


def test_una_traslacion_global_no_cambia_el_calce(tmp_path: Path) -> None:
    """Fija el hallazgo que evito shipear un consejo falso.

    La comparacion centra las dos siluetas por su caja de tinta, asi que mover
    el objeto entero ya esta normalizado antes de calcular nada. Se habia
    agregado una "correccion vertical" que decia cuanto mover la malla; al
    probarla contra la gorra el puntaje dio identico hasta el cuarto decimal.
    Si este test empieza a fallar, alguien rompio el centrado —o volvio a creer
    que mover sirve.
    """
    def _ovalo_corrido(draw, color):
        draw.ellipse((140, 60, 340, 220), fill=color)

    quieto = _comparar(tmp_path, _ovalo, _ovalo)
    corrido = fit.comparar_vista(
        "front",
        _silueta(tmp_path / "s2.png", _ovalo_corrido),
        _dibujo(tmp_path / "d2.png", _ovalo),
        UN_MILIMETRO,
        UN_MILIMETRO,
    )

    assert corrido.anclado == quieto.anclado
    assert corrido.mejor == quieto.mejor


def test_el_mejor_calce_nunca_es_peor_que_el_anclado(tmp_path: Path) -> None:
    """El corrimiento (0,0) siempre esta dentro de la ventana de busqueda."""
    for silueta in (_ovalo, _rectangulo, _ovalo_grande, _ovalo_con_apendice):
        ajuste = _comparar(tmp_path, silueta, _ovalo)
        assert ajuste.mejor >= ajuste.anclado


def test_el_dibujo_a_otra_escala_de_pixel_calza_igual(tmp_path: Path) -> None:
    """Lo que decide es la escala METRICA, no el tamano en pixeles.

    Si el dibujo viniera escaneado al doble de resolucion, tiene que dar el
    mismo resultado; que no lo haga significa que se esta comparando pixeles
    en vez de milimetros.
    """
    def _ovalo_al_doble(draw, color):
        draw.ellipse((50, 60, 150, 140), fill=color)

    ajuste = _comparar(tmp_path, _ovalo, _ovalo_al_doble, mpp_dibujo=UN_MILIMETRO * 2)

    assert ajuste.anclado > 0.98


def test_una_silueta_vacia_se_reporta_y_no_devuelve_cero(tmp_path: Path) -> None:
    """Cero seria un numero valido con cara de medicion: no hay que darlo."""
    with pytest.raises(fit.SiluetaVaciaError):
        _comparar(tmp_path, lambda draw, color: None, _ovalo)


def test_un_dibujo_vacio_se_reporta(tmp_path: Path) -> None:
    with pytest.raises(fit.SiluetaVaciaError):
        _comparar(tmp_path, _ovalo, lambda draw, color: None)


def test_la_escala_de_la_hoja_sale_del_alto_de_tinta(tmp_path: Path) -> None:
    """Se afirma la PROPIEDAD, no una cuenta hecha a mano.

    Contar los pixeles de tinta a ojo desde las coordenadas del ovalo se
    equivoca por uno —`ellipse` incluye los dos extremos— y entonces el test
    falla por la aritmetica del test y no por el codigo.
    """
    dibujo = _dibujo(tmp_path / "d.png", _ovalo)
    _, arriba, _, abajo = fit.caja_de_tinta(fit.mascara_de_dibujo(dibujo))

    metros_por_pixel = fit.metros_por_pixel_de(dibujo, 0.32)

    assert metros_por_pixel * (abajo - arriba) == pytest.approx(0.32)


def test_la_escala_es_proporcional_a_la_altura_declarada(tmp_path: Path) -> None:
    dibujo = _dibujo(tmp_path / "d.png", _ovalo)

    assert fit.metros_por_pixel_de(dibujo, 0.64) == pytest.approx(
        fit.metros_por_pixel_de(dibujo, 0.32) * 2
    )


def test_una_altura_real_no_positiva_se_rechaza(tmp_path: Path) -> None:
    dibujo = _dibujo(tmp_path / "d.png", _ovalo)

    with pytest.raises(ValueError):
        fit.metros_por_pixel_de(dibujo, 0)


def test_comparar_ignora_las_vistas_sin_pareja(tmp_path: Path) -> None:
    """Pedir el calce de dos vistas sobre una hoja de cuatro no es un error."""
    silueta = _silueta(tmp_path / "s.png", _ovalo)
    dibujo = _dibujo(tmp_path / "d.png", _ovalo)

    calce = fit.comparar(
        {"front": silueta, "back": silueta},
        {"front": dibujo},
        UN_MILIMETRO,
        UN_MILIMETRO,
    )

    assert [ajuste.vista for ajuste in calce.ajustes] == ["front"]
    assert calce.promedio == 1.0
    assert calce.peor_vista == "front"


def test_comparar_sin_ninguna_pareja_se_reporta(tmp_path: Path) -> None:
    silueta = _silueta(tmp_path / "s.png", _ovalo)
    dibujo = _dibujo(tmp_path / "d.png", _ovalo)

    with pytest.raises(fit.SiluetaVaciaError):
        fit.comparar({"front": silueta}, {"back": dibujo}, UN_MILIMETRO, UN_MILIMETRO)


def test_el_lienzo_no_encoge_en_silencio_lo_que_no_entra() -> None:
    """Encoger para que entre romperia la escala metrica sin avisar."""
    import numpy as np

    recorte = np.ones((10, 10), dtype=bool)

    with pytest.raises(ValueError):
        fit.centrar_en(recorte, (5, 5))


def test_las_culpas_se_reportan_por_vista(tmp_path: Path) -> None:
    calce = fit.comparar(
        {"front": _silueta(tmp_path / "s.png", _ovalo_grande)},
        {"front": _dibujo(tmp_path / "d.png", _ovalo)},
        UN_MILIMETRO,
        UN_MILIMETRO,
    )

    assert calce.culpas == {"front": "escala"}


def test_una_malla_sin_escala_real_se_iguala_por_alto(tmp_path: Path) -> None:
    """Un generador entrega en unidades propias, no en metros.

    Compararlas contra centímetros mide que la malla no está en metros y no si
    la forma se parece. Medido: la misma malla generada quedaba cinco veces más
    grande que el dibujo y las tres vistas culpaban a la escala, que no existía.
    """
    def _ovalo_gigante(draw, color):
        draw.ellipse((20, 40, 380, 330), fill=color)

    con_escala = fit.comparar_vista(
        "front",
        _silueta(tmp_path / "s.png", _ovalo_gigante),
        _dibujo(tmp_path / "d.png", _ovalo),
        UN_MILIMETRO,
        UN_MILIMETRO,
    )
    sin_escala = fit.comparar_vista(
        "front",
        _silueta(tmp_path / "s2.png", _ovalo_gigante),
        _dibujo(tmp_path / "d2.png", _ovalo),
        UN_MILIMETRO,
        UN_MILIMETRO,
        con_escala_real=False,
    )

    assert con_escala.culpa == "escala"
    assert sin_escala.culpa != "escala"
    # Igualar el alto tiene que RECUPERAR parecido, no sólo cambiar la etiqueta.
    assert sin_escala.mejor > con_escala.mejor


def test_sin_escala_real_el_veredicto_de_escala_queda_apagado(tmp_path: Path) -> None:
    """Aunque las medidas sigan difiriendo, la pregunta no aplica."""
    ajuste = fit.comparar_vista(
        "front",
        _silueta(tmp_path / "s.png", _ovalo_grande),
        _dibujo(tmp_path / "d.png", _ovalo),
        UN_MILIMETRO,
        UN_MILIMETRO,
        con_escala_real=False,
    )

    assert ajuste.escala_medible is False
    assert ajuste.culpa in {"partes", "forma"}


def test_con_escala_real_es_el_comportamiento_por_defecto(tmp_path: Path) -> None:
    """Una malla construida a escala real NO debe normalizarse en silencio."""
    ajuste = _comparar(tmp_path, _ovalo_grande, _ovalo)

    assert ajuste.escala_medible is True
    assert ajuste.culpa == "escala"


def test_la_escala_gana_sobre_las_partes_cuando_las_dos_aplican() -> None:
    """El ORDEN de precedencia, afirmado sobre el propio veredicto.

    Es el test que faltaba: durante esta tanda alguien invirtió los dos `if`
    de `culpa` y la suite entera siguió en verde, porque ningún caso tenía las
    dos condiciones activas a la vez. Un error de escala se disfraza de las
    otras dos causas, así que revisarlo último manda a modelar de nuevo algo
    que sólo había que agrandar.
    """
    # Las dos condiciones ciertas a la vez: crece parejo (escala) Y gana mucho
    # al correrse (partes). Construido a mano para no depender de un render.
    ambas = fit.Ajuste(
        vista="front",
        anclado=0.40,
        mejor=0.90,  # gana 0.50 moviéndose, muy por encima del umbral de partes
        corrimiento_cm=(1.0, 1.0),
        ancho=fit.Medida(modelo_cm=60.0, dibujo_cm=50.0),
        alto=fit.Medida(modelo_cm=120.0, dibujo_cm=100.0),
    )

    assert ambas.gana_moviendo >= fit.SALTO_QUE_DELATA_PARTES
    assert min(ambas.ancho.desvio, ambas.alto.desvio) >= fit.DESVIO_QUE_DELATA_ESCALA
    assert ambas.culpa == "escala"


def test_sin_escala_medible_esa_misma_medida_cae_a_partes() -> None:
    """La precedencia sólo aplica cuando la escala se puede medir."""
    sin_metros = fit.Ajuste(
        vista="front",
        anclado=0.40,
        mejor=0.90,
        corrimiento_cm=(1.0, 1.0),
        ancho=fit.Medida(modelo_cm=60.0, dibujo_cm=50.0),
        alto=fit.Medida(modelo_cm=120.0, dibujo_cm=100.0),
        escala_medible=False,
    )

    assert sin_metros.culpa == "partes"
