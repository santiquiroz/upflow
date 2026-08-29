"""Mide cuanto se parece una malla al dibujo, y de que tipo es la diferencia.

Es la balanza del banco de pruebas: cualquier malla —generada por un modelo,
esculpida a mano o armada con primitivas— se renderiza a silueta y se compara
contra la vista correspondiente de la hoja. El numero sale igual para todas, asi
que la eleccion de motor queda medida y no opinada.

LO QUE SE APRENDIO PELEANDO CON UNA GORRA, y es la razon de que esto no
devuelva un solo numero: un IoU bajo no dice QUE hay que arreglar, y los tres
motivos posibles se arreglan en lugares distintos. Perder horas afinando la
forma cuando lo que fallaba era la escala es el modo de fallo real, medido.
Por eso cada vista viaja con su culpable:

  - `escala`: las cajas de tinta no tienen la misma proporcion. Ninguna cantidad
    de modelado arregla esto; hay que reescalar. Se detecta comparando ancho y
    alto reales, no el IoU.
  - `partes`: el contorno general esta bien pero las partes no caen en el mismo
    lugar adentro de el. Se detecta porque dejar correr la silueta sube el IoU
    de golpe. NO significa mover la malla en la escena: la comparacion centra
    las dos siluetas, asi que una traslacion global ya esta normalizada y
    mover no cambia el numero (probado, ver `Calce`). Significa mover una
    PARTE —la visera, un brazo— respecto del resto.
  - `forma`: proporcion correcta y centrado correcto, y aun asi no calza. Recien
    aca hay que modelar.

Cuando la malla NO esta en metros —lo que devuelve cualquier generador, que
entrega en unidades propias— el veredicto de escala se APAGA y las siluetas se
igualan por alto antes de comparar. Medir la escala de algo que no la tiene da
un numero cierto sobre una pregunta que no aplica: medido el 2026-08-28, una
malla generada quedaba cinco veces mas grande que el dibujo en la comparacion y
las tres vistas culpaban a la "escala" cuando la escala no existia. Eso no
degradaba un numero: castigaba a un motor por una unidad.

Se revisa en ese orden porque un error de escala se disfraza de error de forma,
y un reparto de partes tambien, pero nunca al reves.

Las tolerancias son PORCENTAJES del tamano, nunca valores fijos en cm: la misma
tolerancia absoluta que es razonable en una figura de 1,70 m es media figura en
una gorra de 35 cm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from app.services.turnaround import ink_mask, open_sheet

# Cuanto se deja correr la silueta al buscar el mejor calce, como fraccion de la
# diagonal de la tinta. Mas que esto ya no es "esta corrido": es otra cosa.
RADIO_BUSQUEDA = 0.10

# Debajo de esto la silueta ocupa tan poco del cuadro que el IoU lo domina el
# borde dentado en vez de la forma.
MINIMO_PIXELES_TINTA = 64

# Cuanto tiene que subir el IoU al permitir el corrimiento para culpar al
# reparto de las partes y no a la forma. Cinco puntos es mucho mas que el ruido
# de rasterizado medido entre dos renders del mismo objeto.
SALTO_QUE_DELATA_PARTES = 0.05

# Cuanto puede diferir una medida del modelo de la del dibujo antes de que el
# problema sea la escala. 8% sobre una figura de 1,70 m son 14 cm: visible a
# simple vista, y muy por encima del grosor del trazo.
DESVIO_QUE_DELATA_ESCALA = 0.08


class SiluetaVaciaError(ValueError):
    """La imagen no tiene silueta que medir."""


@dataclass(frozen=True, slots=True)
class Medida:
    """Un tamano del modelo contra el mismo tamano del dibujo, en cm."""

    modelo_cm: float
    dibujo_cm: float

    @property
    def crecimiento(self) -> float:
        """Cuanto sobra (o falta) el modelo, como fraccion del dibujo.

        Va CON SIGNO: el signo es lo que distingue "esta todo mas grande" de
        "le sobra algo de un lado", que piden arreglos opuestos.
        """
        if not self.dibujo_cm:
            return 0.0
        return round((self.modelo_cm - self.dibujo_cm) / self.dibujo_cm, 4)

    @property
    def desvio(self) -> float:
        return abs(self.crecimiento)


@dataclass(frozen=True, slots=True)
class Ajuste:
    """Que tan bien calza una vista, y de que tipo es lo que falta."""

    vista: str
    anclado: float
    mejor: float
    corrimiento_cm: tuple[float, float]
    ancho: Medida
    alto: Medida
    # False cuando la malla no esta en metros —lo que devuelve cualquier
    # generador—. Entonces las medidas en cm son de sus unidades propias y no
    # significan nada contra el dibujo.
    escala_medible: bool = True

    @property
    def gana_moviendo(self) -> float:
        return round(self.mejor - self.anclado, 4)

    @property
    def culpa(self) -> str:
        """Donde esta el problema, para no afinar lo que no falla.

        El orden importa: un error de escala se disfraza de error de forma, y
        un reparto de partes tambien. Revisarlos al reves manda a modelar de
        nuevo algo que solo habia que agrandar.

        Para llamarlo escala tienen que irse las DOS medidas para el MISMO lado.
        Una sola dimension pasada de largo no es escala sino geometria que
        sobra de un lado —un apendice, una visera— y decirle a alguien que
        reescale por eso lo manda a achicar todo lo que ya estaba bien. Medido
        con una silueta de prueba: un apendice chico inflaba el ancho 15% con
        el alto intacto, y bastaba correr la silueta para recuperar 0.975.
        """
        ancho, alto = self.ancho.crecimiento, self.alto.crecimiento
        mismo_lado = ancho * alto > 0
        if self.gana_moviendo >= SALTO_QUE_DELATA_PARTES:
            return "partes"
        if self.escala_medible and mismo_lado and min(abs(ancho), abs(alto)) >= DESVIO_QUE_DELATA_ESCALA:
            return "escala"
        return "forma"


@dataclass(frozen=True, slots=True)
class Calce:
    """El resultado de comparar una malla contra las vistas de una hoja.

    NO SE OFRECE UNA "CORRECCION VERTICAL", y conviene dejar escrito por que
    para que nadie la vuelva a agregar. La idea era tentadora: las tres vistas
    de la gorra pedian corrimiento vertical para el mismo lado (back 2.34 cm,
    front 1.60, side 1.23), el eje vertical es el mismo en todas, asi que
    parecia un solo desalineado con un solo arreglo —promediar y mover 1,72 cm.

    Se probo el 2026-08-28: se movio la malla 1,72 cm en Z, se volvio a medir, y
    los tres numeros dieron IDENTICOS hasta el cuarto decimal (0.5989 antes y
    despues). Tiene que ser asi: la comparacion centra las dos siluetas por su
    caja de tinta, o sea que cualquier traslacion global ya esta normalizada
    antes de calcular nada. El consejo habria mandado a mover una malla para no
    cambiar absolutamente nada.

    Lo que ese corrimiento comun SI indica es que la malla reparte su masa a lo
    alto distinto que el dibujo —aca porque mide 33,0 cm contra 36,5-37,9
    dibujados—, y eso ya viaja medido en `Ajuste.alto`.
    """

    ajustes: tuple[Ajuste, ...]
    metros_por_pixel: float

    @property
    def promedio(self) -> float:
        if not self.ajustes:
            return 0.0
        return round(sum(a.mejor for a in self.ajustes) / len(self.ajustes), 4)

    @property
    def peor_vista(self) -> str:
        return max(self.ajustes, key=lambda a: a.mejor).vista if self.ajustes else ""

    @property
    def culpas(self) -> dict[str, str]:
        return {a.vista: a.culpa for a in self.ajustes}


def mascara_de_silueta(ruta: Path) -> np.ndarray:
    """La silueta renderizada, del canal alfa.

    El render viene con fondo transparente, asi que el alfa ES la silueta y no
    hace falta adivinar un umbral de color ni depender del material.
    """
    with Image.open(ruta) as imagen:
        rgba = imagen.convert("RGBA")
        alfa = np.array(rgba)[:, :, 3]
    return alfa > 127


def mascara_de_dibujo(ruta: Path) -> np.ndarray:
    """La tinta de una vista de la hoja, con el mismo criterio que el resto."""
    return ink_mask(open_sheet(ruta))


def caja_de_tinta(mascara: np.ndarray) -> tuple[int, int, int, int]:
    filas = np.where(mascara.any(axis=1))[0]
    columnas = np.where(mascara.any(axis=0))[0]
    if not filas.size or not columnas.size:
        raise SiluetaVaciaError("la imagen no tiene ningun pixel de silueta")
    return int(columnas[0]), int(filas[0]), int(columnas[-1]) + 1, int(filas[-1]) + 1


def recortar(mascara: np.ndarray) -> np.ndarray:
    izq, arriba, der, abajo = caja_de_tinta(mascara)
    return mascara[arriba:abajo, izq:der]


def reescalar(mascara: np.ndarray, factor: float) -> np.ndarray:
    """Lleva una mascara a otra escala de pixel.

    Se reescala en gris y se vuelve a umbralar en vez de usar vecino mas
    cercano: el vecino mas cercano come tramos finos enteros —una visera, un
    dedo— y despues el IoU castiga al modelo por algo que hizo el remuestreo.
    """
    if abs(factor - 1.0) < 1e-6:
        return mascara
    alto, ancho = mascara.shape
    destino = (max(1, round(ancho * factor)), max(1, round(alto * factor)))
    imagen = Image.fromarray((mascara * 255).astype(np.uint8))
    return np.array(imagen.resize(destino, Image.BILINEAR)) > 127


def centrar_en(recorte: np.ndarray, forma: tuple[int, int]) -> np.ndarray:
    """Pone un recorte ya ajustado en el centro de un lienzo del tamano pedido.

    Centrar por la CAJA y no por el centroide es deliberado: el centroide se
    corre cuando una parte del objeto tiene mas area (una visera ancha tira del
    centroide hacia atras) y entonces dos siluetas de la misma forma quedan
    desalineadas por como esta repartida la masa.

    El lienzo NUNCA reescala: si el recorte no entra, el llamador pidio un
    lienzo mal calculado, y encogerlo aca romperia la escala metrica en
    silencio, que es exactamente el error que este modulo existe para no
    cometer.
    """
    alto_destino, ancho_destino = forma
    alto, ancho = recorte.shape
    if alto > alto_destino or ancho > ancho_destino:
        raise ValueError(f"el recorte {recorte.shape} no entra en el lienzo {forma}")

    lienzo = np.zeros(forma, dtype=bool)
    arriba = (alto_destino - alto) // 2
    izq = (ancho_destino - ancho) // 2
    lienzo[arriba:arriba + alto, izq:izq + ancho] = recorte
    return lienzo


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return round(float(np.count_nonzero(a & b) / union), 4) if union else 0.0


def mejor_calce(a: np.ndarray, b: np.ndarray, radio: int) -> tuple[float, int, int]:
    """El IoU mas alto corriendo `b` dentro de un radio, y cuanto hubo que correr.

    Se resuelve por correlacion cruzada con FFT y no probando corrimiento por
    corrimiento: la interseccion para TODOS los corrimientos sale de una sola
    transformada, y como el area de los dos no cambia al mover, maximizar la
    interseccion es lo mismo que maximizar el IoU. Probar de a uno son millones
    de operaciones por vista y esto son dos FFT.
    """
    if radio <= 0:
        return iou(a, b), 0, 0

    alto, ancho = a.shape
    relleno = (alto * 2, ancho * 2)  # sin esto la correlacion da la vuelta y miente
    fa = np.fft.rfft2(a.astype(np.float32), s=relleno)
    fb = np.fft.rfft2(b.astype(np.float32), s=relleno)
    interseccion = np.fft.irfft2(fa * np.conj(fb), s=relleno)

    area_a, area_b = np.count_nonzero(a), np.count_nonzero(b)
    desplazamientos = range(-radio, radio + 1)
    mejor, mejor_dy, mejor_dx = -1.0, 0, 0
    for dy in desplazamientos:
        fila = interseccion[dy % relleno[0]]
        for dx in desplazamientos:
            comun = fila[dx % relleno[1]]
            union = area_a + area_b - comun
            puntaje = float(comun / union) if union > 0 else 0.0
            if puntaje > mejor:
                mejor, mejor_dy, mejor_dx = puntaje, dy, dx
    return round(mejor, 4), mejor_dy, mejor_dx


def metros_por_pixel_de(dibujo: Path, alto_real_m: float) -> float:
    """La escala de la hoja, a partir de UNA vista de altura conocida.

    Se saca una sola vez y vale para todos los recortes de la misma hoja, que es
    lo que evita el error que techo la gorra: si cada vista se escala por SU
    propia altura de tinta, dos vistas del mismo objeto quedan a escalas
    distintas y ningun modelo puede calzar las dos.
    """
    if alto_real_m <= 0:
        raise ValueError("'alto_real_m' tiene que ser mayor que cero")
    _, arriba, _, abajo = caja_de_tinta(mascara_de_dibujo(dibujo))
    return alto_real_m / (abajo - arriba)


def _preparar(mascara: np.ndarray, vista: str, que: str) -> np.ndarray:
    if np.count_nonzero(mascara) < MINIMO_PIXELES_TINTA:
        raise SiluetaVaciaError(f"{que} de '{vista}' esta practicamente vacio")
    return recortar(mascara)


def igualar_altura(modelo: np.ndarray, referencia: np.ndarray) -> np.ndarray:
    """Lleva la silueta del modelo al alto de la del dibujo.

    Es para mallas SIN escala real. Un generador entrega en unidades propias
    —TripoSG devuelve algo del orden de 2 unidades para cualquier objeto—, asi
    que compararlas contra centimetros mide que la malla no esta en metros y no
    si la forma se parece. Medido el 2026-08-28: la misma malla generada quedaba
    cinco veces mas grande que el dibujo en la comparacion, y las tres vistas
    culpaban a la "escala" cuando la escala no existia.

    Se iguala por ALTO y no por area ni por diagonal: el alto es la medida que
    la hoja conoce de verdad (es la que fija la escala del dibujo), y usar el
    area haria que un apendice que sobra encogiera todo lo demas.
    """
    alto_modelo = modelo.shape[0]
    if not alto_modelo:
        raise SiluetaVaciaError("la silueta del modelo no tiene alto")
    return recortar(reescalar(modelo, referencia.shape[0] / alto_modelo))


def comparar_vista(
    vista: str,
    silueta: Path,
    dibujo: Path,
    metros_por_pixel_modelo: float,
    metros_por_pixel_dibujo: float,
    *,
    con_escala_real: bool = True,
) -> Ajuste:
    """Que tan bien calza una vista, con las dos siluetas a la misma escala.

    `con_escala_real=False` es para mallas que NO estan en metros —lo que
    devuelve cualquier generador—: se iguala el alto antes de comparar y el
    veredicto de escala se apaga, porque medir la escala de algo que no la
    tiene devuelve un numero cierto sobre una pregunta que no aplica.
    """
    modelo = _preparar(mascara_de_silueta(silueta), vista, "la silueta")
    referencia = _preparar(mascara_de_dibujo(dibujo), vista, "el dibujo")

    # El dibujo se lleva a la escala del render. Las dos escalas son metricas y
    # conocidas, asi que el factor es exacto y no hay que estimar nada.
    referencia = reescalar(referencia, metros_por_pixel_dibujo / metros_por_pixel_modelo)
    if not con_escala_real:
        modelo = igualar_altura(modelo, referencia)

    diagonal = float(np.hypot(*referencia.shape))
    radio = max(1, round(diagonal * RADIO_BUSQUEDA))

    # El lienzo tiene que aguantar la mas grande de las dos MAS la ventana de
    # busqueda entera; si no, el mejor calce cae fuera del borde y se reporta
    # como si no existiera.
    margen = 2 * radio + 2
    forma = (
        max(modelo.shape[0], referencia.shape[0]) + margen,
        max(modelo.shape[1], referencia.shape[1]) + margen,
    )
    modelo_centrado = centrar_en(modelo, forma)
    referencia_centrada = centrar_en(referencia, forma)

    anclado = iou(modelo_centrado, referencia_centrada)
    mejor, dy, dx = mejor_calce(modelo_centrado, referencia_centrada, radio)
    if mejor < anclado:  # el corrimiento (0,0) siempre esta en la ventana
        mejor, dy, dx = anclado, 0, 0

    a_cm = metros_por_pixel_modelo * 100
    return Ajuste(
        vista=vista,
        anclado=anclado,
        mejor=mejor,
        corrimiento_cm=(round(dx * a_cm, 2), round(dy * a_cm, 2)),
        ancho=Medida(round(modelo.shape[1] * a_cm, 2), round(referencia.shape[1] * a_cm, 2)),
        alto=Medida(round(modelo.shape[0] * a_cm, 2), round(referencia.shape[0] * a_cm, 2)),
        escala_medible=con_escala_real,
    )


def comparar(
    siluetas: dict[str, Path],
    dibujos: dict[str, Path],
    metros_por_pixel_modelo: float,
    metros_por_pixel_dibujo: float,
    *,
    con_escala_real: bool = True,
) -> Calce:
    """Compara todas las vistas que tengan silueta Y dibujo.

    Las vistas sueltas se ignoran en silencio a proposito: una hoja puede traer
    cuatro vistas y pedirse el calce de dos, y eso no es un error.
    """
    ajustes = tuple(
        comparar_vista(
            vista,
            siluetas[vista],
            dibujos[vista],
            metros_por_pixel_modelo,
            metros_por_pixel_dibujo,
            con_escala_real=con_escala_real,
        )
        for vista in sorted(siluetas)
        if vista in dibujos
    )
    if not ajustes:
        raise SiluetaVaciaError("ninguna vista tiene silueta y dibujo a la vez")
    return Calce(ajustes=ajustes, metros_por_pixel=metros_por_pixel_modelo)
