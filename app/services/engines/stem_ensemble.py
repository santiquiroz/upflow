"""Combinar el mismo stem estimado por varios modelos en uno.

Por que promediar sirve: cada arquitectura falla distinto. Lo que los modelos
comparten es la señal —todos estiman LA MISMA voz— y eso se suma en fase; lo que
no comparten son sus artefactos, que al no estar correlacionados se cancelan
parcialmente. El resultado tiene menos artefacto propio de un modelo que
cualquiera de las entradas.

Lo que NO hace: inventar separacion que ninguno logro. Si los tres se dejan la
voz adentro del instrumental, el promedio tambien la tiene. Promediar baja el
ruido de estimacion, no el error comun.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def align_lengths(pistas: list[np.ndarray]) -> list[np.ndarray]:
    """Todas al largo de la mas corta.

    Los modelos rellenan distinto al final —cada uno redondea a su tamaño de
    bloque— y unas pocas muestras de diferencia rompen la suma. Se recorta y no
    se rellena: rellenar con ceros mete un silencio que ninguno estimo.
    """
    minimo = min(pista.shape[0] for pista in pistas)
    return [pista[:minimo] for pista in pistas]


def average_stem_files(sources: list[Path], destination: Path) -> None:
    if not sources:
        raise ValueError("No hay nada que combinar.")
    pistas: list[np.ndarray] = []
    sample_rate = 0
    for source in sources:
        audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
        pistas.append(audio)
    alineadas = align_lengths(pistas)
    # float64 para la suma: con tres pistas float32 cerca de escala completa la
    # suma satura antes de dividir.
    combinado = np.mean(np.stack(alineadas).astype(np.float64), axis=0)
    # subtype explicito: soundfile escribe WAV en PCM_16 por default, y los
    # separadores entregan float32. Sin esto el ensemble —que existe para SUMAR
    # calidad— le cortaria 8 bits al intermedio antes de codificar la salida.
    sf.write(destination, combinado.astype(np.float32), sample_rate, subtype="FLOAT")
