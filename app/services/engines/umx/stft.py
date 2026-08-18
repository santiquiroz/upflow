"""STFT/iSTFT en numpy, con la misma convención que usa Open-Unmix en torch.

ONNX no tiene dtype complejo ni `istft`, así que las puntas de la transformada
quedan afuera del grafo y se reimplementan acá. Es el mismo corte que hace el
port de RoFormer: el grafo va de magnitud a magnitud y la fase nunca lo toca.

umxhq usa n_fft=4096, hop=1024, ventana Hann periódica y `center=True`. Torch
rellena por reflexión y devuelve `n_fft//2 + 1 = 2049` bins.
"""
from __future__ import annotations

import numpy as np

N_FFT = 4096
HOP = 1024


def hann_periodic(size: int) -> np.ndarray:
    """La de torch: periódica, no simétrica. Con la simétrica el overlap-add no suma 1."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(size) / size)


def stft(audio: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """audio [canales, muestras] -> espectro complejo [canales, bins, cuadros]."""
    ventana = hann_periodic(n_fft)
    relleno = n_fft // 2
    acolchado = np.pad(audio, ((0, 0), (relleno, relleno)), mode="reflect")
    cuadros = 1 + (acolchado.shape[-1] - n_fft) // hop
    salida = np.empty((audio.shape[0], n_fft // 2 + 1, cuadros), dtype=np.complex128)
    for i in range(cuadros):
        trozo = acolchado[:, i * hop : i * hop + n_fft] * ventana
        salida[:, :, i] = np.fft.rfft(trozo, axis=-1)
    return salida


def istft(espectro: np.ndarray, muestras: int, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """Inversa con normalización por la suma de ventanas al cuadrado (WOLA)."""
    ventana = hann_periodic(n_fft)
    canales, _, cuadros = espectro.shape
    largo = n_fft + hop * (cuadros - 1)
    acumulado = np.zeros((canales, largo), dtype=np.float64)
    pesos = np.zeros(largo, dtype=np.float64)
    for i in range(cuadros):
        trozo = np.fft.irfft(espectro[:, :, i], n=n_fft, axis=-1) * ventana
        acumulado[:, i * hop : i * hop + n_fft] += trozo
        pesos[i * hop : i * hop + n_fft] += ventana**2
    # Donde la ventana no llega, dividir sería 0/0: esas muestras son el relleno
    # que se recorta abajo, así que se dejan en cero en vez de producir inf.
    utiles = pesos > 1e-11
    acumulado[:, utiles] /= pesos[utiles]
    relleno = n_fft // 2
    return acumulado[:, relleno : relleno + muestras]
