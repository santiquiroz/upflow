"""Driver numpy de Open-Unmix, vendorizado desde santiquiroz/port-openunmix-onnx.

Se copia y no se importa: el port es un repo aparte con su propio venv de torch
para exportar, y esta app no arrastra nada de eso en tiempo de inferencia. Lo
unico que viaja es la parte que corre sin torch — STFT/iSTFT en numpy.
"""
