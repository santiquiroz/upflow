"""Exporta el encoder de x-vector de speechbrain a ONNX, una sola vez.

Uso (entorno APARTE, no el de la app):
    python -m venv .venv-xvect
    .venv-xvect\\Scripts\\pip install torch==2.5.1 torchaudio==2.5.1 speechbrain==1.0.2 onnx requests soundfile numpy
    .venv-xvect\\Scripts\\python scripts\\export_xvector_onnx.py

Por que existe:

SpeechT5 VC clona bien, pero SOLO con x-vectors del encoder de speechbrain
(medido 2026-08-05: con embeddings de WavLM el resultado se parecia mas a la voz
de origen que a la de destino). Y speechbrain no se puede instalar en el entorno
de la app: 1.1.0 tiene un lazy import roto y 1.0.2 choca con torchaudio 2.11.

La salida es un ONNX que corre con onnxruntime a secas, igual que el resto de
los modelos de la app, sin arrastrar speechbrain a produccion.

NO se usa un export de terceros: los dos que hay en Hugging Face se probaron el
2026-08-05 y ninguno sirve — `arneyjfs` no inicializa la sesion, y `t-neethesh`
carga y corre pero devuelve embeddings fuera del espacio de SpeechT5 (la
conversion sale en NaN y con 52 s de audio en vez de 3).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Solo el TDNN. El frontend de features NO se exporta: ver la nota de abajo.
DESTINO = Path("vendor/xvector/tdnn.onnx")

# Parametros EXACTOS del Fbank de speechbrain, leidos del modelo cargado
# (2026-08-05). El llamador tiene que reproducirlos o los embeddings caen fuera
# del espacio y la conversion deja de clonar.
FBANK = {
    "sample_rate": 16000,
    "n_mels": 24,
    "n_fft": 400,
    "win_length_ms": 400,
    "hop_length_ms": 160,
}
# Y despues del Fbank va una normalizacion POR ORACION, restando la media y
# SIN dividir por el desvio (`norm_type="sentence"`, `std_norm=False`).
NORMALIZACION = "restar la media de cada oracion, sin dividir por el desvio"
SAMPLE_RATE = 16000
# Dos segundos de audio de ejemplo: la longitud es dinamica en el grafo, este
# valor solo traza la forma.
TRACE_SAMPLES = SAMPLE_RATE * 2


def main() -> int:
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError as exc:
        print(f"FALTA UNA DEPENDENCIA: {exc}")
        print("Este script corre en el venv APARTE, no en el de la app.")
        return 2

    print("Cargando speechbrain/spkrec-xvect-voxceleb ...")
    import speechbrain.utils.fetching as sb_fetching
    from speechbrain.utils.fetching import LocalStrategy

    # Windows sin modo desarrollador no deja crear symlinks, y el pretrainer de
    # speechbrain los usa por default aunque se le pida COPY al cargador. Se
    # fuerza la copia a nivel del modulo, que es el unico punto por el que pasan
    # todas las descargas.
    _link_original = sb_fetching.link_with_strategy

    def _siempre_copiar(src, dst, strategy):  # noqa: ANN001
        return _link_original(src, dst, LocalStrategy.COPY)

    sb_fetching.link_with_strategy = _siempre_copiar

    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-xvect-voxceleb",
        savedir=".tmp-xvect-src",
        local_strategy=LocalStrategy.COPY,
    )

    class XvectorWrapper(torch.nn.Module):
        """Envuelve el pipeline en un solo forward exportable.

        `encode_batch` hace normalizacion de features + el TDNN. Exportar solo el
        TDNN dejaria al llamador reimplementando la normalizacion, que es
        justo donde se rompe la compatibilidad de espacios.
        """

        def __init__(self, inner: object) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, feats: torch.Tensor) -> torch.Tensor:
            return self.inner(feats)  # type: ignore[operator]

    # MEDIDO 2026-08-05: exportar el pipeline COMPLETO (Fbank + TDNN) es
    # imposible con las dos rutas de torch.onnx. El exportador clasico revienta
    # trazando el Reshape del STFT de speechbrain, y el de dynamo falla al
    # convertir el grafo. El TDNN solo, en cambio, exporta limpio y coincide con
    # el original dentro de 2e-4.
    #
    # La consecuencia es que el Fbank hay que calcularlo del lado de la app, con
    # los parametros de FBANK y la normalizacion de NORMALIZACION.
    wrapper = XvectorWrapper(classifier.mods.embedding_model).eval()
    ejemplo = torch.randn(1, 200, FBANK["n_mels"])

    with torch.no_grad():
        referencia = wrapper(ejemplo)
    print(f"forma de salida: {tuple(referencia.shape)}")
    if referencia.shape[-1] != 512:
        print(f"INESPERADO: se esperaban 512 valores y salieron {referencia.shape[-1]}")
        return 1

    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exportando a {DESTINO} ...")
    # El exportador clasico (TorchScript) no puede trazar el STFT de speechbrain:
    # falla en el Reshape del `compute_STFT`. El exportador nuevo basado en
    # dynamo si maneja esas formas dinamicas.
    torch.onnx.export(
        wrapper,
        (ejemplo,),
        str(DESTINO),
        input_names=["feats"],
        output_names=["embedding"],
        # La cantidad de cuadros es dinamica: depende del largo del audio.
        dynamic_axes={"feats": {0: "batch", 1: "frames"}, "embedding": {0: "batch"}},
        opset_version=17,
    )

    print("Verificando el ONNX contra el original ...")
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(str(DESTINO), providers=["CPUExecutionProvider"])
    salida = session.run(["embedding"], {"feats": ejemplo.numpy()})[0]
    diferencia = float(np.abs(salida - referencia.numpy()).max())
    print(f"diferencia maxima contra el original: {diferencia:.6f}")
    if diferencia > 1e-3:
        print("EL EXPORT NO COINCIDE con el modelo original. No usarlo.")
        return 1

    print(f"\nListo: {DESTINO} ({DESTINO.stat().st_size / 1024 / 1024:.1f} MB)")
    print("Corre con onnxruntime a secas; la app no necesita speechbrain.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
