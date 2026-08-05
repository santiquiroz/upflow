"""Spike: traducción local por ONNX, con verificación de ida y vuelta.

Uso:
    .venv\\Scripts\\python scripts\\spike_translation.py

Es la última pieza del doblaje automático: transcribir, TTS, clonar voz y
muxear ya están; falta traducir.

La verificación es de IDA Y VUELTA: se traduce inglés → español con un modelo y
el resultado vuelve español → inglés con OTRO. Si lo que vuelve se parece al
original, la traducción del medio dice lo que tenía que decir. Comparar contra
una traducción "correcta" escrita a mano no serviría: hay muchas traducciones
válidas de la misma frase.

OJO con el decoder merged: en este repo, Whisper sobre DirectML con el decoder
merged devolvía texto fluido y equivocado sin fallar. Acá se mide igual.
"""

from __future__ import annotations

import difflib
import re
import sys
import time

EN_ES = "onnx-community/opus-mt-en-es"
ES_EN = "onnx-community/opus-mt-es-en"

FRASES = [
    "The quick brown fox jumps over the lazy dog.",
    "Upscale your videos locally, without sending anything to the cloud.",
    "This model runs on your own graphics card.",
]


def normalizar(texto: str) -> str:
    return re.sub(r"[^a-z ]", "", texto.lower()).strip()


def main() -> int:
    try:
        from optimum.onnxruntime import ORTModelForSeq2SeqLM
        from transformers import AutoTokenizer
    except ImportError as exc:
        print(f"FALTA UNA DEPENDENCIA: {exc}")
        return 2

    def cargar(repo: str):
        print(f"Cargando {repo} ...")
        return (
            AutoTokenizer.from_pretrained(repo),
            ORTModelForSeq2SeqLM.from_pretrained(repo, use_merged=False),
        )

    try:
        tok_es, mod_es = cargar(EN_ES)
        tok_en, mod_en = cargar(ES_EN)
    except Exception as exc:  # noqa: BLE001
        print(f"NO CARGA: {type(exc).__name__}: {exc}")
        return 1

    def traducir(texto: str, tok, mod) -> str:
        entrada = tok(texto, return_tensors="pt")
        salida = mod.generate(**entrada, max_new_tokens=200)
        return tok.batch_decode(salida, skip_special_tokens=True)[0].strip()

    puntajes = []
    for frase in FRASES:
        started = time.perf_counter()
        al_espanol = traducir(frase, tok_es, mod_es)
        de_vuelta = traducir(al_espanol, tok_en, mod_en)
        elapsed = time.perf_counter() - started

        parecido = difflib.SequenceMatcher(
            None, normalizar(frase), normalizar(de_vuelta)
        ).ratio()
        puntajes.append(parecido)

        print(f"\n  original : {frase}")
        print(f"  español  : {al_espanol}")
        print(f"  de vuelta: {de_vuelta}")
        print(f"  parecido : {parecido:.0%}  ({elapsed:.2f} s ida y vuelta)")

        if normalizar(al_espanol) == normalizar(frase):
            print("  OJO: el 'español' es idéntico al inglés. No tradujo nada.")
            return 1

    promedio = sum(puntajes) / len(puntajes)
    print("\n================ VEREDICTO ================")
    print(f"parecido promedio en la ida y vuelta: {promedio:.0%}")
    if promedio >= 0.7:
        print("TRADUCE BIEN. Última pieza del doblaje resuelta.")
        return 0
    if promedio >= 0.4:
        print("Traduce, pero pierde bastante. Revisar antes de usarlo para doblaje.")
        return 0
    print("NO SIRVE: lo que vuelve no se parece a lo que se mandó.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
