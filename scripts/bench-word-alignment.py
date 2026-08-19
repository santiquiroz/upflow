"""Cuanto se equivoca la alineacion, en milisegundos, contra una verdad conocida.

    powershell scripts/make-word-alignment-bench.ps1   # genera el banco de voz
    python scripts/bench-word-alignment.py            # mide

Necesita un whisper-tiny.en descomprimido en runtime/bench/whisper-tiny-en.

Cada palabra se sintetizo por separado, asi que su duracion exacta se conoce; al
pegarlas con silencios fijos, los limites de cada palabra son un CALCULO y no una
estimacion. Eso permite medir el error en vez de mirar si "se ve razonable", que
es lo que no se podia hacer con una cancion en japones.

SAPI deja un poco de silencio al principio y al final de cada clip, asi que la
verdad se refina recortando ese silencio por energia: el limite real de la voz es
donde empieza a sonar, no donde arranca el archivo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

RAIZ = Path(__file__).resolve().parents[1]
AQUI = RAIZ / "runtime" / "bench"
BANCOS = {
    "en": {
        "modelo": "whisper-tiny-en",
        "palabras": ["hello", "world", "this", "is", "a", "test", "of", "word",
                     "level", "timing"],
    },
    "es": {
        "modelo": "whisper-tiny",
        "palabras": ["hola", "mundo", "esto", "es", "una", "prueba", "de",
                     "tiempos", "por", "palabra"],
    },
}
SILENCIO_ENTRE = 0.25
SR = 16000
# Debajo de esto se considera silencio de relleno del sintetizador.
UMBRAL_ENERGIA = 0.02


def recortar_silencio(audio: np.ndarray) -> tuple[np.ndarray, int, int]:
    fuerte = np.abs(audio) > UMBRAL_ENERGIA
    if not fuerte.any():
        return audio, 0, len(audio)
    primero = int(np.argmax(fuerte))
    ultimo = len(fuerte) - int(np.argmax(fuerte[::-1]))
    return audio[primero:ultimo], primero, ultimo


def construir(idioma: str) -> tuple[np.ndarray, list[tuple[str, float, float]]]:
    banco = AQUI / f"bench_words_{idioma}"
    palabras_del_banco = BANCOS[idioma]["palabras"]
    pistas: list[np.ndarray] = []
    verdad: list[tuple[str, float, float]] = []
    cursor = 0.0
    silencio = np.zeros(int(SILENCIO_ENTRE * SR), dtype=np.float32)
    for palabra in palabras_del_banco:
        audio, _ = sf.read(banco / f"{palabra}.wav", dtype="float32")
        recortado, _, _ = recortar_silencio(audio)
        inicio = cursor
        fin = cursor + len(recortado) / SR
        verdad.append((palabra, inicio, fin))
        pistas.append(recortado)
        pistas.append(silencio)
        cursor = fin + SILENCIO_ENTRE
    return np.concatenate(pistas), verdad


def main() -> int:
    idioma = sys.argv[1] if len(sys.argv) > 1 else "en"
    if idioma not in BANCOS:
        print(f"idioma desconocido: {idioma}. Validos: {', '.join(BANCOS)}")
        return 2
    señal, verdad = construir(idioma)
    print(f"idioma: {idioma}")
    mezcla = AQUI / f"bench_mix_{idioma}.wav"
    sf.write(mezcla, señal, SR)
    print(f"banco: {len(verdad)} palabras, {len(señal)/SR:.2f} s")

    sys.path.insert(0, str(RAIZ))
    from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
    from transformers import AutoProcessor

    from app.services.engines.whisper_alignment import align_words

    D = AQUI / BANCOS[idioma]["modelo"]
    model = ORTModelForSpeechSeq2Seq.from_pretrained(str(D), use_merged=False)
    proc = AutoProcessor.from_pretrained(str(D))

    feats = proc(señal, sampling_rate=SR, return_tensors="pt").input_features
    extra = {} if idioma == "en" else {"language": idioma}
    tokens = model.generate(
        input_features=feats, max_new_tokens=200, return_timestamps=True, **extra
    )
    texto = proc.batch_decode(tokens, skip_special_tokens=True)[0]
    print("transcripcion:", texto.strip())

    predichas = align_words(
        model=model,
        processor=proc,
        features=feats,
        tokens=tokens,
        model_dir=D,
        audio_seconds=len(señal) / SR,
    )
    print(f"alineadas: {len(predichas)}")

    # Solo se comparan las palabras que el modelo efectivamente reconocio, por
    # nombre: si transcribio de mas o de menos, comparar por posicion mediria el
    # error de transcripcion y no el de alineacion.
    por_nombre = {p.word.strip().lower().strip(".,!?"): p for p in predichas}
    errores_inicio: list[float] = []
    print(f"\n{'palabra':10} {'verdad':>14}  {'predicho':>14}   error inicio")
    for palabra, inicio, fin in verdad:
        p = por_nombre.get(palabra)
        if p is None:
            print(f"{palabra:10} {inicio:6.2f}-{fin:5.2f}   (no reconocida)")
            continue
        error = abs(p.start - inicio)
        errores_inicio.append(error)
        print(
            f"{palabra:10} {inicio:6.2f}-{fin:5.2f}  {p.start:6.2f}-{p.end:5.2f}"
            f"   {error*1000:7.0f} ms"
        )

    if not errores_inicio:
        print("\nNinguna palabra reconocida: el banco no sirve para medir.")
        return 1
    arr = np.array(errores_inicio)
    print(f"\nreconocidas : {len(arr)}/{len(verdad)}")
    print(f"error medio : {arr.mean()*1000:.0f} ms")
    print(f"error p90   : {np.percentile(arr,90)*1000:.0f} ms")
    print(f"error maximo: {arr.max()*1000:.0f} ms")
    comparar_con_el_reparto(idioma)
    return 0


def comparar_con_el_reparto(idioma: str) -> None:
    """El mismo banco, medido contra la aproximacion que ya usa el karaoke.

    Sin esto el numero de la alineacion no dice nada: 147 ms de error puede ser
    bueno o malo segun contra que. Lo unico que decide si conviene prenderla es
    si le gana a lo que ya hay.
    """
    import sys
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(RAIZ))
    from app.services.karaoke_subtitles import split_line_proportionally

    _, verdad = construir(idioma)
    texto = " ".join(p for p, _, _ in verdad)
    linea = split_line_proportionally(texto, verdad[0][1], verdad[-1][2])
    errores = [
        abs(pred.start - real_inicio)
        for pred, (_, real_inicio, _) in zip(linea.words, verdad)
    ]
    arr = np.array(errores)
    print("\n--- reparto proporcional por letras (lo que ya hay) ---")
    print(f"error medio : {arr.mean()*1000:.0f} ms")
    print(f"error p90   : {np.percentile(arr,90)*1000:.0f} ms")
    print(f"error maximo: {arr.max()*1000:.0f} ms")


if __name__ == "__main__":
    sys.exit(main())
