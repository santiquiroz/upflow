# Corrección: este camino de Whisper SÍ devuelve marcas de tiempo

**Fecha:** 2026-08-04
**Script reproducible:** `scripts/spike_whisper_timestamps.py`
**Veredicto: la capacidad `video.subtitles` es construible. No hacía falta otro runtime.**

## Qué decía el repo

`app/services/engines/transcribe_onnx.py`, en el docstring de `split_into_chunks`:

> El solape mejoraria los cortes a mitad de palabra, pero deduplicar lo repetido
> necesita timestamps por token, y en este camino `return_timestamps=True` no
> devuelve marcas (medido 2026-07-29).

Esa línea era el bloqueo de toda la feature de subtítulos: sin tiempos hay
transcripción, no subtítulos.

## Qué se midió ahora

Mismo modelo verificado en el spike anterior (`onnx-community/whisper-tiny.en`),
voz real (`Narsil/asr_dummy`, 10,4 s), CPU, `use_merged=False`.

| Vía | Resultado |
|---|---|
| `generate(return_timestamps=True)` + `batch_decode(skip_special_tokens=True)` | Texto sin marcas — **esto es lo que se midió en julio** |
| `batch_decode(..., decode_with_timestamps=True)` | `<\|0.00\|> He hoped...fat<\|5.56\|><\|5.56\|> mutton pieces...sauce.<\|10.16\|>` ✅ |
| `batch_decode(..., output_offsets=True)` | `[{'text': ' He hoped...', 'timestamp': (0.0, 5.56)}, {'text': ' mutton pieces...', 'timestamp': (5.56, 10.16)}]` ✅ |

## Por qué se concluyó mal en julio

`return_timestamps=True` sí hace que el modelo EMITA los tokens de tiempo. Lo
que los borra es el decodificado: `skip_special_tokens=True` los descarta, y es
el default que usa el engine. Mirando solo el texto devuelto parece que no hay
marcas.

Las marcas estaban ahí todo el tiempo; el paso que las tira estaba a una línea
de distancia.

## Consecuencia

- La granularidad es por SEGMENTO (frase), no por palabra. Alcanza de sobra para
  subtítulos: es la misma granularidad que produce un `.srt` normal.
- No hace falta vendorizar whisper.cpp ni cambiar de runtime, que era la
  alternativa que el spike de julio dejaba planteada.
- El comentario de `split_into_chunks` sobre el solape queda desactualizado: con
  tiempos por segmento, deduplicar lo repetido en un solape SÍ es posible.

## Lección

Una medición negativa envejece peor que una positiva: "no anda" puede
significar "no anda", pero también "no lo llamé bien". Esta se citó como hecho
durante meses y bloqueaba una feature entera. El spike que la desmiente corre en
menos de un minuto.
