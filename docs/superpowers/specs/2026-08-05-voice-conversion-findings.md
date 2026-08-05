# Conversión de voz desde una muestra: por qué todavía no

**Fecha:** 2026-08-05
**Veredicto: no hay un camino listo como el que sí tuvo el TTS. No se implementa hasta que lo haya.**

## Qué se pedía

Transformar una voz para que suene como una muestra de referencia — "que lo diga
con esta voz" — a diferencia del TTS, que genera habla desde cero.

Es un problema DISTINTO del TTS y por eso se midió aparte.

## Qué se encontró

| Candidato | Licencia | ONNX listo | Sirve |
|---|---|---|---|
| Seed-VC (`Plachta/Seed-VC`) | **GPL-3.0** ⛔ | 0 archivos | No |
| OpenVoice v2 (`myshell-ai/OpenVoiceV2`) | MIT ✅ | **0 archivos** | Pesos PyTorch nada más |
| `onnx-community/knn-vc` | — | — | **No existe** |
| RVC y derivados (`*/VoiceConversionWebUI`) | MIT en algunos forks | Sin export declarado | Repos de WebUI, no modelos servibles |

La búsqueda en Hugging Face por `voice-conversion` devuelve sobre todo **forks
de la WebUI de RVC**, no modelos listos para consumir. Varios sin licencia
declarada.

## Los dos problemas, y son distintos

1. **El que tiene mejor licencia no tiene ONNX.** OpenVoice v2 es MIT pero
   publica pesos PyTorch. La app instala modelos ONNX desde Hugging Face y los
   corre con onnxruntime; meter un camino PyTorch paralelo es una decisión de
   arquitectura, no un detalle.
2. **El que está más maduro es GPL-3.0.** Seed-VC arrastra la misma restricción
   que este repo ya esquiva con Magpie corriéndolo como proceso aparte. Acá no
   se puede: la conversión tiene que pasar audio de ida y vuelta, no lanzar una
   ventana.

`torch` YA está instalado (2.13.0+cpu, llegó con optimum), así que un camino
PyTorch es técnicamente posible hoy. Pero sería el primero de la app, y esa
decisión merece tomarse a propósito y no de rebote.

## Qué haría falta para desbloquearlo

Cualquiera de estas tres, en orden de preferencia:

1. **Que aparezca un export ONNX con licencia limpia.** Es lo que paso con
   Kokoro para TTS: existía y por eso la feature salió en un día.
2. **Exportar OpenVoice v2 a ONNX nosotros.** Trabajo real y sin garantía: hay
   modelos que no exportan limpio, y habría que medir el resultado con la misma
   verificación de ida y vuelta que uso el TTS.
3. **Aceptar un camino PyTorch** para esta capacidad, asumiendo que el modelo no
   entra por el instalador de modelos existente.

## Lo que NO se hizo, a propósito

No se implementó nada. Empezar por la UI o por el job manager sin un modelo
verificado es exactamente lo que este repo ya pagó caro dos veces: la variante
fp16 de Kokoro devuelve audio NaN sin fallar, y el decoder merged de Whisper
sobre DirectML devuelve texto fluido pero equivocado.

Un modelo sin medir no es una feature, es una promesa.
