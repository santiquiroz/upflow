# Conversión de voz desde una muestra: por qué todavía no

**Fecha:** 2026-08-05
**Veredicto ORIGINAL (equivocado): "no hay camino listo".**
**Veredicto CORREGIDO el mismo día: SÍ lo hay. `microsoft/speecht5_vc`, licencia
MIT, medido y funcionando.** Ver la corrección al final.

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


---

# CORRECCIÓN (2026-08-05, mismo día)

**El veredicto de arriba estaba mal, y lo destapó una pregunta del usuario:
"¿no hay otra manera? ¿no hay modelos en Hugging Face?".**

## Qué estuvo mal en el método

Se buscó por el NOMBRE `voice-conversion` y se miraron cuatro repos elegidos de
memoria. Eso no es una búsqueda: es confirmar una corazonada. Nunca se buscó por
tarea, ni se probaron los modelos que transformers ya soporta nativo.

## Lo que aparece cuando se busca bien

| Candidato | Licencia | ONNX | Sirve |
|---|---|---|---|
| **`microsoft/speecht5_vc`** | **MIT** ✅ | 0, pero transformers lo corre nativo | **Sí, medido** |
| `FunAudioLLM/CosyVoice2-0.5B` | Apache 2.0 ✅ | **4 archivos** | Sin medir |
| `coqui/XTTS-v2` | Coqui PML (no comercial) ⛔ | 0 | No |
| `SWivid/F5-TTS` | CC-BY-NC-4.0 ⛔ | 0 | No |

## Medición de SpeechT5 VC

`scripts/spike_voice_conversion.py`. Voz de origen generada con Kokoro, voz
destino con un x-vector fijo.

| Medición | Resultado |
|---|---|
| Contenido tras convertir | `"The quick bound fox jumps over the lazy dog."` — **95%** |
| Timbre cambió | diferencia espectral **0,754** (0 sería audio idéntico) |
| Velocidad | 3,26 s de audio en 2,31 s → **0,71x tiempo real** |
| Audio finito | sí |

### Por qué se miden DOS cosas y no una

En TTS alcanza con preguntar "¿dice lo que le pedí?". En conversión no: un modelo
que devolviera el audio de entrada intacto pasaría esa prueba con 100% y no
habría convertido nada. Por eso se mide también que el timbre haya cambiado.

## Lo que sigue sin estar resuelto

- **No hay ONNX de SpeechT5 VC**, así que corre por PyTorch. Sería el primer
  camino PyTorch de la app y esa decisión de arquitectura sigue en pie — pero
  ahora es una decisión sobre CÓMO integrarlo, no sobre si se puede.
- **Clonar una voz concreta desde una muestra** necesita sacar el x-vector de esa
  grabación. Acá se usó uno sintético: alcanza para probar que la conversión
  corre y cambia el timbre, no para clonar a alguien puntual.
- CosyVoice2 (Apache 2.0, CON ONNX) no se midió y podría ser mejor encaje que
  SpeechT5, justamente por traer ONNX.

## La lección

Una búsqueda negativa vale menos que una positiva y hay que sospecharla más. En
un solo día este repo dio dos veredictos "no se puede" que resultaron falsos: el
de los timestamps de Whisper (que bloqueaba subtítulos) y este. Los dos cayeron
con menos de una hora de medición.
