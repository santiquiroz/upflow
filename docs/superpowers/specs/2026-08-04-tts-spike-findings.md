# Spike de viabilidad: generación de voz desde texto

**Fecha:** 2026-08-04
**Script reproducible:** `scripts/spike_tts.py`
**Veredicto: TTS es viable y rápido acá. La traba NO es técnica, es de licencia.**

## Qué se midió

Verificación de IDA Y VUELTA, no "la clase carga": se generó habla desde una
frase conocida y se transcribió de vuelta con el mismo Whisper que usa la app.

| Medición | Resultado |
|---|---|
| Modelo | `facebook/mms-tts-eng` (VITS, char-level, sin phonemizador) |
| Audio generado | 3,79 s a 16 kHz |
| Tiempo de generación | **0,64 s** |
| Factor de tiempo real | **0,17x** — genera 6 veces más rápido de lo que dura |
| Transcripción de vuelta | `"The quick brown fox jumps over the lazy dog."` |
| Parecido con lo pedido | **100%** |

Es habla inteligible de verdad, no ruido con la duración correcta. Y es
holgadamente rápido para doblaje: el cuello de botella no va a ser el TTS.

## La traba: el que funciona no se puede usar

| Modelo | Licencia | Runtime | ¿Necesita phonemizador? | ¿Verificado? |
|---|---|---|---|---|
| `facebook/mms-tts-eng` | **CC-BY-NC-4.0** ⛔ no comercial | PyTorch (VITS) | No, char-level | **Sí, 100%** |
| `onnx-community/Kokoro-82M-v1.0-ONNX` | Apache 2.0 ✅ | **ONNX** | **Sí, IPA** | No |
| `rhasspy/piper-voices` | MIT ✅ | ONNX | Sí, espeak-ng | No |

El único que corre sin dependencias extra es el único que no se puede
distribuir. Es exactamente el tipo de trampa que este repo ya evita con Magpie
(GPL-3.0, por eso corre como proceso aparte).

Se confirmó que Kokoro espera FONEMAS y no texto: su vocabulario tiene 115
símbolos e incluye IPA (`æ ə ɪ θ ɹ ʃ ʌ ʒ ˈ ː`). Sin un paso texto→fonemas no se
le puede pasar una frase.

## Las tres salidas, y qué cuesta cada una

1. **Sumar un phonemizador** (espeak-ng, ~5 MB) y usar Kokoro o Piper. Licencias
   limpias, corre por ONNX igual que el resto de la app, y Kokoro trae varias
   voces. Costo: otro binario en el instalador — precedente existe, ffmpeg ya
   viaja adentro.
2. **Usar MMS-TTS igual**, marcando la restricción no comercial en la UI. Cero
   dependencias nuevas, pero le pone una condición legal a una app que hoy no
   tiene ninguna.
3. **Buscar un modelo char-level con licencia limpia.** No se encontró uno
   verificado en este spike; sería más búsqueda.

**Recomendación: la opción 1.** Encaja con la arquitectura que ya existe (ONNX,
instalable desde Hugging Face, buscable) y no le agrega condiciones legales a la
app. El costo es un binario más, que es el mismo trato que ya se aceptó por
ffmpeg.

## Lo que este spike NO responde

- **Conversión de voz desde una muestra** ("que suene como esta grabación") no se
  midió. Es un problema distinto del TTS y necesita su propio spike.
- Kokoro y Piper no se corrieron: sin phonemizador instalado no se les puede
  pasar texto. Antes de elegir uno hay que medirlo igual que a MMS.
- No se midió calidad subjetiva ni voces en español.
