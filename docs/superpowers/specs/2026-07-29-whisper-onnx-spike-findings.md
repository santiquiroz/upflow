# Spike de viabilidad: transcripción y subtítulos por Whisper sobre ONNX

**Fecha:** 2026-07-29
**Script reproducible:** `scripts/spike_whisper_onnx.py --transcribe`
**Veredicto: el camino existe y es el buscable.**

## Por qué hacía falta el spike

El README tenía whisper.cpp como camino previsto. Pero whisper.cpp sería un
**paquete vendorizado** (un binario que se baja con un script), y eso choca con el
requisito del usuario: que los modelos de esta capacidad se puedan **buscar**, como
cualquier otro. Un paquete vendorizado no es buscable.

El camino ONNX sí lo es: entra por el registro de modelos, el instalador y la
búsqueda en Hugging Face que ya existen.

## Resultado medido

| Pregunta | Respuesta |
|---|---|
| ¿Existe la clase de runtime? | Sí, `ORTModelForSpeechSeq2Seq` |
| ¿El exportador conoce whisper? | Sí: `automatic-speech-recognition` y su variante `-with-past` |
| ¿Carga un repo ONNX ya exportado, sin exportar nada? | **Sí**, `onnx-community/whisper-tiny.en` |
| ¿Sobre DirectML? | **Sí, pero sólo con `use_merged=False`** — ver abajo |
| ¿Sobre CPU? | Sí, con el default |
| ¿Transcribe voz real bien? | Sí: `"I know Kung Fu."` exacto |

Clase resultante: `ORTModelForWhisper`.

Versiones: optimum 2.1.0, optimum-onnx 0.1.0, onnxruntime-directml 1.24.4,
transformers 4.57.6.

## El hallazgo que importa: DirectML con el decoder merged devuelve basura

Medido con voz real (`Narsil/asr_dummy`, `i-know-kung-fu.mp3`, transcripción
conocida):

| Configuración | Resultado |
|---|---|
| CPU, default | `"I know Kung Fu."` ✅ |
| DirectML, decoder **merged** (el DEFAULT) | `"Ioti plur plur plur plur fighting fighting…"` ❌ |
| DirectML, `use_merged=False` | `"I know Kung Fu."` ✅ |

**No falla: carga bien, corre sin excepción, y devuelve texto equivocado.** Es la
peor forma de fallar, porque nada avisa.

**Consecuencia obligatoria para la implementación:** el motor tiene que pasar
`use_merged=False`. No es una optimización ni una preferencia — con el default, la
transcripción sobre DirectML es inservible y silenciosamente.

### Cómo casi se me pasa

La primera corrida usó un tono sinusoidal de 220 Hz como entrada. Dio esto:

- CPU: `"you"`
- DirectML: `"you treatedholderededed volunteer volunteer volunteer…"`

Y lo escribí como *"es lo esperado: Whisper alucina sobre entrada sin voz"*. Eso
es cierto para el caso de CPU, y **tapó un defecto real** en el de DirectML: la
diferencia entre una salida mínima y un delirio en bucle era la señal, y la
atribuí entera a la entrada. Sólo la voz real lo expuso.

Es el motivo por el que medir calidad antes de construir el motor era el orden
correcto: con el veredicto anterior habría cableado el engine sobre una ruta que
produce basura.

### Lo que sigue sin estar medido

- Velocidad. No se midió tiempo de transcripción ni RTF, ni en CPU ni en DirectML.
- Modelos más grandes que `tiny`. La aritmética de DirectML podría degradarse
  distinto en `base`, `small` o `large`.
- Idiomas distintos del inglés (se probó `whisper-tiny.en`).

## Un falso negativo que casi me hace descartar el camino

El primer chequeo que hice fue:

```python
TasksManager.get_supported_tasks_for_model_type("whisper", "onnx", library_name="transformers")
# KeyError: 'whisper is not supported yet ... Only [] are supported'
```

Leído solo, eso dice "whisper no se puede exportar". **Es falso.** El registro de
`TasksManager` se puebla de forma perezosa cuando se importa
`optimum.onnxruntime`; consultarlo antes devuelve vacío **para todos** los model
types, no solo para whisper.

Lo que lo detectó fue meter un **control** en el spike: consultar también `bert`,
que sin duda está soportado. Con `bert` fallando igual, el "Only []" queda expuesto
como registro no inicializado y no como veredicto.

Es la sexta vez en la sesión que aparece el mismo patrón —
ausencia de señal leída como veredicto — y la primera en que el control lo atrapa
antes de que costara trabajo. El spike conserva ese control a propósito.

## Consecuencias de diseño

1. **Transcripción y subtítulos son dos capacidades distintas**, en dominios
   distintos:
   - `audio.transcribe`: audio → texto. Es la capacidad base.
   - `video.subtitles`: ese mismo resultado, alineado en el tiempo y muxeado al
     contenedor. Depende de la primera y agrega el paso de sincronizado.
2. **El modelo entra por el registro, no por un paquete vendorizado.** Un repo
   ONNX ya exportado se instala directo (`ready_onnx`); uno de PyTorch se puede
   convertir, porque el exportador soporta la tarea. Es el mismo par de caminos
   que generación ya tiene, así que reusa `CompatStrategy`, el pre-flight y la
   tarjeta.
3. **Va a necesitar su propia `CompatStrategy`.** La clasificación de un repo de
   ASR no es la de difusión ni la de upscalers: lo que hay que buscar es el par
   encoder/decoder de whisper, no un `model_index.json` ni un `.pth` suelto.

## Lo que sigue

1. `AsrCompatStrategy` con sus tests, y `strategy_for("audio")` dejando de tirar
   `ValueError`.
2. Un motor `TranscribeEngine` sobre `ORTModelForSpeechSeq2Seq`, con el mismo
   patrón de caché por `(model_id, device)` que el de generación — y con el modo en
   la clave desde el principio, que en generación fue un bug.
   **Obligatorio: `use_merged=False`.** Con el default, DirectML devuelve basura sin
   fallar. Y un test que transcriba voz real y verifique el texto, porque un test
   que sólo chequee "devolvió string" habría pasado con la basura.
3. El job de transcripción, con el audio de entrada y el idioma.
4. Recién después, subtítulos: el paso de sincronizado más el muxeo, que ya sabe
   hacer el pipeline de video.
5. Medir **velocidad** antes de prometer que sirve para un video largo, y volver
   a medir calidad al pasar de `tiny` a un modelo más grande.
