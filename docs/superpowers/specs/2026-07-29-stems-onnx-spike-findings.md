# Spike de viabilidad: separación de stems sobre ONNX

**Fecha:** 2026-07-29
**Veredicto: NO hace falta un motor de inferencia nuevo, pero la separación
todavía NO produce audio usable. No se puede construir el motor todavía.**

## Corrección de una afirmación previa mía

El spec del gestor unificado clasificó stems como **nivel 3**: *"necesita un modelo
tipo Demucs y un motor de inferencia que hoy la app no incluye"*. Lo repetí varias
veces durante la sesión.

**Es falso**, y lo escribí antes de aprender cuánto cubre ONNX Runtime. El error fue
razonar desde el nombre del modelo (Demucs, que es PyTorch) en vez de preguntar qué
formatos existen para la tarea.

## Lo medido

Modelo: `masszhou/mdxnet`, archivo `UVR-MDX-NET-Voc_FT.onnx`, **63,7 MB**.

| Pregunta | Respuesta |
|---|---|
| ¿Lo carga `onnxruntime.InferenceSession` crudo? | **Sí** |
| ¿Sobre DirectML? | **Sí** |
| ¿Sobre CPU? | Sí |
| ¿Hace falta optimum? | **No** |
| ¿Hace falta una dependencia nueva? | **No** |

Firma del grafo, idéntica en los dos proveedores:

```
entrada: input   [batch_size, 4, 3072, 256]  float32
salida:  output  [batch_size, 4, 3072, 256]  float32
```

La app ya trae `onnxruntime-directml` 1.24.4, y ya corre sesiones ONNX crudas para
los upscalers. Así que el runtime **ya está**.

## Qué hace falta de verdad

No un motor: **procesamiento de señal**. La forma del tensor lo dice todo — es un
espectrograma, no audio:

- `4` canales = 2 de estéreo × (real, imaginario)
- `3072` bins de frecuencia ⇒ **n_fft = 6144**
- `256` frames por ventana

Entonces el trabajo es:

1. **STFT** del audio estéreo con `n_fft=6144`, quedarse con 3072 bins, separar real
   e imaginario en 4 canales.
2. **Trocear** en ventanas de 256 frames con solape, igual que el troceo de 30 s de
   whisper resultó obligatorio ahí. Sin solape hay artefactos en los bordes; con
   solape hay que hacer overlap-add.
3. Correr la sesión ONNX.
4. **STFT inversa** y overlap-add para volver a audio.

`numpy` 2.4.6 y `scipy` 1.17.1 ya están instalados, así que la STFT no agrega nada.

Es la misma categoría de trabajo que la cadena de mejora de voz: DSP con su
matemática, no infraestructura nueva.

## Segunda medición: se corrió de verdad, y NO funciona todavía

`scripts/spike_stems_onnx.py --separate`. Tres controles en orden, cada uno
invalidando al siguiente si falla.

### Control 1 — round-trip de la STFT, SIN el modelo: PASA

```
forma del espectrograma : [4, 3072, 256]   coincide con la firma del grafo
correlación round-trip  : 1.0
```

Esto **confirma los parámetros**, que hasta acá eran conjetura: `n_fft = 6144`,
`dim_f = 3072`, `hop = 1024`, 267264 muestras por trozo. Un round-trip perfecto no
deja lugar a duda sobre esa parte.

El control existe por el mismo motivo que el control con `bert` en el spike de
whisper: si mi STFT estuviera mal, no habría forma de saber si un mal resultado es
del modelo o mío.

### Controles 2 y 3 — la sesión corre, la salida NO sirve

```
                        DirectML    CPU
forma de salida         [4,3072,256]  igual
rms original            0.02594     0.02594
rms separado            1.27573     1.27573   (~49x mas fuerte)
correlación c/original  0.0078      0.0078
```

La entrada es **voz sola** y el modelo apunta a la voz, así que una salida correcta
tendría correlación alta: no hay instrumental que sacar. Da **0.008**, o sea nada.

**El dato que orienta: los dos proveedores dan el resultado IDÉNTICO.** Con whisper,
DirectML difería de CPU y eso señalaba un problema de aritmética del runtime. Acá
coinciden, así que es determinista y el error es **mío**, en cómo alimento o
interpreto el modelo — no del runtime.

### Hipótesis descartadas

Probadas y todas fallan; no vale la pena repetirlas:

| Interpretación de la salida | Correlación |
|---|---|
| Es el espectrograma del stem, directo | 0.0078 |
| Es una máscara que multiplica la entrada | 0.0051 |
| Es una máscara acotada a [0,1] | 0.0569 |

La salida tiene magnitud ~2.4× la entrada, así que tampoco es un problema de
ganancia simple: un error de escala mantendría la correlación alta.

### El sospechoso principal para la próxima vuelta

UVR usa `torch.stft`, que por defecto va con **`center=True`**: rellena `n_fft//2`
en los dos extremos. Mi STFT no centra. Eso cambia la alineación de los frames y su
cantidad, y explicaría una correlación nula mucho mejor que un error de escala.
Confirmarlo requiere reimplementar la STFT centrada y volver a medir.

Otros candidatos, en orden: normalización del audio antes de la STFT (UVR normaliza),
y el factor `compensate` que UVR aplica por modelo.

## Por qué esto NO se construye todavía

El motor de transcripción se construyó **después** de verificar que Whisper
transcribía `"I know Kung Fu."` correctamente. Ese orden atrapó que el decoder merged
devolvía basura en DirectML.

Acá la verificación **falló**, así que construir el motor ahora significaría shipear
una capacidad que produce ruido. Lo que falta es una vuelta más de spike sobre la
convención de entrada/salida del modelo, no código de producción.

## Lo que sigue sin medirse

- **Velocidad.** Sin medir.
- **Qué stems da cada modelo.** `Voc_FT` es voz contra instrumental (2 stems), no los
  4 clásicos de Demucs (voz, batería, bajo, otros). Cuántos stems ofrece la capacidad
  depende del modelo elegido, y eso hay que exponerlo en la UI en vez de prometer 4.
- **Los parámetros exactos** (hop length, tipo de ventana, compensación de ganancia)
  salen de la implementación de UVR y hay que confirmarlos contra ella, no inferirlos.

## Consecuencia para el catálogo

El motivo que hoy muestra `audio.stems` dice que falta un motor de inferencia. Hay
que reescribirlo: lo que falta es el procesamiento de espectrograma y la validación
de calidad, no el runtime. Prometer menos trabajo del que hay sería igual de malo que
prometer más.
