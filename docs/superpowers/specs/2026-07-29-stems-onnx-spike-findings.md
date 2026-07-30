# Spike de viabilidad: separación de stems sobre ONNX

**Fecha:** 2026-07-29
**Veredicto: NO hace falta un motor de inferencia nuevo.**

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

## Lo que sigue sin medirse

- **Calidad.** No se separó ni un archivo todavía. La lección de whisper aplica
  directo: DirectML cargó y corrió, y devolvía basura. **Cargar no es funcionar.**
  Hay que separar audio real y escuchar el resultado antes de prometer nada.
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
