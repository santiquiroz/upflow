# Dónde se pierde el 10 bits (medido 2026-08-05)

`scripts/spike_10bit.py`

## La pregunta

El pipeline decodifica a `rgb24` y encodea a `yuv420p`. Los dos son de 8 bits, o
sea que hay dos sospechosos. Cambiar el encoder no sirve de nada si la pérdida ya
pasó al decodificar, así que primero se mide cuál manda.

## Método

Un degradado suave encodeado en `yuv420p10le` — es donde el banding de 8 bits se
ve, porque 10 bits tienen 1024 niveles por canal contra 256. Se cuenta cuántos
valores distintos sobreviven leyendo el mismo archivo de las dos maneras.

## Resultado

| Camino de lectura | Niveles distintos |
|---|---|
| `rgb24` (el de hoy) | 220 |
| `rgb48le` (16 bits) | 840 |

**Leer a 16 bits recupera 3,8x más niveles: la pérdida empieza al DECODIFICAR.**

Encodear a 10 bits sin cambiar el decode no recuperaría nada — los cuadros ya
llegarían achatados.

## Qué puede escribir 10 bits en el ffmpeg que viaja con la app

Los cinco encoders probados aceptan `yuv420p10le`, incluidos los de hardware:

| Encoder | 10 bits |
|---|---|
| `libx265` | sí |
| `libx264` | sí |
| `hevc_amf` | sí |
| `av1_amf` | sí |
| `libsvtav1` | sí |

## Qué haría falta para soportarlo de punta a punta

1. Decodificar a `rgb48le` en `ffmpeg_frame_source.py` (hoy `rgb24`).
2. El camino ONNX puede llevarlo: normaliza a float32, y de 16 bits a float no se
   pierde nada. **El camino ncnn no**: es un binario externo que trabaja sobre
   imágenes de 8 bits.
3. Encodear con `-pix_fmt yuv420p10le` en `video_encoders.py`.

O sea que sería una capacidad del backend ONNX, no del pipeline entero, y habría
que decirlo en pantalla en vez de dejar que el usuario suponga.

## Lo que NO se midió

Cuánto se nota en material real. El degradado sintético es el peor caso a
propósito; un BDrip de anime en Hi10P puede perder mucho menos. Antes de pagar el
costo de un pipeline de 16 bits convendría medir sobre material de verdad.
