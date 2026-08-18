# AudioSR en fp16: la mitad del disco, 9% más rápido, y por qué NO se cambió solo

Medido 2026-08-17 en RX 7800 XT, `onnxruntime-directml` 1.24.4, con la receta que
salió del port de RoFormer (`port-bs-roformer-onnx/docs/fp16-findings.md`).
Spike reproducible: `scripts/spike_audiosr_fp16.py`.

AudioSR es el modelo de audio más pesado de la app: cuatro grafos, 2.51 GiB, y el
UNet `ddpm` corre una vez por paso DDIM por ventana. Era el mejor candidato para
la misma palanca que en el modelo de 4 stems dio 12x.

## Qué dio

| | fp32 | fp16 |
|---|---:|---:|
| `vocoder` | 725.9 MiB | 363.0 MiB |
| `vae_decoder` | 509.4 MiB | 254.8 MiB |
| `vae_feature_extract` | 343.4 MiB | 171.8 MiB |
| `ddpm` | 989.8 MiB | 497.0 MiB |
| **total** | **2.51 GiB** | **1.26 GiB** |
| restauración de 5.1 s de audio, 50 pasos | ~6.5 s | ~5.9 s |

**Fidelidad: 59.4 dB SI-SDR** entre la salida fp16 y la fp32, diferencia máxima
0.0007 sobre un pico de 0.5. Inaudible. Para que ese número signifique algo se
verificó primero que el pipeline es determinista: dos corridas fp32 seguidas dan
**232.6 dB** entre sí (el driver siembra su RNG en 42), así que los 59.4 dB son
el costo de fp16 y no ruido de difusión.

## La medición de velocidad, que casi sale mal

La primera lectura fue "1.66x más rápido" (13.1 s fp32 contra 7.9 s fp16). **Era
falsa**: la corrida fp32 era la primera del día y pagó el calentamiento —caché de
página de 2.5 GB de modelo más compilación de shaders del driver—. La segunda
corrida fp32, idéntica, tardó 7.1 s.

Con calentamiento pagado la varianza corrida a corrida (6.4 a 10.1 s) resultó
más grande que el efecto que se quería medir, así que una sola comparación no
alcanzaba. A/B pareado, 6 pares, orden ABBA balanceado:

| par | fp32 | fp16 | delta |
|---|---:|---:|---:|
| 1 | 6.4 s | 5.7 s | -10.9% |
| 2 | 6.5 s | 5.7 s | -12.3% |
| 3 | 6.4 s | 5.9 s | -7.8% |
| 4 | 6.5 s | 6.0 s | -7.7% |
| 5 | 6.6 s | 6.0 s | -9.1% |
| 6 | 6.5 s | 5.9 s | -9.2% |

6 de 6 pares a favor de fp16, mediana **-9.2%**. Real, consistente, y un orden de
magnitud menos que lo que decía la primera medición.

## Por qué 9% acá y 12x en el modelo de 4 stems

Porque la ganancia de fp16 no es aritmética, es de memoria. El grafo de 4 stems
sostiene un intermedio de atención de ~1.3 GB que no le entra cómodo en VRAM: ahí
partir el tamaño a la mitad cambia el régimen entero. Los grafos de AudioSR ya
entraban, así que fp16 solo ahorra ancho de banda al margen. **La regla que queda:
esperar el 2x aritmético solo si el grafo está limitado por memoria; si ya entra,
esperar un dígito.**

## Por qué no se cambió y qué haría falta

El ahorro de 1.25 GiB en disco (y aproximadamente la mitad de VRAM, lo que además
afloja la admisión por capacidad) es un beneficio real y la pérdida de calidad es
inaudible. Pero convertir el pack a fp16 y listo **rompería a los usuarios de CPU**:
el EP de CPU tiene muchos menos kernels fp16 —en el port de RoFormer,
`ScatterElements` con `reduction='add'` directamente no existe en fp16 para CPU— y
donde existen suelen ser más lentos que los fp32. Upflow corre estos grafos tanto
en GPU como en CPU.

O sea que hacerlo bien es elegir por dispositivo, y eso implica tener AMBOS packs
en disco, que es exactamente el ahorro que se buscaba. Las salidas posibles:

1. **Convertir en la instalación solo si el dispositivo por defecto es GPU** y
   guardar únicamente el fp16. Conserva el ahorro completo; cuesta ~35 s de
   conversión al instalar y deja al usuario de CPU sin el pack óptimo si después
   cambia de dispositivo.
2. **Publicar los grafos fp16 como assets aparte** en `port-audiosr-onnx` y bajar
   los que correspondan al dispositivo. Ahorra también la descarga (2.51 → 1.26 GiB)
   y es la mejor experiencia; requiere publicar y versionar assets nuevos.
3. **No hacer nada.** 9% no justifica el riesgo por sí solo.

**Se tomó la 2, autorizada por el dueño el 2026-08-17.** Los grafos fp16 se
publicaron en
[`models-fp16-v1.0`](https://github.com/santiquiroz/port-audiosr-onnx/releases/tag/models-fp16-v1.0)
con los mismos nombres de archivo que los fp32, así que el motor no cambió: lo
único que decide es de qué release baja el script.

Cómo quedó:

- `scripts/download-audiosr-onnx.ps1 -Precision fp32|fp16`. Si ya hay un pack de
  la otra precisión instalado, borra los grafos viejos antes de bajar: mezclar
  las dos deja archivos de ambas y un manifest que describe solo una.
- `pack_provisioner.default_variant()` elige por el dispositivo por defecto —
  fp16 en GPU, fp32 en CPU — y el botón de la tarjeta la pasa. Nadie tiene que
  saber qué es fp16 para beneficiarse.
- El manifest fp16 declara `precision` y sus `required_files` incluyen los tres
  `.data` nuevos, así que `AudioSrAssets.is_complete` valida el pack correcto sin
  tocar código.
- La contradicción que quedaba —pack elegido al instalar, dispositivo elegido por
  trabajo— se cierra con una guarda: correr un pack fp16 en CPU falla ANTES de
  crear la sesión, con un mensaje que dice reinstalar en fp32. Sin eso el usuario
  vería un error de kernel de ONNX Runtime a mitad de la corrida.

El manifest fp16 lleva además el SHA-256 de cada asset y el de los grafos fp32 de
los que salió, así que la procedencia se verifica en las dos direcciones.
