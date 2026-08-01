# Generación de video local sin CUDA — hallazgos verificados (2026-08-01)

Todo lo de acá se corrió en esta máquina (RX 7800 XT, 16 GB, Windows 11) contra el
binario que Upflow ya distribuye. Lo que no se midió está marcado como no verificado.

## El hallazgo principal

`vendor/sdcpp/sd-cli.exe` (stable-diffusion.cpp, commit `e31a86c`) — el mismo binario
que la v0.26.0 ya usa para generar imágenes por Vulkan — **soporta generación de video
nativamente**. No hace falta ComfyUI, ni ROCm, ni ZLUDA, ni exportar a ONNX.

Verificado corriendo `sd-cli.exe --help` y leyendo los flags:

| Capacidad | Flags |
|---|---|
| Modo video | `-M vid_gen`, `--video-frames N`, `--fps N` |
| Salida de video directa | `-o out.webm` / `.avi` / `.webp` animado |
| Wan 2.1 / 2.2 | `--vae-format wan`, `--flow-shift` |
| Wan 2.2 MoE (14B) | `--high-noise-diffusion-model`, `--moe-boundary`, `--high-noise-steps` |
| Imagen→video | `-i init.png` |
| Primer y último frame | `--init-img` + `--end-img` (flf2v) |
| Video→video guiado | `--control-video <dir>`, `--vace-strength` |
| LTX-2 con audio | `--embeddings-connectors`, `--audio-vae`, `--llm`, scheduler `ltx2` |
| Memoria | `--offload-to-cpu`, `--max-vram`, `--stream-layers`, `--params-backend disk`, `--vae-tiling`, `--temporal-tiling` |
| Reparto por dispositivo | `--backend te=cpu,diffusion=vulkan0,vae=vulkan0` |

## El pack que se descargó y funciona

Wan 2.2 TI2V-5B. Un solo modelo hace texto→video **y** imagen→video. Apache 2.0, sin gate.

| Archivo | Tamaño | Origen |
|---|---|---|
| `Wan2.2-TI2V-5B-Q8_0.gguf` | 5,40 GB | `QuantStack/Wan2.2-TI2V-5B-GGUF` |
| `Wan2.2_VAE.safetensors` | 1,41 GB | mismo repo, carpeta `VAE/` |
| `umt5-xxl-encoder-Q5_K_M.gguf` | 4,15 GB | `city96/umt5-xxl-encoder-gguf` |

Total 10,96 GB. Viven en `vendor/sdcpp/models/video/`, que **no** contamina la lista de
modelos de imagen: `list_sdcpp_models` recorre el directorio con `iterdir()` + `is_file()`,
así que ignora subcarpetas.

## El error que hay que evitar en AMD (medido, no supuesto)

Primer intento, con todo en Vulkan:

```
ggml_vulkan: Device memory allocation of size 4200595456 failed.
ggml_vulkan: Requested buffer size exceeds device buffer size limit: ErrorOutOfDeviceMemory
[ERROR] t5 prepare graph weights failed
GGML_ASSERT(!chunk_hidden_states.empty()) failed
```

**No es falta de VRAM.** Son 4,20 GB pedidos en *un solo buffer* y el driver Vulkan de
AMD topea el buffer individual en 4 GiB. La GPU tenía los 16 GB libres.

Arreglo: mandar el text encoder a CPU con `--backend te=cpu,diffusion=vulkan0,vae=vulkan0`.
Se usa una vez por generación, así que el costo es acotado (16,8 s por prompt medidos).

Consecuencia para el instalador: cualquier encoder de texto grande (umt5-xxl, gemma-12b de
LTX-2) va a CPU por defecto en el lane Vulkan. No es opcional en AMD.

## Lo que se midió (RX 7800 XT, 16 GB, Vulkan)

Todas las corridas con el mismo prompt y el encoder de texto en CPU.

| Config | Difusión | Decode | Total | Video |
|---|---|---|---|---|
| Base, 480×480, 17f, 10 pasos, CFG 6, VAE nativo | 238,3 s | 85,8 s | **355,6 s** | quemado y borroso |
| Turbo, 480×480, 17f, 4 pasos, CFG 1, VAE nativo | 27,9 s | 85,8 s | **126,7 s** | sujeto nítido, fondo con bloques |
| Turbo, 832×480, 33f, 4 pasos, CFG 1, **TAE** | 97,2 s | **5,4 s** | **116,8 s** | fondo limpio, luz cinematográfica |
| Turbo, 704×704, 17f, VAE nativo + tiling | 64,6 s | se cayó a CPU | **>25 min, matado** | — |

Dos conclusiones que cambian el diseño:

1. **El destilado Turbo no es opcional.** 4 pasos con CFG 1 (una sola pasada, sin
   prompt negativo) contra 10 pasos con CFG 6: sampling 8,5× más rápido **y** mejor
   imagen. Con los defaults de imagen (25 pasos, CFG 7,5) el modelo sale quemado.
2. **El decode del VAE de Wan es el cuello, no la difusión.** Pide 6,36 GB de VRAM a
   480×480 y a 704×704 ya no entra: ggml se cae a CPU y el job pasa de 2 minutos a
   más de 25. El TAE (23 MB) lo baja a 5,4 s y es lo que permite 832×480 con 33
   cuadros. Sin TAE, el techo práctico en 16 GB es 480×480.

Punto de comparación honesto: una RX 7600 XT (misma generación) publicada con ROCm y
el VAE en CPU tardó 27 min 15 s para 2 s de video. Acá son 116,8 s para ~2 s.

## Comando de referencia

```
sd-cli.exe -M vid_gen \
  --diffusion-model models/video/Wan2.2-TI2V-5B-Q8_0.gguf \
  --vae            models/video/Wan2.2_VAE.safetensors \
  --t5xxl          models/video/umt5-xxl-encoder-Q5_K_M.gguf \
  --backend te=cpu,diffusion=vulkan0,vae=vulkan0 \
  -p "<prompt>" -n "<negativo>" \
  --cfg-scale 6.0 --sampling-method euler --flow-shift 3.0 --steps 10 \
  -W 480 -H 480 --video-frames 17 --fps 16 \
  --diffusion-fa --offload-to-cpu \
  -o salida.webm
```

`--video-frames` conviene que sea `4n+1` (17, 33, 49...) por la compresión temporal del VAE.
Para imagen→video se agrega `-i imagen.png` y nada más cambia.

## Lo que NO se va a construir

Reemplazo de rostros / face swap. Es decisión mía, no una limitación técnica: un sistema de
sustitución de caras dentro de una app que ya genera desnudos sin filtro es, en la práctica,
herramienta para imágenes sexuales no consentidas de personas reales.

## Pendiente

- **Frontend**: el lane está entero en backend (motor, cola, API, capacidades) con
  tests, pero todavía no hay pantalla. Falta el formulario en Generate y el `<video>`
  para reproducir el `.webm`.
- **Serializar contra los otros jobs de GPU.** Dos difusiones a la vez en la misma
  placa se degradan mutuamente (medido en otro contexto: 24 s/it → 60 s/it). Los jobs
  de video deberían tomar el semáforo de dispositivo como los de upscaling.
- **Degradación temporal**: en el clip de 33 cuadros la textura del sujeto se
  desarma hacia el final. Falta probar si es el TAE, los 4 pasos, o el modelo.
- **Componente del instalador** para el pack (10,9 GB) y su entrada en `upflow.iss`.
- **LTX-2** no se probó: 22B más un Gemma-3-12B como encoder no entran cómodos en 16 GB.
- **Generar chico y arreglar después**: Upflow ya tiene Real-ESRGAN y RIFE/GMFSS
  adentro. Generar a 480p/16 fps y subirlo a 1080p/48 fps con el pipeline propio es
  mucho más barato que pedirle la resolución final al modelo. Es el diferencial real
  contra ComfyUI pelado y todavía no está cableado.
