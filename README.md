# Image Upscaler AMD

Proyecto nuevo y aislado para reescalado de **imágenes y videos** con alta calidad en **Windows + AMD Radeon RX 7800 XT**.

## Qué incluye

- **Web UI** para imágenes y videos.
- **REST API** para que otras apps envíen trabajos y consulten su estado.
- **Cola de trabajos** separada para imágenes y videos.
- **Motor desacoplado** para usar Real-ESRGAN NCNN con Vulkan, ideal para AMD en Windows.
- **Pipeline de video** con FFmpeg: extracción de frames, upscale por lotes, preservación de audio y re-ensamblado final.
- Manejo más robusto de encode H.265/libx265 en Windows para evitar fallos por exceso de threads.
- Estructura pensada para crecer sin mezclar la UI con el motor ni la API.

## Por qué esta arquitectura

Tu máquina:

- **GPU:** AMD Radeon RX 7800 XT
- **CPU:** Ryzen 9 7900X3D
- **RAM:** 128 GB
- **SO:** Windows 11 Pro

Para AMD en Windows, la ruta más sensata para calidad + compatibilidad es **Real-ESRGAN NCNN Vulkan**. DirectML existe, pero hoy suele ser menos predecible para este tipo de flujo. Por eso la app está preparada para arrancar con NCNN/Vulkan y dejar el motor como componente intercambiable.

## Modelos disponibles

**Generales**
- **realesrgan-x4plus** → fotos, imágenes generales, mejor equilibrio calidad/detalle.

**Especializados en anime**
- **realesrgan-x4plus-anime** → anime, ilustración, line art estático.
- **realesr-animevideov3-x2** → anime/video con escala 2x.
- **realesr-animevideov3-x3** → anime/video con escala 3x.
- **realesr-animevideov3-x4** → anime/video con escala 4x.
- **realesr-animevideov3** → preset automático que selecciona x2/x3/x4 según la escala elegida.

La UI ahora expone estos modelos en un **dropdown** y la API también los publica en `GET /api/v1/engine`.

## Estructura

```text
image-upscaler-amd/
  app/
    api/
    services/
    templates/
    static/
    web/
    main.py
  scripts/
  tests/
  runtime/
```

## Flujo de trabajo

1. La imagen entra por la web o por la API.
2. Se guarda en `runtime/uploads/`.
3. Se crea un job.
4. Un worker procesa la cola respetando `GPU_CONCURRENCY`.
5. La salida se guarda en `runtime/outputs/`.
6. El cliente consulta estado y descarga el resultado.

## Instalación

```powershell
cd image-upscaler-amd
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\download-realesrgan.ps1
```

## Arranque

```powershell
cd image-upscaler-amd
.\.venv\Scripts\uvicorn app.main:app --host 127.0.0.1 --port 8090 --reload
```

## Endpoints

- `GET /` → web UI
- `GET /api/v1/health` → healthcheck
- `GET /api/v1/engine` → estado del motor/configuración/modelos/perfiles de video
- `POST /api/v1/jobs` → crear trabajo de imagen
- `GET /api/v1/jobs/{job_id}` → ver estado de imagen
- `GET /api/v1/jobs/{job_id}/download` → descargar imagen
- `POST /api/v1/video/jobs` → crear trabajo de video
- `GET /api/v1/video/jobs/{job_id}` → ver estado de video
- `GET /api/v1/video/jobs/{job_id}/download` → descargar video

## Ejemplo de uso por API

```bash
curl -X POST http://127.0.0.1:8090/api/v1/jobs \
  -F "file=@input.png" \
  -F "model_name=realesrgan-x4plus" \
  -F "scale=4" \
  -F "output_format=png"
```

## Notas de rendimiento

- Empieza con `GPU_CONCURRENCY=1` para evitar pelear VRAM y degradar latencia.
- Si luego quieres throughput, podemos probar `GPU_CONCURRENCY=2` y perfilar.
- La CPU y RAM sobran; el cuello real será el motor GPU.
- Para producción más pesada, el siguiente paso natural sería mover la cola a Redis y separar workers.

## Buenas prácticas ya aplicadas

- Proyecto aislado en carpeta nueva.
- Motor desacoplado de UI/API.
- Validación de entrada.
- Límite de tamaño de subida.
- Cola de trabajos en vez de bloquear request largos.
- Un solo backend para servir web y servicios.

## Qué mejoraría después

1. Persistencia real de jobs en SQLite/PostgreSQL.
2. Redis + workers separados para escala horizontal.
3. Métricas Prometheus/OpenTelemetry.
4. Catálogo formal de modelos con presets por tipo de imagen.
5. Batch processing y autenticación para API externa.
