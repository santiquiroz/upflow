# Apartado de descargas — diseño

Fecha: 2026-07-30. Todo lo marcado **medido** se verificó en esta máquina hoy; lo demás está
marcado como documentado o pendiente.

## Lo que se midió, no se supuso

| Hecho | Cómo se verificó |
|---|---|
| yt-dlp es **The Unlicense** (dominio público, cero obligaciones) | Leído el `LICENSE` del paquete instalado, no un blog |
| yt-dlp 2026.07.04 **extrae YouTube bien hoy** | 33 formatos, **23/23** de video con URL usable, alturas hasta 2160, **cero** marcadores SABR |
| La cadena completa funciona | Descarga + **merge por ffmpeg** (av1 video + aac audio → un mp4) en 4,9 s |
| Los hooks de progreso dan bytes y total | `downloaded_bytes` y `total_bytes` presentes |
| **`ffmpeg_location` NO alcanza** | Falló con "ffmpeg is not installed" pese a `available=True` en el postprocesador. El camino de descarga parcial usa OTRO chequeo. Hay que poner el bin vendorizado **también en PATH** |
| ffmpeg ya está vendorizado en Upflow | `vendor/ffmpeg/bin/{ffmpeg,ffprobe,ffplay}.exe` |
| No existe librería embebible que envuelva yt-dlp | PyPI: los wrappers son CLI (`yt-dlp-helper`) o **forks** (`yt-dlp-enhanced-progress`, `yt-dlp-ffmpeg-progress`) |

Sobre el pánico de SABR/PO tokens: los issues son reales pero **condicionales** — clientes
específicos, algunos videos, algunas redes. No es un "YouTube está roto" general. La extracción
básica anda hoy sin login y sin PO token. Eso no prueba que ande para video con edad restringida,
solo para miembros, en vivo, o YT Music: **no medido**, no se afirma.

## Decisiones

### 1. Motor: yt-dlp embebido como librería, NO forkeado

Unlicense permite todo, incluso redistribuir dentro del instalador. Forkear es un callejón: yt-dlp
libera casi mensualmente para seguirle el paso a las defensas de YouTube, y un fork queda atrás
justo cuando más importa. Es exactamente el error que cometieron `yt-dlp-enhanced-progress` y
`yt-dlp-ffmpeg-progress`.

Librería embebida (`yt_dlp.YoutubeDL`) y no subprocess, porque los hooks de progreso y la
cancelación son de primera clase en la API de Python. El costo es acoplarse a su API interna, que
mitigamos aislándola detrás de una capa fina.

### 2. NO forkear cobalt

42K estrellas y la mejor UX del rubro, pero: **AGPL-3.0** con una licencia **distinta y más
restrictiva en `web/`**, y stack Svelte contra nuestro React/FastAPI. Se toma la *filosofía de
producto* (pegar URL → recibir archivo, sin publicidad, sin fricción), no el código.

### 3. Repo público propio: sí, pero como LIBRERÍA, no como otra app

Otro "web UI para yt-dlp" no aporta nada: metube (14K), pinchflat, cobalt (42K) y Parabolic ya
existen y están bien hechos. Publicar el número cinco sería ruido.

Lo que **no** existe es una librería Python embebible que resuelva el pegamento que todos
reimplementan:

- progreso estructurado (no parsear stdout)
- cancelación cooperativa a mitad de descarga
- cola con límites
- provisioning de ffmpeg **incluido el detalle del PATH que medimos**
- auto-update de yt-dlp sin romper la app
- salud por sitio, para poder decir "esto se rompió del lado de ellos" en vez de un stacktrace

Eso se publica como repo propio y Upflow lo consume. Doble beneficio: la app no carga con el
pegamento y el pegamento se prueba solo.

### 4. El diferenciador: descargar → mejorar en un solo flujo

cobalt y metube te entregan un archivo. Upflow puede **descargar y después escalar, interpolar,
limpiar el audio o transcribir** con el pipeline que ya existe. Nadie más tiene eso, y es el
argumento real para construirlo acá y no simplemente instalar metube.

Implementación: la descarga aterriza en el directorio de uploads con un token, o sea exactamente
la misma entrada que un archivo subido a mano. El resto del pipeline no se entera.

### 5. Postura legal: local por diseño

El usuario pidió uso interno, sin exponer a internet. Salvaguardas concretas:

- Nada de instancia pública. Igual que el resto de Upflow: escucha local.
- No eludir DRM. yt-dlp ya se niega; hay que **mostrar ese motivo** en vez de un error genérico.
- No agregar extractores de sitios de piratería.
- Sin telemetría; el registro queda local.
- Techo de playlist explícito, para que nadie dispare 500 videos sin querer.

## Forma del módulo

Sigue el patrón que ya usan los otros cinco módulos:

```
app/models.py                        DownloadJob (dataclass)
app/services/fetch_options.py        request -> dict de opciones de yt-dlp   [PURO, testeable sin red]
app/services/fetch_engine.py         capa fina sobre yt_dlp.YoutubeDL       [mockeable]
app/services/download_job_manager.py cola y ciclo de vida
app/api/routes.py                    /api/v1/download/*
frontend/src/modules/download/       la UI
```

`fetch_options.py` primero y con tests: ahí vive la lógica de verdad (formato, techo de altura,
contenedor, playlist, subtítulos) y se prueba entera sin tocar la red.

## Qué se le pregunta al usuario y qué se decide por él

| Decisión | Quién |
|---|---|
| URL | usuario |
| Altura máxima | usuario, con un techo por defecto sano (1080p) — no 4K por accidente |
| Solo audio | usuario, es un pedido claramente distinto |
| Contenedor | la app: mp4 salvo que haga falta mkv |
| Subtítulos y metadata | la app: se traen por defecto, es lo que la gente espera |
| Playlist completa | usuario, **explícito**, con tope |
| Formato exacto, códec | la app. Pedirlo es la trampa de UX de todos los descargadores |

## Medido después (los dos pendientes que sostenían el diseño)

**Progreso, granularidad real:** descarga de 124 MB → **180 eventos `downloading`**, todos con
`downloaded_bytes`, estrictamente monótonos crecientes, de 1.024 a 124.386.876 bytes. Eso es una
muestra cada ~0,7 MB: sobra para una barra de progreso honesta. El test anterior dio un solo evento
solo porque el archivo era diminuto.

**Cancelación a mitad:** lanzar una excepción desde el hook de progreso corta la descarga limpio. La
excepción propaga con la causa raíz intacta (se puede distinguir "canceló el usuario" de "falló"), y
deja un `.part` identificable para limpiar. Cortó exactamente en el evento 3, como se pidió.

O sea: las dos capacidades sobre las que se apoya la librería están verificadas, no supuestas.

## "Cualquier web": qué tan cierto es

**1751 extractores registrados** en la versión instalada. De 15 hosts probados por matching
determinista (sin red, sin adivinar URLs), **13 tienen extractor propio**: Vimeo, SoundCloud,
Reddit, X/Twitter, TikTok, Instagram, Dailymotion, Twitch, Facebook, Bluesky, Bilibili, Rumble,
Odysee. LinkedIn y Google Drive caen al extractor **genérico**, que busca medios embebidos en
páginas arbitrarias.

Así que "cualquier web" es en gran medida cierto, con una salvedad que importa para el diseño: la
confiabilidad **por sitio** varía y se rompe sin aviso. Un intento en vivo de Vimeo falló con
`Failed to fetch macos OAuth token: HTTP Error 401` — un fallo del propio extractor, no de la URL.
Eso es exactamente el hueco que la librería debe cubrir: decir "esto se rompió del lado de ellos, en
este sitio, hoy" en vez de escupir un stacktrace.

### Nota de método (error propio, vale registrarlo)

El primer intento de probar sitios no-YouTube dio 5 de 5 fallos y por un momento pareció que "nada
fuera de YouTube funciona". Falso: **4 de esas 5 URLs las inventé**, así que los 404 eran la
respuesta correcta. Solo Vimeo era un fallo real. Un test con datos fabricados no mide nada, y leer
sus fallos como veredicto es la misma trampa que este proyecto ya pisó varias veces.

## Pendiente de medir antes de prometer

- PO token provider: no probado. Solo hace falta cuando el camino básico falla.
- Video con edad restringida, solo para miembros, en vivo, y YT Music: no probados.
- Vimeo: el 401 del OAuth necesita confirmarse con una URL real antes de afirmar que está roto.
