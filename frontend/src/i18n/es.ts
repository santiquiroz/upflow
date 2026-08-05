// Catalogo espanol. Los textos que ya existian en espanol (los que construimos
// entre v0.15.0 y v0.16.2) se conservan tal cual estaban: eran correctos, solo
// estaban en el lugar equivocado.
export const es = {
  // --- modulo de generacion ----------------------------------------------
  "generation.compat.readyOnnx": "ONNX listo",
  "generation.vulkan.install": "Instalar para Vulkan (rápido)",
  "generation.vulkan.installing": "Descargando…",
  "generation.vulkan.done": "Listo en Vulkan",
  "generation.vulkan.hint":
    "Corre el checkpoint tal cual en el motor Vulkan: solo se descarga, sin la conversión de ~40 min.",
  "generation.compat.needsConversion": "Requiere conversión",
  "generation.compat.singleFile": "Archivo único",
  "generation.compat.gated": "Acceso restringido",
  "generation.compat.incompatible": "Incompatible",
  "generation.compat.unknown": "Compatibilidad desconocida",
  "generation.precision.title": "Precisión",
  "generation.precision.download": "{{size}} de descarga",
  "generation.checkpoint.required":
    "Elegí primero qué archivo instalar",
  "generation.checkpoint.title": "Checkpoint",
  "generation.details.hide": "Ocultar detalles",
  "generation.details.show": "Ver detalles",
  "generation.preflight.loading": "Evaluando descarga y capacidad…",
  "generation.capacity.vramUnknown": "VRAM libre desconocida",
  "generation.capacity.vramFree": "{{free}} VRAM libre",
  "generation.capacity.ramFree": "{{free}} libre",
  "generation.warnings.ariaLabel": "Avisos de instalación",
  "generation.warning.degraded":
    "No se pudo evaluar este modelo. Podés instalarlo igual.",
  "generation.warning.gated":
    "Repo con acceso restringido: necesitás un token de Hugging Face y aceptar la licencia.",
  "generation.warning.incompatible.fallback":
    "No parece un pipeline diffusers.",
  "generation.warning.diskLow":
    "Quedan {{free}} libres en {{path}} y hace falta {{needed}}.",
  "generation.warning.diskLow.singleFile":
    "Convertir este checkpoint necesita ~{{peak}} de pico en {{path}} " +
    "(deja el checkpoint, el pipeline y el ONNX en disco a la vez) y quedan " +
    "{{free}} libres.",
  "generation.warning.ramLow":
    "La conversión carga el checkpoint completo en RAM: {{needed}} requeridos y {{free}} libres.",
  "generation.warning.deviceWontFit":
    "{{device}}: no entra. Necesita ~{{needed}} estimados a {{width}}×{{height}} " +
    "y tiene {{free}} libres.",
  "generation.warning.cpuOnly":
    "Sin GPU compatible: generar en CPU tarda varios minutos por imagen.",
  "generation.warning.cpuSlow":
    "En CPU tarda varios minutos por imagen.",

  // --- cadena de mejora de voz -----------------------------------------
  "voice.step.denoise.label": "Quitar ruido de fondo",
  "voice.step.denoise.description":
    "Saca el zumbido, el aire y el ruido constante. Va primero porque todo lo " +
    "que viene después trabaja mejor sobre una grabación limpia.",
  "voice.step.highpass.label": "Limpiar los graves",
  "voice.step.highpass.description":
    "Corta lo que está por debajo de la voz humana: golpes de mesa, pisadas, " +
    "viento. Además hace que el control de volumen no tenga que pelear con esos graves.",
  "voice.step.compress.label": "Nivelar el volumen",
  "voice.step.compress.description":
    "Acerca las partes muy fuertes a las muy suaves, para que no haya que subir " +
    "y bajar el volumen mientras se escucha.",
  "voice.step.presence.label": "Enfocar la voz",
  "voice.step.presence.description":
    "Realza la banda donde vive la inteligibilidad de la voz. Es lo que hace que " +
    "el diálogo se entienda por encima de la música y los efectos.",
  "voice.step.deesser.label": "Suavizar las eses",
  "voice.step.deesser.description":
    "Baja los sibilantes que suenan filosos y molestan al escuchar con " +
    "auriculares. Va casi al final, sobre la sibilancia que realmente quedó.",
  "voice.step.loudness.label": "Ajustar al destino",
  "voice.step.loudness.description":
    "Lleva el volumen general al valor que pide la plataforma donde va a " +
    "publicarse. Va siempre último, porque mide el resultado final: medirlo " +
    "antes daría un número que ya no sería cierto.",

  // --- panel de la cadena de voz ---------------------------------------
  "common.cancel": "Cancelar",
  "generate.continueAnyway": "Continuar igual",
  "generate.model.select": "Elegí un modelo…",
  "generate.model.converting": "{{name}} (convirtiendo… ~30-45 min)",
  "generate.video.fastModel": "{{name}} - rápido, 4 pasos",
  "generate.video.size.square": "640 x 640 (cuadrado)",
  "generate.video.size.fastest": "480 x 480 (más rápido)",
  "generate.video.size.vertical": "480 x 832 (vertical)",
  "generate.video.frames": "Cuadros",
  "generate.video.fps": "Cuadros por segundo",
  "generate.cpuConfirm":
    "No se detectó GPU compatible (DirectX 12). Generar en CPU tarda varios minutos " +
    "por imagen. ¿Continuar igual?",
  "generate.video.noModels":
    "Todavía no hay modelos de video instalados. Instalá el pack de generación de " +
    "video desde el instalador, o corré {{script}}.",
  "generate.video.duration": "Duración: {{seconds}} s. Generar video tarda minutos, no segundos.",
  "generate.video.driftWarning":
    " Pasados los 17 cuadros el sujeto se va deformando hacia el final.",
  "generate.video.initImageHint":
    "Opcional: con una imagen de partida, el clip arranca desde ella. Sin imagen, se " +
    "genera solo desde el texto.",
  "generate.upscaleOnFinish": "Escalar automáticamente al terminar",
  "download.thumbnailAlt": "Miniatura de {{title}}",
  "download.downloading": "Descargando",
  "download.cancel": "Cancelar",
  "download.readyForEnhance": "Ya podés escalarlo o limpiarle el audio desde Enhance.",
  "download.url": "Dirección del video",
  "download.probing": "Viendo qué hay…",
  "download.playlistNotice":
    "Es una lista de {{count}} elementos. Se van a descargar {{willDownload}}.",
  "download.playlistOverLimit": " Revisá el límite antes de seguir.",
  "download.whatToFetch": "Qué traer",
  "download.maxQuality": "Calidad máxima",
  "download.wholeList": "Bajar la lista completa",
  "download.listMax": "Máximo",
  "download.submit": "Descargar",
  "models.compat.legend": "Filtrar por compatibilidad",
  "models.compat.all": "Todos",
  "models.compat.ready": "Listos para usar",
  "models.compat.conversion": "Con conversión",
  "audio.page.subtitle": "Quita el ruido y restaura los artefactos de compresión de un archivo de audio.",
  "audio.dropzone": "Suelta un archivo de audio aquí, o haz clic para buscarlo",
  "audio.mode.none": "Ninguno",
  "audio.format.flac": "FLAC (recomendado)",
  "audio.format.flac.description": "Calidad sin pérdida, cerca de 50% más chico que WAV.",
  "audio.format.wav": "WAV",
  "audio.format.wav.description": "Sin pérdida, sin comprimir. Compatible con todo.",
  "audio.format.mp3": "MP3",
  "audio.format.mp3.description": "Con pérdida, el archivo más chico — solo si el tamaño importa más que la calidad.",
  "models.device.title": "Dispositivo por defecto",
  "models.device.loading": "Cargando la información del dispositivo…",
  "models.device.loadError": "No se pudo cargar la información del dispositivo.",
  "models.device.none": "No se detectó un dispositivo por defecto.",
  "models.device.info": "Se elige solo al arrancar; se puede cambiar por trabajo en Mejorar. Cambiarlo desde acá todavía no está soportado.",
  "models.status.converting": "Convirtiendo…",
  "models.builtin.cannotRemove": "Los modelos incluidos no se pueden borrar",
  "models.builtin.group": "Incluidos",
  "models.installed.loading": "Cargando los modelos instalados…",
  "models.installed.loadError": "No se pudieron cargar los modelos instalados.",
  "models.onnx.empty": "Todavía no hay modelos ONNX propios instalados — busca arriba para agregar uno.",
  "models.generation.empty": "Todavía no hay modelos de generación instalados.",
  "models.delete.aria": "Borrar {{name}}",
  "models.delete.failed": "No se pudo borrar el modelo.",
  "models.delete.title": "¿Borrar {{name}}?",
  "models.delete.description": "Esto borra el archivo del modelo del disco. No se puede deshacer.",
  "models.delete.cancel": "Cancelar",
  "models.delete.confirm": "Borrar",
  "models.hfSearch.empty": "Busca en Hugging Face un modelo ONNX de reescalado para instalar.",
  "models.install.starting": "Empezando la instalación…",
  "models.install.downloading": "Descargando…",
  "models.install.validating": "Validando…",
  "models.install.converting": "Convirtiendo…",
  "models.install.working": "Trabajando…",
  "models.install.convertingStage": "Convirtiendo — {{stage}}",
  "models.install.retry": "Reintentar",
  "settings.videoLimit.title": "Trabajos de video pesados",
  "settings.videoLimit.explanation": "El límite protege el disco, no la red: el pipeline escribe cada cuadro como imagen, así que un video de 2 GB puede volverse cientos de GB. Subirlo es seguro — antes de escribir un solo cuadro, Upflow calcula el pico real y rechaza el trabajo si no entra.",
  "settings.videoLimit.label": "Subida máxima de video",
  "settings.videoLimit.current": "Ahora: {{mb}} MB",
  "settings.videoLimit.save": "Guardar",
  "settings.videoLimit.saving": "Guardando…",
  "settings.videoLimit.saved": "Guardado",
  "settings.videoLimit.saveError": "No se pudo guardar el límite.",
  "upload.tooLarge.admin": "Un administrador puede subirlo en Ajustes.",
  "settings.section.engine": "Motor",
  "settings.section.capacity": "Capacidad",
  "settings.engine.name": "Motor",
  "settings.engine.binary": "Binario del motor",
  "settings.engine.defaultModel": "Modelo por defecto",
  "settings.engine.allowedScales": "Escalas permitidas",
  "settings.engine.gpuConcurrency": "Trabajos simultáneos en GPU",
  "settings.engine.imageQueueDepth": "Profundidad de la cola de imágenes",
  "settings.engine.videoQueueDepth": "Profundidad de la cola de video",
  "settings.engine.loading": "Cargando la información del motor…",
  "settings.engine.loadError": "No se pudo cargar la información del motor.",
  "settings.capacity.loading": "Cargando la información de capacidad…",
  "settings.capacity.loadError": "No se pudo cargar la información de capacidad.",
  "settings.page.subtitle": "Configuración actual de motor, dispositivo y capacidad.",
  "settings.levers.ok": "OK",
  "settings.levers.unavailable": "No disponible",
  "settings.levers.notApplicable": "No aplica",
  "settings.levers.needsAdmin": "Requiere administrador",
  "settings.levers.noCpuFallback": "Sin respaldo en CPU",
  "settings.levers.none": "No se detectaron palancas de capacidad en este equipo.",
  "settings.levers.loading": "Cargando las palancas de capacidad…",
  "settings.levers.loadError": "No se pudieron cargar las palancas de capacidad.",
  "settings.diagnostics.notScanned": "Sin escanear",
  "settings.diagnostics.none": "Todavía no se escaneó ninguna combinación de modelo ONNX y dispositivo.",
  "settings.diagnostics.loading": "Cargando el diagnóstico…",
  "settings.diagnostics.loadError": "No se pudo cargar el diagnóstico.",
  "settings.token.notConfigured": "Sin configurar",
  "settings.token.placeholder": "Escribe un token nuevo",
  "settings.token.saveError": "No se pudo guardar el token.",
  "settings.token.loadError": "No se pudo cargar el estado de la credencial.",
  "settings.device.loading": "Cargando la información del dispositivo…",
  "settings.device.loadError": "No se pudo cargar la información del dispositivo.",
  "enhance.video.dropzone": "Suelta un video aquí, o haz clic para buscarlo",
  "enhance.image.dropzone": "Suelta una imagen aquí, o haz clic para buscarla",
  "enhance.image.submit": "Agrandar",
  "enhance.batch.selected": "{{count}} archivos elegidos",
  "enhance.batch.submit": "Agrandar {{count}} archivos",
  "enhance.batch.pending": "Faltan {{count}} archivos por enviar…",
  "enhance.batch.failed": "{{count}} archivos no se pudieron enviar",
  "enhance.batch.failedOne": "1 archivo no se pudo enviar",
  "enhance.image.scaleAndFormat": "Escala y formato",
  "enhance.image.scaleAndFormat.tooltip": "Elige el multiplicador de resolución de salida y el formato del archivo.",
  "enhance.model.tooltip.video": "Elige el modelo de IA que reescala el video. Los modelos incluidos corren en ncnn/Vulkan; los ONNX pueden correr en CPU o GPU.",
  "enhance.model.tooltip.image": "Elige el modelo de IA que reescala la imagen. Los modelos incluidos corren en ncnn/Vulkan; los ONNX pueden correr en CPU o GPU.",
  "enhance.runtime.tooltip": "Elige qué backend corre el modelo. Automático toma el más rápido para tu GPU (ONNX/DirectML es ~2x más rápido en GPUs modernas para video); NCNN Vulkan es el respaldo portátil que corre en cualquier GPU.",
  "enhance.encoder.tooltip": "Cómo se codifica el video final. Software (x264/x265) da la mejor calidad por bit. Automático (GPU) usa el codificador por hardware de tu GPU (NVENC/AMF/QSV) — mucho más rápido en 4K a un costo chico de calidad y tamaño, con respaldo automático a software.",
  "enhance.device.tooltip": "Elige el dispositivo que corre el trabajo. Un dispositivo CPU no puede correr un modelo incluido (ncnn): eso necesita una GPU con Vulkan.",
  "enhance.profile.tooltip": "Un perfil es un preset que combina modelo, escala, códec y calidad ajustados a un tipo de contenido. Al elegir uno se completan los campos de abajo; igual podés cambiarlos.",
  "enhance.profile.select": "Elige un perfil…",
  "enhance.interp.tooltip": "Interpola cuadros extra para subir la tasa de cuadros del video, con un multiplicador fijo o apuntando a una tasa concreta. Solo un modo puede estar activo a la vez.",
  "enhance.audio.tooltip": "Conserva la pista de audio original, opcionalmente limpiada con reducción de ruido. Mejorarla exige conservarla.",
  "enhance.output.tooltip": "Ajusta el contenedor de salida, el códec de video, el preset del codificador y la calidad (CRF). Un CRF más bajo da más calidad y un archivo más grande.",
  "enhance.audio.flacHint": "Recomendado — FLAC sin pérdida automáticamente cuando la restauración de audio está activa.",
  "enhance.audio.aacHint": "Estándar, archivos más chicos, con una pérdida leve de calidad si la restauración está activa.",
  "enhance.device.requiresVulkan": "Necesita una GPU con Vulkan para este modelo (ncnn)",
  "enhance.device.autoRoutes": "Manda al dispositivo compatible menos ocupado",
  "enhance.device.loading": "Cargando los dispositivos…",
  "enhance.device.loadError": "No se pudieron cargar los dispositivos.",
  "enhance.model.noAi": "Sin IA (redimensionado clásico)",
  "enhance.model.notReady": "No está listo",
  "enhance.model.loading": "Cargando los modelos…",
  "enhance.model.loadError": "No se pudieron cargar los modelos.",
  "enhance.encoder.autoGpuHint": "Usa el codificador de la GPU — mucho más rápido en 4K (NVENC/AMF/QSV)",
  "enhance.runtime.onnxHint": "~2× más rápido en GPUs modernas para video",
  "enhance.encoder.softwareHint": "Compatible siempre — la mejor calidad por bit",
  "enhance.runtime.best": "El mejor backend para tu dispositivo",
  "enhance.runtime.ncnnHint": "Respaldo portátil — corre en cualquier GPU con Vulkan",
  "common.userMenu": "Menú de usuario",
  "job.card.uploading": "Subiendo",
  "job.notice.upscaleFailed": "No se pudo agrandar la imagen: este es el tamaño generado.",
  "job.notice.encoderFallback": "El codificador de la GPU falló y el trabajo lo terminó el procesador.",
  "job.notice.containerUpgraded": "Hubo que cambiar el contenedor.",
  "job.notice.slowerPipeline": "El camino rápido falló y lo terminó el lento.",
  "job.notice.masteringSkipped": "Se saltó la masterización.",
  "job.card.generatedImage": "Imagen generada",
  "job.card.selectFile": "Elige un archivo para empezar.",
  "job.detail.negativePrompt": "Prompt negativo",
  "common.update.available": "Hay una actualización",
  "common.update.dismiss": "Descartar el aviso de actualización",
  "audio.voice.tooltip": "Trabaja la voz en sí: quita ruido, empareja el volumen, enfoca el diálogo, suaviza las eses y ajusta la sonoridad a la que pide la plataforma de destino. Los pasos corren en un orden fijo porque cada uno trabaja mejor sobre lo que dejó el anterior.",
  "enhance.summary.model": "Modelo",
  "enhance.summary.device": "Dispositivo",
  "enhance.summary.selectModel": "Elige un modelo…",
  "enhance.summary.selectDevice": "Elige un dispositivo…",
  "enhance.audio.keepOriginal": "Conservar el audio original",
  "enhance.audio.requiresKeep": "Hace falta activar \"Conservar el audio original\".",
  "enhance.fpsBoost.title": "Más cuadros por segundo",
  "enhance.interp.gmfss": "GMFSS (calidad máxima, muy lento)",
  "enhance.subtitles.keep": "Conservar los subtítulos incrustados",
  "enhance.profile.loading": "Cargando los perfiles de video…",
  "enhance.profile.loadError": "No se pudieron cargar los perfiles de video.",
  "generate.unavailable": "La generación no está disponible en este equipo.",
  "models.generation.sectionTitle": "Modelos de generación (Stable Diffusion)",
  "download.audioQuality": "Calidad de audio",
  "download.audioQuality.best": "Mejor (VBR)",
  "upload.tooLarge": "Ese archivo pesa {{size}}. El límite es {{limit}}.",
  "transcribe.models.verified.title": "Modelo probado",
  "transcribe.models.verified.hint": "Verificado en esta app con voz real. Chico y rápido — un buen primer modelo.",
  "transcribe.models.verified.action": "Buscar {{repo}}",
  "generate.saved.title": "Tus prompts",
  "generate.saved.save": "Guardar este prompt",
  "generate.saved.namePlaceholder": "Ponle un nombre",
  "generate.saved.delete": "Borrar {{name}}",
  "generate.saved.empty": "Todavía no guardaste ninguno. Escribe un prompt y guárdalo para reusarlo.",
  "generate.saved.failed": "No se pudo guardar el prompt.",
  "generate.preset.title": "Prompts listos",
  "generate.preset.hint": "Completa el prompt. Después lo podés editar.",
  "generate.preset.portrait": "Retrato",
  "generate.preset.cinematic": "Cinematográfico",
  "generate.preset.anime": "Anime",
  "generate.preset.landscape": "Paisaje",
  "generate.preset.product": "Foto de producto",
  "generate.preset.restylePainting": "Convertir en pintura",
  "generate.preset.restyleAnime": "Convertir en anime",
  "generate.preset.restylePhoto": "Hacerlo fotorrealista",
  "generate.preset.videoSlowPan": "Paneo lento",
  "generate.preset.videoCloseup": "Primer plano",
  "generate.preset.videoNature": "Naturaleza",
  "job.queue.title": "Cola de trabajos",
  "job.queue.showAll": "Ver todos",
  "job.queue.empty": "No hay trabajos activos.",
  "job.queue.clearCompleted": "Limpiar terminados",
  "job.status.queued": "En cola",
  "job.status.processing": "Procesando",
  "job.status.completed": "Terminado",
  "job.status.cancelled": "Cancelado",
  "job.failed": "El trabajo falló.",
  "job.install.failed": "La instalación falló.",
  "generate.conversion.failed": "La conversión del modelo falló.",
  "generate.conversion.cancelled": "La conversión del modelo se canceló.",
  "job.download": "Descargar",
  "job.download.aria": "Descargar {{name}}",
  "job.dismiss.aria": "Quitar {{name}}",
  "job.cancel.aria": "Cancelar {{name}}",
  "job.viewDetails.aria": "Ver el detalle de {{name}}",
  "job.owner": "dueño: {{owner}}",
  "nav.voice": "Voz",
  "voice.convert.title": "Cambiar una voz",
  "voice.convert.subtitle": "Conserva lo que se dice y cambia quién lo dice. Dale una grabación y una muestra de la voz que querés.",
  "voice.convert.source": "Grabación a cambiar",
  "voice.convert.reference": "Muestra de la voz que querés",
  "voice.convert.run": "Cambiar la voz",
  "voice.convert.working": "Convirtiendo…",
  "voice.convert.failed": "No se pudo convertir la voz.",
  "voice.convert.limit": "Hasta {{seconds}} segundos.",
  "voice.tab.tts": "Desde texto",
  "voice.tab.convert": "Desde una grabación",
  "voice.tts.title": "Voz",
  "voice.tts.subtitle": "Convierte texto en habla en tu máquina. No sale nada del equipo.",
  "voice.tts.text": "Qué tiene que decir",
  "voice.tts.voice": "Voz",
  "voice.tts.speak": "Que lo diga",
  "voice.tts.working": "Generando…",
  "voice.tts.download": "Descargar el audio",
  "voice.tts.failed": "No se pudo generar la voz.",
  "nav.tasks": "Tareas",
  "nav.enhance": "Mejorar",
  "nav.audio": "Audio",
  "nav.transcribe": "Transcribir",
  "nav.download": "Descargar",
  "nav.generate": "Generar",
  "nav.editor": "Editor",
  "nav.models": "Modelos",
  "nav.realtime": "Tiempo real",
  "nav.settings": "Ajustes",
  "nav.users": "Usuarios",
  "nav.mainLabel": "Navegación principal",
  "nav.queueLabel": "Cola de trabajos",
  "realtime.preset.anime4k.label": "Anime4K",
  "realtime.preset.anime4k.description": "Para anime y dibujo. Shader puro, el más liviano de todos.",
  "realtime.preset.cunny.label": "CuNNy",
  "realtime.preset.cunny.description":
    "Red chica para anime, más detalle que Anime4K y todavía en tiempo real.",
  "realtime.preset.fsr.label": "FSR",
  "realtime.preset.fsr.description": "Para imagen real y juegos. Escala y afila, muy barato.",
  "realtime.preset.lanczos.label": "Lanczos",
  "realtime.preset.lanczos.description": "Sin IA. El más rápido y el que menos inventa.",
  "audio.mastering.preset.streaming.label": "Streaming",
  "audio.mastering.preset.streaming.description":
    "El volumen que piden Spotify y YouTube. Para música y video en general.",
  "audio.mastering.preset.broadcast.label": "Televisión y radio",
  "audio.mastering.preset.broadcast.description":
    "El estándar europeo de emisión (EBU R128). Más margen dinámico.",
  "audio.mastering.preset.voice.label": "Voz",
  "audio.mastering.preset.voice.description":
    "Para podcast o narración: suaviza las eses y empareja el nivel.",
  "realtime.title": "Tiempo real",
  "realtime.subtitle":
    "Reescala en vivo lo que estés mirando o jugando, en una ventana superpuesta. El " +
    "overlay lo presenta Magpie, un programa libre que corre aparte y que Upflow " +
    "configura y abre por vos. No hace falta instalar ningún driver.",
  "realtime.mode": "Modo de reescalado",
  "realtime.frameCap": "Límite de cuadros",
  "realtime.frameCap.none": "Sin límite",
  "realtime.open": "Abrir overlay",
  "realtime.opening": "Abriendo…",
  "realtime.startFailed": "No se pudo iniciar el overlay.",
  "realtime.opened":
    "Overlay abierto. Poné el foco en la ventana que querés escalar y usá el atajo que " +
    "muestra la ventana de Magpie. El mismo atajo lo cierra.",
  "realtime.install": "Instalalo tildando Tiempo real en el instalador, o corriendo {{script}}.",
  "realtime.notViable.title": "Lo que todavía no se puede",
  "realtime.notViable.frameGen":
    "Generar cuadros intermedios como hace Lossless Scaling: hoy no existe una opción " +
    "de código abierto en Windows.",
  "realtime.notViable.afmf":
    "AFMF de AMD no se puede manejar desde una app externa: es un interruptor del driver, sin API pública.",
  "realtime.notViable.fidelityfx":
    "FidelityFX Frame Interpolation necesita vectores de movimiento que solo puede dar el motor del juego.",
  "realtime.techDetail": "Ver el detalle técnico",
  "common.loading": "Cargando…",
  "common.close": "Cerrar",
  "auth.username": "Usuario",
  "auth.password": "Contraseña",
  "auth.signIn": "Ingresar",
  "auth.login.failed": "No se pudo iniciar sesión",
  "auth.setup.title": "Configuración inicial",
  "auth.setup.subtitle": "Crea la cuenta de administrador de Upflow.",
  "auth.setup.submit": "Crear administrador",
  "auth.setup.failed": "No se pudo crear el administrador",
  "auth.setup.done": "Administrador creado. Recarga la página para iniciar sesión.",
  "auth.password.change.title": "Cambia tu contraseña",
  "auth.password.current": "Contraseña actual",
  "auth.password.new": "Contraseña nueva",
  "auth.password.change.submit": "Guardar",
  "auth.password.change.failed": "No se pudo cambiar la contraseña",
  "users.role": "Rol",
  "users.status": "Estado",
  "users.usageToday": "Uso hoy",
  "users.actions": "Acciones",
  "users.create.submit": "Crear usuario",
  "users.create.failed": "No se pudo crear el usuario",
  "users.update.failed": "No se pudo actualizar el usuario",
  "users.temporaryPassword": "Contraseña temporal para el nuevo usuario:",
  "users.status.active": "Activo",
  "users.status.disabled": "Deshabilitado",
  "users.action.enable": "Habilitar",
  "users.action.disable": "Deshabilitar",
  "users.action.resetPassword": "Restablecer contraseña",
  "users.action.viewJobs": "Ver trabajos",
  "users.jobs.title": "Trabajos",
  "users.jobs.empty": "Sin trabajos.",
  "audio.mastering.tooltip":
    "Deja el audio al volumen que piden las plataformas, medido según el estándar EBU " +
    "R128. Se mide primero y se corrige después con esa medición, que es como se hace " +
    "un master de verdad: en una sola pasada el volumen bombea.",
  "audio.section.denoise": "Reducción de ruido",
  "audio.section.mastering": "Acabado",
  "audio.section.restore": "Restauración",
  "audio.section.device": "Dispositivo",
  "audio.section.outputFormat": "Formato de salida",
  "audio.mastering.none": "Sin nivelar",
  "audio.mastering.target": "Objetivo: {{lufs}} LUFS.",
  "audio.hint.pickOne": "Elige al menos una: {{options}}.",
  "voice.sectionTitle": "Voz",
  "voice.activate": "Trabajar la voz",
  "voice.summary.none": "Sin usar",
  "voice.summary.one": "1 paso",
  "voice.summary.many": "{{count}} pasos",
  "voice.chainHint":
    "Los pasos corren de arriba hacia abajo. El orden es fijo porque cada uno " +
    "trabaja mejor sobre lo que dejó el anterior.",
  "voice.loading": "Cargando la cadena de voz…",
  "voice.loadFailed": "No se pudo cargar la cadena de voz. Probá de nuevo en un momento.",
  "voice.deliveryRequired": "Elegí un destino para que haya un volumen al que ajustar.",
  "voice.strategy.dsp": "Rápido",
  "voice.strategy.model": "Modelo IA",
  "voice.presence.label": "Cuánto realzar",
  "voice.presence.hint":
    "Poco alcanza: pasando los 4 dB la voz empieza a sonar a llamada telefónica.",
  "voice.delivery.legend": "¿A dónde va esto?",

  // --- destinos de entrega ---------------------------------------------
  "voice.delivery.streaming.label": "Streaming (Spotify, YouTube, TIDAL, Amazon)",
  "voice.delivery.streaming.description":
    "Si el audio va a Spotify, YouTube, TIDAL o Amazon. Esas plataformas bajan o " +
    "suben todo a un mismo volumen, así que entregarlo más fuerte no lo hace " +
    "sonar más fuerte: solo le saca dinámica.",
  "voice.delivery.apple_music.label": "Apple Music",
  "voice.delivery.apple_music.description":
    "Apple Music normaliza un poco más bajo que el resto. Usá este si es su " +
    "destino principal.",
  "voice.delivery.ebu_r128.label": "Broadcast / cine (EBU R128)",
  "voice.delivery.ebu_r128.description":
    "El estándar de televisión, radio y cine en Europa y buena parte del mundo. " +
    "Mucho más bajo que streaming porque se escucha en ambientes silenciosos, " +
    "donde la dinámica amplia se aprecia.",
  "voice.delivery.atsc_a85.label": "Broadcast EEUU (ATSC A/85)",
  "voice.delivery.atsc_a85.description":
    "El estándar de televisión en Estados Unidos. Casi igual al europeo, con un " +
    "techo de picos un poco más permisivo.",

  // --- motivos de checkpoint no instalable -----------------------------
  "checkpoint.noTensors": "El header del archivo no declara ningún tensor.",
  "checkpoint.isLora":
    "Es un LoRA o un adapter, no un modelo completo: se aplica sobre un modelo " +
    "base en vez de instalarse solo.",
  "checkpoint.incomplete": "No es un pipeline completo: le falta {{missing}}.",
  "checkpoint.unknownArchitecture": "No se pudo determinar la arquitectura del checkpoint.",
  "checkpoint.unsupportedArchitecture":
    "Arquitectura {{architecture}}: optimum-onnx no puede ejecutarla, así que " +
    "convertirla no serviría de nada.",
  "checkpoint.ready": "Checkpoint {{architecture}} listo para convertir.",
  "checkpoint.headerUnreadable": "no se pudo leer el header ({{detail}}).",
  "checkpoint.prefix.unknown": "No se pudo evaluar: ",
  "checkpoint.prefix.notInstallable": "No instalable: ",

  // --- roles de componente (para checkpoint.incomplete) ----------------
  "component.backbone": "backbone (unet/transformer)",
  "component.textEncoder": "text encoder",
  "component.vae": "VAE",

  // --- ajustes: idioma ---------------------------------------------------
  "settings.language.title": "Idioma",
  "settings.language.description":
    "Se aplica al instante y queda recordado en este equipo.",
  // --- editor de imagen --------------------------------------------------
  "editor.title": "Editor",
  "editor.description":
    "Elegí qué quitar o reemplazar: pintalo con el pincel, o tocá un objeto para seleccionarlo solo.",
  "editor.dropzone.label": "Elegir una imagen",
  "editor.dropzone.inputLabel": "Imagen a editar",
  "editor.dropzone.hint": "Elegí una foto para empezar a editar.",
  "editor.tools.label": "Herramientas de edición",
  "editor.tool.brush": "Pincel",
  "editor.tool.eraser": "Goma",
  "editor.tool.tap": "Tocar objeto",
  "editor.tool.tapUnavailable":
    "La selección por toque no está instalada todavía. Instalala desde Ajustes, o usá el pincel.",
  "editor.tool.pan": "Mover vista",
  "editor.brushSize": "Tamaño",
  "editor.zoom.in": "Acercar",
  "editor.zoom.out": "Alejar",
  "editor.zoom.reset": "Ajustar a la vista",
  "editor.zoomHint": "Girá la rueda para acercar. Arrastrá con la manito o el botón del medio para moverte.",
  "editor.undo": "Deshacer",
  "editor.clear": "Limpiar selección",
  "editor.segmenting": "Buscando el objeto…",
  "editor.maskHint": "El área iluminada es lo que va a cambiar. El resto queda intacto.",
  "editor.mode.label": "Qué hacer con la selección",
  "editor.mode.erase": "Quitar",
  "editor.mode.replace": "Reemplazar",
  "editor.prompt.label": "Reemplazar por…",
  "editor.prompt.placeholder": "ej. un banco de madera, un ramo de flores",
  "editor.model.label": "Modelo",
  "editor.submit.erase": "Quitar selección",
  "editor.submit.replace": "Reemplazar selección",
  "editor.changeImage": "Usar otra imagen",
  "editor.useAsBase": "Seguir editando este resultado",
  "editor.eraser.hint":
    "Este motor rellena continuando lo que hay alrededor — instantáneo y sin prompt. Para poner algo concreto en su lugar, pasá a Reemplazar.",
  "editor.advanced.title": "Vista avanzada",
  "editor.advanced.tooltip": "Parámetros del modelo: prompts, pasos, guía, semilla y fuerza.",
  "editor.advanced.summary": "{{steps}} pasos · guía {{guidance}}",
  "editor.advanced.fillPrompt": "Prompt de relleno (qué va en el área quitada)",
  "editor.advanced.negativePrompt": "Prompt negativo (qué evitar)",
  "editor.advanced.steps": "Pasos",
  "editor.advanced.guidance": "Guía",
  "editor.advanced.seed": "Semilla",
  "editor.advanced.seedPlaceholder": "aleatoria",
  "editor.advanced.strength": "Fuerza",
  "editor.advanced.strengthHint":
    "Con modelos estándar la edición corre siempre a 1.0; valores menores solo aplican con checkpoints de inpainting dedicados.",

  // --- ajustes: aceleracion por dispositivo -----------------------------
  "settings.acceleration.title": "Aceleración",
  "settings.acceleration.description":
    "Execution provider por dispositivo. DirectML es siempre el baseline; el proveedor nativo se usa solo cuando su plugin y su hardware están presentes.",
  "settings.acceleration.ready": "listo — lo usará el próximo trabajo",
  "settings.acceleration.native": "nativa",
  "settings.acceleration.fallback": "fallback (la nativa falló)",
  "settings.acceleration.preparing": "Preparando aceleración para tu GPU…",
  // --- catalogo de capacidades -----------------------------------------
  "capability.domain.video": "Video",
  "capability.domain.image": "Imágenes",
  "capability.domain.audio": "Audio",
  "capability.domain.generate": "Generar",

  "capability.video.upscale": "Reescalar video",
  "capability.video.interpolate": "Generar fotogramas",
  "capability.video.subtitles": "Generar subtítulos",
  "capability.image.upscale": "Reescalar imágenes",
  "capability.audio.denoise": "Quitar ruido",
  "capability.audio.restore": "Restaurar calidad",
  "capability.audio.voice": "Trabajar la voz",
  "capability.audio.transcribe": "Transcribir a texto",
  "capability.audio.stems": "Separar stems",
  "capability.generate.textToImage": "Texto a imagen",
  "capability.generate.imageToImage": "Imagen a imagen",
  "capability.generate.textToVideo": "Texto a video",
  "capability.generate.videoToVideo": "Video a video",
  "capability.generate.textTo3d": "Texto a 3D",
  "capability.generate.imageTo3d": "Imagen a 3D",
  "capability.generate.textToSound": "Texto a sonido",
  "capability.generate.soundToSound": "Sonido a sonido",

  // Motivos honestos: dicen QUE falta, sin prometer fecha.
  "capability.reason.subtitles":
    "Necesita un modelo de reconocimiento de voz más el paso de sincronizado que " +
    "convierte la transcripción en una pista de subtítulos. La transcripción va " +
    "primero: los subtítulos son ese resultado, alineado y muxeado al contenedor.",
  "capability.reason.stems":
    "El runtime ya está: los modelos MDX-Net corren sobre ONNX Runtime, medido en " +
    "DirectML y en CPU. Lo que falta es acertar la convención de espectrograma que " +
    "el modelo espera — un primer intento corrió pero devolvió ruido, así que " +
    "publicarlo ahora sería publicar algo roto.",
  "capability.reason.noOnnxPath":
    "Todavía no hay un camino confirmado a un ONNX ejecutable. Que el modelo " +
    "exista no alcanza: también tiene que poder correrlo ONNX Runtime, y acá se " +
    "chocó con el mismo techo ya medido para FLUX.2 y Z-Image.",

  "capability.setup.missingPack": "Falta descargar el paquete que necesita.",
  "capability.setup.missingModel": "Todavía no hay ningún modelo compatible instalado.",
  "capability.tree.loading": "Cargando capacidades…",
  "capability.tree.loadFailed": "No se pudieron cargar las capacidades. Probá de nuevo en un momento.",
  "capability.tree.roadmap": "Mapa de ruta",
  "capability.tree.download": "Descargar paquete",

  // --- selector de tareas ------------------------------------------------
  "tasks.title": "¿Qué querés hacer?",
  "tasks.subtitle": "Elegí una tarea y Upflow te lleva a la pantalla que corresponde.",
  "capability.provision.running": "Descargando el paquete… puede tardar unos minutos.",
  "capability.provision.done": "Paquete descargado. La capacidad quedó lista.",
  "capability.provision.failed": "La descarga falló: {{error}}",

  // --- pila de pasos de video -------------------------------------------
  "video.steps.title": "Pasos",
  "video.steps.description":
    "El perfil llena esta pila. Quitar o agregar un paso actualiza los ajustes del panel.",
  "video.steps.listLabel": "Pasos del trabajo de video",
  "video.steps.empty": "No hay pasos activos. Elegí un perfil o agregá uno.",
  "video.steps.add": "Agregar paso",
  "video.steps.addStep": "Agregar {{step}}",
  "video.steps.removeStep": "Quitar {{step}}",
  "video.steps.upscale.label": "Reescalar",
  "video.steps.upscale.description": "{{model}} · {{scale}}×",
  "video.steps.modelUnknown": "Modelo seleccionado",
  "video.steps.interpolate.label": "Interpolar",
  "video.steps.interpolate.multiplierDescription": "{{engine}} · {{multiplier}}× fps",
  "video.steps.interpolate.targetDescription": "{{engine}} · objetivo {{target}}",
  "video.steps.audio.label": "Audio",
  "video.steps.audio.description": "Mejorar con {{modes}}.",
  "video.steps.subtitles.label": "Subtítulos",
  "video.steps.subtitles.description": "Conservar las pistas de subtítulos incluidas.",
  // --- compatibilidad detectada de un repo -----------------------------
  "compat.gated":
    "Repo con acceso restringido: necesita un token de Hugging Face y aceptar la licencia.",
  "compat.gatedAuth":
    "Repo con acceso restringido: Hugging Face rechazó el pedido ({{detail}}). " +
    "Hace falta un token y aceptar la licencia.",
  "compat.singleFile":
    "Tiene checkpoints .safetensors sueltos en la raíz: hay que evaluar sus " +
    "headers antes de saber cuáles se pueden instalar.",
  "compat.noModelIndex": "No es un pipeline diffusers: falta {{filename}}.",
  "compat.weightsAtRoot":
    "Los pesos están sueltos en la raíz del repo, sin carpetas por componente " +
    "(unet, vae, text_encoder…). Es un checkpoint single-file, y este instalador " +
    "necesita el layout de carpetas de diffusers.",
  "compat.needsConversion":
    "Sin ONNX propio para {{components}}: requiere conversión local.",
  "compat.readyOnnx": "Trae ONNX para todos los componentes: se instala directo.",
  "compat.upscaler.readyOnnx": "Trae un .onnx: se instala directo.",
  "compat.upscaler.needsConversion":
    "Trae pesos en formato de PyTorch: se convierten a ONNX con Spandrel al instalar.",
  "compat.upscaler.noWeights":
    "No hay ningún archivo de pesos .onnx, .safetensors ni .pth en el repo.",
  "video.steps.upscaleRequired":
    "Un job de video siempre reescala, así que ese paso no se puede quitar. " +
    "Volvé a agregarlo para habilitar el botón.",

  // --- pre-flight de modelos de reescalado -----------------------------
  "upscaler.compat.readyOnnx": "ONNX listo",
  "upscaler.compat.needsConversion": "Requiere conversión",
  "upscaler.compat.gated": "Acceso restringido",
  "upscaler.compat.incompatible": "Incompatible",
  "upscaler.compat.unknown": "Compatibilidad desconocida",
  "upscaler.details.hide": "Ocultar detalles",
  "upscaler.details.show": "Ver detalles",
  "upscaler.preflight.loading": "Evaluando descarga y capacidad…",
  "upscaler.capacity.ariaLabel": "Capacidad de instalación medida",
  "upscaler.capacity.vramUnknown": "VRAM libre desconocida",
  "upscaler.capacity.vramFree": "{{free}} VRAM libre",
  "upscaler.capacity.ramLabel": "RAM",
  "upscaler.capacity.ramFree": "{{free}} libre",
  "upscaler.capacity.diskLabel": "Disco en {{path}}",
  "upscaler.capacity.diskFree": "{{free}} libres",
  "upscaler.capacity.downloadLabel": "Descarga",
  "upscaler.warnings.ariaLabel": "Avisos de instalación",
  "upscaler.warning.degraded":
    "No se pudo evaluar este modelo. Podés instalarlo igual.",
  "upscaler.warning.gated":
    "Repo con acceso restringido: necesitás un token de Hugging Face y aceptar la licencia.",
  "upscaler.warning.incompatibleFallback":
    "Este repositorio no contiene un modelo de reescalado compatible.",
  "upscaler.warning.diskLow":
    "Quedan {{free}} libres en {{path}}; hacen falta aproximadamente {{needed}} para instalarlo.",

  "generation.page.title": "Generar",
  "generation.page.description":
    "Creá una imagen desde un prompt de texto o transformá una imagen de partida, con reescalado opcional al terminar.",
  "generation.mode.label": "Modo de generación",
  "generation.mode.textToImage": "Texto a imagen",
  "generation.mode.imageToImage": "Imagen a imagen",
  "generation.initImage.drop": "Arrastrá una imagen de partida acá o hacé clic para buscarla",
  "generation.initImage.formats": "PNG, JPEG, WEBP y otros formatos de imagen legibles",
  "generation.initImage.inputLabel": "Elegir una imagen de partida",
  "generation.initImage.uploading": "Subiendo {{filename}}…",
  "generation.initImage.dimensions": "{{width}} × {{height}} px",
  "generation.initImage.replace": "Hacé clic o arrastrá otra imagen para reemplazarla",
  "generation.initImage.unknownError": "Error de subida desconocido",
  "generation.initImage.uploadFailed": "No se pudo subir la imagen de partida: {{error}}",
  "generation.initImage.required":
    "Agregá una imagen de partida para habilitar la generación de imagen a imagen.",
  "generation.strength.label": "Fuerza de transformación",
  "generation.strength.hint":
    "Los valores bajos conservan más la imagen de partida; los altos se alejan más.",
  "compat.asr.readyOnnx":
    "Trae el encoder y el decoder ya exportados: se instala directo.",
  "compat.asr.needsConversion":
    "Trae pesos en formato de PyTorch: se convierten a ONNX al instalar.",
  "compat.asr.noWeights":
    "No parece un modelo de reconocimiento de voz instalable: le falta el par " +
    "encoder/decoder exportado o los pesos de PyTorch.",

  "transcribe.page.title": "Transcribir",
  "transcribe.page.description":
    "Convertí un archivo de audio en texto editable con un modelo de reconocimiento de voz instalado.",
  "transcribe.file.drop": "Arrastrá un audio acá o hacé clic para buscarlo",
  "transcribe.file.formats": "WAV, MP3, FLAC, M4A, OGG y OPUS",
  "transcribe.file.inputLabel": "Elegir un audio para transcribir",
  "transcribe.model.label": "Modelo de reconocimiento de voz",
  "transcribe.language.label": "Idioma del audio",
  "transcribe.language.auto": "Detectar automáticamente",
  "transcribe.language.es": "Español (es)",
  "transcribe.language.en": "Inglés (en)",
  "transcribe.language.pt": "Portugués (pt)",
  "transcribe.language.fr": "Francés (fr)",
  "transcribe.language.de": "Alemán (de)",
  "transcribe.language.it": "Italiano (it)",
  "transcribe.device.label": "Dispositivo",
  "transcribe.device.loading": "Cargando dispositivos…",
  "transcribe.device.loadFailed":
    "No se pudieron cargar los dispositivos. Se usará el predeterminado del backend.",
  "transcribe.submit": "Transcribir audio",
  "transcribe.job.waiting": "Esperando un archivo de audio.",
  "transcribe.job.uploading": "Subiendo el audio…",
  "transcribe.job.queued": "Transcripción en cola…",
  "transcribe.job.running": "Transcribiendo el audio…",
  "transcribe.job.cancel": "Cancelar transcripción",
  "transcribe.job.cancelled": "La transcripción fue cancelada.",
  "transcribe.job.failedFallback": "La transcripción falló.",
  "transcribe.result.title": "Transcripción",
  "transcribe.result.waiting":
    "El texto transcripto va a aparecer acá cuando termine el job.",
  "transcribe.result.empty": "La transcripción terminada está vacía.",
  "transcribe.result.copy": "Copiar texto",
  "transcribe.result.copied": "Copiado",
  "transcribe.result.copyFailed": "No se pudo copiar el texto.",
  "transcribe.result.download": "Descargar .txt",
  "transcribe.result.downloadSubtitles": "Descargar .srt",
  "transcribe.result.downloadVideo": "Descargar video con subtítulos",
  "transcribe.translate.label": "Traducir los subtítulos a",
  "transcribe.translate.none": "Dejarlos en el idioma original",
  "transcribe.output.label": "Salida",
  "transcribe.output.text": "Solo la transcripción",
  "transcribe.output.video": "Video con pista de subtítulos",
  "transcribe.output.videoBurned": "Video con los subtítulos quemados",
  "transcribe.output.dubbed": "Video doblado a otro idioma",
  "transcribe.dub.language": "Doblar a",
  "transcribe.dub.hint": "Cada línea se traduce y se dice en su propio hueco. Necesita el par de idiomas instalado.",
  "transcribe.dub.overflow": "{{count}} líneas se salieron de su hueco: se dijeron tan rápido como todavía se entienden.",
  "transcribe.output.hint": "Quemar re-encodea el video entero: más lento, pero se ve en cualquier lado.",
  "transcribe.models.loadingInstalled":
    "Cargando modelos de reconocimiento de voz instalados…",
  "transcribe.models.loadInstalledFailed":
    "No se pudieron cargar los modelos de reconocimiento de voz instalados.",
  "transcribe.models.noneTitle": "No hay ningún modelo de transcripción instalado",
  "transcribe.models.noneDescription":
    "Buscá abajo en Hugging Face e instalá un modelo de reconocimiento de voz antes de transcribir tu primer audio.",
  "transcribe.models.title": "Modelos de reconocimiento de voz",
  "transcribe.models.description":
    "Buscá un modelo ASR en Hugging Face e instalalo localmente.",
  "transcribe.models.searchLabel":
    "Buscar modelos de reconocimiento de voz en Hugging Face",
  "transcribe.models.searchPlaceholder":
    "Buscar modelos de reconocimiento de voz…",
  "transcribe.models.searching": "Buscando en Hugging Face…",
  "transcribe.models.searchFailed":
    "No se pudieron buscar modelos de reconocimiento de voz en Hugging Face.",
  "transcribe.models.noResults":
    "No se encontraron modelos de reconocimiento de voz.",
  "transcribe.models.author": "Por {{author}}",
  "transcribe.models.downloads": "Descargas",
  "transcribe.models.likes": "Me gusta",
  "transcribe.models.compat.readyOnnx": "ONNX listo",
  "transcribe.models.compat.needsConversion": "Requiere conversión",
  "transcribe.models.compat.gated": "Acceso restringido",
  "transcribe.models.compat.incompatible": "Incompatible",
  "transcribe.models.compat.unknown": "Compatibilidad desconocida",
  "transcribe.models.install": "Instalar",
  "transcribe.models.installed": "Instalado",
  "transcribe.models.install.starting": "Iniciando instalación…",
  "transcribe.models.install.queued": "Instalación en cola…",
  "transcribe.models.install.downloading": "Descargando modelo…",
  "transcribe.models.install.working": "Instalando modelo…",
  "transcribe.models.install.failedFallback":
    "La instalación del modelo falló.",
  "transcribe.models.retry": "Reintentar",
  "video.output.mode.label": "Modo de salida",
  "video.output.mode.resolution": "Resolución objetivo",
  "video.output.mode.resolution.description":
    "Elegí el alto de salida; el ancho conserva la relación de aspecto de la fuente.",
  "video.output.mode.multiplier": "Multiplicador",
  "video.output.mode.multiplier.description":
    "Usá la escala del perfil seleccionado.",
  "video.output.target.label": "Alto objetivo",
  "video.output.target.sourceAlreadyReaches":
    "La fuente ya alcanza o supera este valor: elegí un modelo sin IA para reducir",
  "video.output.preview.label": "Salida esperada",
  "video.output.preview.dimensions": "{{width}} × {{height}} px",
  "video.output.preview.targetOnly":
    "{{height}}p de alto · el ancho conserva la relación de aspecto",
  "video.output.preview.scaleOnly":
    "{{scale}}× las dimensiones de la fuente · tamaño exacto no disponible",
  "video.output.preview.unavailable":
    "Agregá un paso de reescalado para calcular la salida.",
  "video.output.warnings.label": "Avisos de tamaño de salida",
  "video.output.warning.target":
    "Esto produciría {{width}} × {{height}} ({{megapixels}} MP) por cuadro. Un objetivo de {{suggestedHeight}}p es una alternativa más proporcionada.",
  "video.output.warning.keepSource":
    "Esto produciría {{width}} × {{height}} ({{megapixels}} MP) por cuadro. La fuente ya tiene {{sourceHeight}}p; conservar su resolución actual es una alternativa más proporcionada.",
} as const;
