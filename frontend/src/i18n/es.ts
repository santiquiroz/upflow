// Catalogo espanol. Los textos que ya existian en espanol (los que construimos
// entre v0.15.0 y v0.16.2) se conservan tal cual estaban: eran correctos, solo
// estaban en el lugar equivocado.
export const es = {
  // --- modulo de generacion ----------------------------------------------
  "generation.compat.readyOnnx": "ONNX listo",
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
  "editor.brushSize": "Tamaño",
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
