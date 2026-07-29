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
  "generation.warning.incompatible.reason": "{{reason}}",
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
} as const;
