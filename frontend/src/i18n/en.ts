// Catalogo ingles. REGLA: reproduce la copia original de la app VERBATIM. La
// copia de la app es inglesa (medido: 1550 marcadores en ingles contra 53 en
// espanol), el locale de test esta fijado aca, y por eso este catalogo es la red
// que detecta una extraccion mal hecha: una clave mal escrita rinde la clave
// cruda y el assert de texto literal falla.
//
// Las claves de generacion son la excepcion historica: ese modulo se escribio en
// espanol, asi que su entrada inglesa es traduccion y sus tests se pasaron a
// ingles al extraerlas.
export const en = {
  // --- generation module -------------------------------------------------
  "generation.compat.readyOnnx": "ONNX ready",
  "generation.compat.needsConversion": "Requires conversion",
  "generation.compat.singleFile": "Single-file",
  "generation.compat.gated": "Restricted access",
  "generation.compat.incompatible": "Incompatible",
  "generation.compat.unknown": "Compatibility unknown",
  "generation.precision.title": "Precision",
  "generation.precision.download": "{{size}} download",
  "generation.checkpoint.title": "Checkpoint",
  "generation.details.hide": "Hide details",
  "generation.details.show": "View details",
  "generation.preflight.loading": "Evaluating download and capacity…",
  "generation.capacity.vramUnknown": "Available VRAM unknown",
  "generation.capacity.vramFree": "{{free}} VRAM available",
  "generation.capacity.ramFree": "{{free}} available",
  "generation.warnings.ariaLabel": "Installation warnings",
  "generation.warning.degraded":
    "Could not evaluate this model. You can install it anyway.",
  "generation.warning.gated":
    "Restricted repo: you need a Hugging Face token and to accept the license.",
  "generation.warning.incompatible.fallback":
    "This does not look like a diffusers pipeline.",
  "generation.warning.diskLow":
    "Only {{free}} free on {{path}}; {{needed}} is required.",
  "generation.warning.diskLow.singleFile":
    "Converting this checkpoint requires about {{peak}} of peak disk space on {{path}} " +
    "(the checkpoint, the pipeline, and the ONNX are all stored on disk at once), " +
    "with {{free}} available.",
  "generation.warning.ramLow":
    "Conversion loads the whole checkpoint into RAM: {{needed}} required and {{free}} available.",
  "generation.warning.deviceWontFit":
    "{{device}}: does not fit. It needs an estimated {{needed}} at " +
    "{{width}}x{{height}} and has {{free}} available.",
  "generation.warning.cpuOnly":
    "No compatible GPU: generating on CPU takes several minutes per image.",
  "generation.warning.cpuSlow":
    "On CPU this takes several minutes per image.",

  // --- cadena de mejora de voz -----------------------------------------
  "voice.step.denoise.label": "Remove background noise",
  "voice.step.denoise.description":
    "Removes hum, air and constant noise. It goes first because everything " +
    "after it works better on a clean recording.",
  "voice.step.highpass.label": "Clean up the low end",
  "voice.step.highpass.description":
    "Cuts what sits below the human voice: desk bumps, footsteps, wind. It also " +
    "keeps the volume control from fighting those low frequencies.",
  "voice.step.compress.label": "Even out the volume",
  "voice.step.compress.description":
    "Brings the loudest and quietest parts closer together, so nobody has to " +
    "ride the volume knob while listening.",
  "voice.step.presence.label": "Focus the voice",
  "voice.step.presence.description":
    "Lifts the band where speech intelligibility lives. This is what makes " +
    "dialogue cut through music and effects.",
  "voice.step.deesser.label": "Soften the S sounds",
  "voice.step.deesser.description":
    "Tames harsh sibilance that gets fatiguing on headphones. It runs near the " +
    "end, on the sibilance that actually survived.",
  "voice.step.loudness.label": "Match the delivery target",
  "voice.step.loudness.description":
    "Brings overall loudness to what the destination platform asks for. Always " +
    "last, because it measures the finished result: measuring earlier would " +
    "give a number that is no longer true.",

  // --- panel de la cadena de voz ---------------------------------------
  "voice.sectionTitle": "Voice",
  "voice.activate": "Shape the voice",
  "voice.summary.none": "Off",
  "voice.summary.one": "1 step",
  "voice.summary.many": "{{count}} steps",
  "voice.chainHint":
    "The steps run top to bottom. The order is fixed because each one works " +
    "better on what the previous one left behind.",
  "voice.loading": "Loading the voice chain…",
  "voice.loadFailed": "Could not load the voice chain. Try again in a moment.",
  "voice.deliveryRequired": "Pick a delivery target so there is a loudness to match.",
  "voice.strategy.dsp": "Fast",
  "voice.strategy.model": "AI model",
  "voice.presence.label": "How much to lift",
  "voice.presence.hint":
    "A little goes a long way: past about 4 dB the voice starts sounding like a phone call.",
  "voice.delivery.legend": "Where is this going?",

  // --- destinos de entrega ---------------------------------------------
  "voice.delivery.streaming.label": "Streaming (Spotify, YouTube, TIDAL, Amazon)",
  "voice.delivery.streaming.description":
    "For audio headed to Spotify, YouTube, TIDAL or Amazon. Those platforms " +
    "normalise everything to the same loudness, so delivering louder does not " +
    "sound louder: it only costs you dynamics.",
  "voice.delivery.apple_music.label": "Apple Music",
  "voice.delivery.apple_music.description":
    "Apple Music normalises slightly lower than the rest. Use this if it is the " +
    "primary destination.",
  "voice.delivery.ebu_r128.label": "Broadcast / film (EBU R128)",
  "voice.delivery.ebu_r128.description":
    "The television, radio and film standard in Europe and much of the world. " +
    "Much lower than streaming because it is heard in quiet rooms, where wide " +
    "dynamics are an asset.",
  "voice.delivery.atsc_a85.label": "US broadcast (ATSC A/85)",
  "voice.delivery.atsc_a85.description":
    "The television standard in the United States. Almost the same as the " +
    "European one, with a slightly more permissive peak ceiling.",

  // --- motivos de checkpoint no instalable -----------------------------
  "checkpoint.noTensors": "The file header declares no tensors.",
  "checkpoint.isLora":
    "This is a LoRA or an adapter, not a complete model: it is applied on top " +
    "of a base model instead of being installed on its own.",
  "checkpoint.incomplete": "Not a complete pipeline: it is missing {{missing}}.",
  "checkpoint.unknownArchitecture": "Could not determine the checkpoint architecture.",
  "checkpoint.unsupportedArchitecture":
    "Architecture {{architecture}}: optimum-onnx cannot run it, so converting " +
    "it would not help.",
  "checkpoint.ready": "{{architecture}} checkpoint ready to convert.",
  "checkpoint.headerUnreadable": "could not read its header ({{detail}}).",
  "checkpoint.prefix.unknown": "Could not evaluate: ",
  "checkpoint.prefix.notInstallable": "Not installable: ",

  // --- roles de componente (para checkpoint.incomplete) ----------------
  "component.backbone": "backbone (unet/transformer)",
  "component.textEncoder": "text encoder",
  "component.vae": "VAE",

  // --- ajustes: idioma ---------------------------------------------------
  "settings.language.title": "Language",
  "settings.language.description":
    "Applies right away and is remembered on this device.",
  // --- catalogo de capacidades -----------------------------------------
  "capability.domain.video": "Video",
  "capability.domain.image": "Images",
  "capability.domain.audio": "Audio",
  "capability.domain.generate": "Generate",

  "capability.video.upscale": "Upscale video",
  "capability.video.interpolate": "Generate frames",
  "capability.video.subtitles": "Generate subtitles",
  "capability.image.upscale": "Upscale images",
  "capability.audio.denoise": "Remove noise",
  "capability.audio.restore": "Restore quality",
  "capability.audio.voice": "Shape the voice",
  "capability.audio.stems": "Separate stems",
  "capability.generate.textToImage": "Text to image",
  "capability.generate.imageToImage": "Image to image",
  "capability.generate.textToVideo": "Text to video",
  "capability.generate.videoToVideo": "Video to video",
  "capability.generate.textTo3d": "Text to 3D",
  "capability.generate.imageTo3d": "Image to 3D",
  "capability.generate.textToSound": "Text to sound",
  "capability.generate.soundToSound": "Sound to sound",

  // Motivos honestos: dicen QUE falta, sin prometer fecha.
  "capability.reason.subtitles":
    "Needs a speech recognition engine, which the app does not ship yet. " +
    "whisper.cpp is the intended path.",
  "capability.reason.stems":
    "Needs a Demucs-type model and an inference engine the app does not include today.",
  "capability.reason.imageToImage":
    "The path is confirmed and measured: ONNX Runtime can run image to image for " +
    "Stable Diffusion, SDXL, SD3 and Latent Consistency — but not for Flux or " +
    "Sana, which it does run for text to image. Wiring work, not a missing " +
    "dependency.",
  "capability.reason.noOnnxPath":
    "No confirmed path to a runnable ONNX model yet. The model existing is not " +
    "enough: it also has to be executable by ONNX Runtime, and this hit the same " +
    "ceiling already measured for FLUX.2 and Z-Image.",

  "capability.setup.missingPack": "The package it needs is not downloaded yet.",
  "capability.setup.missingModel": "No compatible model is installed yet.",
  "capability.tree.loading": "Loading capabilities…",
  "capability.tree.loadFailed": "Could not load capabilities. Try again in a moment.",
  "capability.tree.roadmap": "Roadmap",
  "capability.tree.download": "Download package",

  // --- selector de tareas ------------------------------------------------
  "tasks.title": "What do you want to do?",
  "tasks.subtitle": "Pick a task and Upflow takes you to the right screen.",
  "capability.provision.running": "Downloading the package… this can take a few minutes.",
  "capability.provision.done": "Package downloaded. The capability is ready.",
  "capability.provision.failed": "The download failed: {{error}}",

  // --- pila de pasos de video -------------------------------------------
  "video.steps.title": "Steps",
  "video.steps.description":
    "The profile fills this stack. Removing or adding a step updates the panel settings.",
  "video.steps.listLabel": "Video job steps",
  "video.steps.empty": "No steps are active. Choose a profile or add one.",
  "video.steps.add": "Add step",
  "video.steps.addStep": "Add {{step}}",
  "video.steps.removeStep": "Remove {{step}}",
  "video.steps.upscale.label": "Upscale",
  "video.steps.upscale.description": "{{model}} · {{scale}}×",
  "video.steps.modelUnknown": "Selected model",
  "video.steps.interpolate.label": "Interpolate",
  "video.steps.interpolate.multiplierDescription": "{{engine}} · {{multiplier}}× fps",
  "video.steps.interpolate.targetDescription": "{{engine}} · target {{target}}",
  "video.steps.audio.label": "Audio",
  "video.steps.audio.description": "Enhance with {{modes}}.",
  "video.steps.subtitles.label": "Subtitles",
  "video.steps.subtitles.description": "Keep the embedded subtitle tracks.",
  // --- compatibilidad detectada de un repo -----------------------------
  "compat.gated":
    "Restricted repo: it needs a Hugging Face token and the license accepted.",
  "compat.gatedAuth":
    "Restricted repo: Hugging Face refused the request ({{detail}}). You need a " +
    "token and the license accepted.",
  "compat.singleFile":
    "It has loose .safetensors checkpoints at the root: their headers have to be " +
    "read before knowing which ones can be installed.",
  "compat.noModelIndex": "Not a diffusers pipeline: {{filename}} is missing.",
  "compat.weightsAtRoot":
    "The weights sit loose at the root of the repo, with no per-component folders " +
    "(unet, vae, text_encoder…). That is a single-file checkpoint, and this " +
    "installer needs the diffusers folder layout.",
  "compat.needsConversion":
    "No ONNX of its own for {{components}}: it needs a local conversion.",
  "compat.readyOnnx": "It ships ONNX for every component: installs directly.",
  "compat.upscaler.readyOnnx": "It ships an .onnx: installs directly.",
  "compat.upscaler.needsConversion":
    "It ships PyTorch weights: they get converted to ONNX with Spandrel on install.",
  "compat.upscaler.noWeights":
    "There is no .onnx, .safetensors or .pth weight file in the repo.",
  "video.steps.upscaleRequired":
    "A video job always upscales, so the upscale step cannot be removed. Add it " +
    "back to enable the button.",

  // --- upscaler model pre-flight ---------------------------------------
  "upscaler.compat.readyOnnx": "ONNX ready",
  "upscaler.compat.needsConversion": "Requires conversion",
  "upscaler.compat.gated": "Restricted access",
  "upscaler.compat.incompatible": "Incompatible",
  "upscaler.compat.unknown": "Compatibility unknown",
  "upscaler.details.hide": "Hide details",
  "upscaler.details.show": "View details",
  "upscaler.preflight.loading": "Evaluating download and capacity…",
  "upscaler.capacity.ariaLabel": "Measured installation capacity",
  "upscaler.capacity.vramUnknown": "Available VRAM unknown",
  "upscaler.capacity.vramFree": "{{free}} VRAM available",
  "upscaler.capacity.ramLabel": "RAM",
  "upscaler.capacity.ramFree": "{{free}} available",
  "upscaler.capacity.diskLabel": "Disk at {{path}}",
  "upscaler.capacity.diskFree": "{{free}} available",
  "upscaler.capacity.downloadLabel": "Download",
  "upscaler.warnings.ariaLabel": "Installation warnings",
  "upscaler.warning.degraded":
    "Could not evaluate this model. You can install it anyway.",
  "upscaler.warning.gated":
    "Restricted repo: you need a Hugging Face token and to accept the license.",
  "upscaler.warning.incompatibleFallback":
    "This repository does not contain a supported upscaler model.",
  "upscaler.warning.diskLow":
    "Only {{free}} free on {{path}}; about {{needed}} is required to install it.",
} as const;
