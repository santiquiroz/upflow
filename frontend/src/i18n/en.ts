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
  "generation.checkpoint.required":
    "Pick which checkpoint file to install first",
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
  // --- editor de imagen --------------------------------------------------
  "editor.title": "Editor",
  "editor.description":
    "Select what to remove or replace: paint over it, or tap an object to select it automatically.",
  "editor.dropzone.label": "Choose an image",
  "editor.dropzone.inputLabel": "Image to edit",
  "editor.dropzone.hint": "Pick a photo to start editing.",
  "editor.tools.label": "Editing tools",
  "editor.tool.brush": "Brush",
  "editor.tool.eraser": "Eraser",
  "editor.tool.tap": "Tap object",
  "editor.tool.tapUnavailable":
    "Tap-to-select is not installed yet. Install it from Settings, or use the brush.",
  "editor.brushSize": "Size",
  "editor.undo": "Undo",
  "editor.clear": "Clear selection",
  "editor.segmenting": "Finding the object…",
  "editor.maskHint": "The highlighted area is what will change. Everything else stays untouched.",
  "editor.mode.label": "What to do with the selection",
  "editor.mode.erase": "Remove",
  "editor.mode.replace": "Replace",
  "editor.prompt.label": "Replace with…",
  "editor.prompt.placeholder": "e.g. a wooden bench, a bouquet of flowers",
  "editor.model.label": "Model",
  "editor.submit.erase": "Remove selection",
  "editor.submit.replace": "Replace selection",
  "editor.changeImage": "Use another image",
  "editor.useAsBase": "Keep editing this result",
  "editor.advanced.title": "Advanced view",
  "editor.advanced.tooltip": "Model parameters: prompts, steps, guidance, seed and strength.",
  "editor.advanced.summary": "{{steps}} steps · guidance {{guidance}}",
  "editor.advanced.fillPrompt": "Fill prompt (what replaces the removed area)",
  "editor.advanced.negativePrompt": "Negative prompt (what to avoid)",
  "editor.advanced.steps": "Steps",
  "editor.advanced.guidance": "Guidance",
  "editor.advanced.seed": "Seed",
  "editor.advanced.seedPlaceholder": "random",
  "editor.advanced.strength": "Strength",
  "editor.advanced.strengthHint":
    "With standard models the edit always runs at 1.0; lower values only take effect with dedicated inpainting checkpoints.",

  // --- ajustes: aceleracion por dispositivo -----------------------------
  "settings.acceleration.title": "Acceleration",
  "settings.acceleration.description":
    "Execution provider per device. DirectML is always the baseline; a native provider is used automatically when its plugin and hardware are present.",
  "settings.acceleration.native": "native",
  "settings.acceleration.fallback": "fallback (native failed)",
  "settings.acceleration.preparing": "Preparing acceleration for your GPU…",
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
  "capability.audio.transcribe": "Transcribe to text",
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
    "Needs a speech recognition model plus the timing pass that turns the " +
    "transcript into a subtitle track. Transcription comes first: subtitles are " +
    "that result, aligned and muxed into the container.",
  "capability.reason.stems":
    "The runtime is already here: MDX-Net models run on ONNX Runtime, measured on " +
    "both DirectML and CPU. What is still missing is matching the spectrogram " +
    "convention the model expects — a first attempt ran but produced noise, so " +
    "shipping it now would mean shipping a broken feature.",
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

  "generation.page.title": "Generate",
  "generation.page.description":
    "Create an image from a text prompt or transform a starting image, with optional upscaling on completion.",
  "generation.mode.label": "Generation mode",
  "generation.mode.textToImage": "Text to image",
  "generation.mode.imageToImage": "Image to image",
  "generation.initImage.drop": "Drop a starting image here or click to browse",
  "generation.initImage.formats": "PNG, JPEG, WEBP, and other readable image formats",
  "generation.initImage.inputLabel": "Choose a starting image",
  "generation.initImage.uploading": "Uploading {{filename}}…",
  "generation.initImage.dimensions": "{{width}} × {{height}} px",
  "generation.initImage.replace": "Click or drop another image to replace it",
  "generation.initImage.unknownError": "Unknown upload error",
  "generation.initImage.uploadFailed": "Could not upload the starting image: {{error}}",
  "generation.initImage.required":
    "Add a starting image to enable image-to-image generation.",
  "generation.strength.label": "Transformation strength",
  "generation.strength.hint":
    "Lower values stay closer to the starting image; higher values move further away.",
  "compat.asr.readyOnnx":
    "It ships the exported encoder and decoder: installs directly.",
  "compat.asr.needsConversion":
    "It ships PyTorch weights: they get converted to ONNX on install.",
  "compat.asr.noWeights":
    "This does not look like an installable speech recognition model: it is " +
    "missing either the exported encoder/decoder pair or the PyTorch weights.",

  "transcribe.page.title": "Transcribe",
  "transcribe.page.description":
    "Turn an audio file into editable text with an installed speech recognition model.",
  "transcribe.file.drop": "Drop an audio file here or click to browse",
  "transcribe.file.formats": "WAV, MP3, FLAC, M4A, OGG, and OPUS",
  "transcribe.file.inputLabel": "Choose an audio file to transcribe",
  "transcribe.model.label": "Speech recognition model",
  "transcribe.language.label": "Audio language",
  "transcribe.language.auto": "Detect automatically",
  "transcribe.language.es": "Spanish (es)",
  "transcribe.language.en": "English (en)",
  "transcribe.language.pt": "Portuguese (pt)",
  "transcribe.language.fr": "French (fr)",
  "transcribe.language.de": "German (de)",
  "transcribe.language.it": "Italian (it)",
  "transcribe.device.label": "Device",
  "transcribe.device.loading": "Loading devices…",
  "transcribe.device.loadFailed":
    "Could not load devices. The backend default will be used.",
  "transcribe.submit": "Transcribe audio",
  "transcribe.job.waiting": "Waiting for an audio file.",
  "transcribe.job.uploading": "Uploading audio…",
  "transcribe.job.queued": "Transcription queued…",
  "transcribe.job.running": "Transcribing audio…",
  "transcribe.job.cancel": "Cancel transcription",
  "transcribe.job.cancelled": "The transcription was cancelled.",
  "transcribe.job.failedFallback": "The transcription failed.",
  "transcribe.result.title": "Transcription",
  "transcribe.result.waiting":
    "The transcribed text will appear here when the job completes.",
  "transcribe.result.empty": "The completed transcription is empty.",
  "transcribe.result.copy": "Copy text",
  "transcribe.result.copied": "Copied",
  "transcribe.result.copyFailed": "Could not copy the text.",
  "transcribe.result.download": "Download .txt",
  "transcribe.models.loadingInstalled":
    "Loading installed speech recognition models…",
  "transcribe.models.loadInstalledFailed":
    "Could not load installed speech recognition models.",
  "transcribe.models.noneTitle": "No transcription model is installed",
  "transcribe.models.noneDescription":
    "Search Hugging Face below and install a speech recognition model before transcribing your first audio file.",
  "transcribe.models.title": "Speech recognition models",
  "transcribe.models.description":
    "Search Hugging Face for an ASR model and install it locally.",
  "transcribe.models.searchLabel":
    "Search Hugging Face for speech recognition models",
  "transcribe.models.searchPlaceholder":
    "Search speech recognition models…",
  "transcribe.models.searching": "Searching Hugging Face…",
  "transcribe.models.searchFailed":
    "Could not search Hugging Face for speech recognition models.",
  "transcribe.models.noResults": "No speech recognition models found.",
  "transcribe.models.author": "By {{author}}",
  "transcribe.models.downloads": "Downloads",
  "transcribe.models.likes": "Likes",
  "transcribe.models.compat.readyOnnx": "ONNX ready",
  "transcribe.models.compat.needsConversion": "Requires conversion",
  "transcribe.models.compat.gated": "Restricted access",
  "transcribe.models.compat.incompatible": "Incompatible",
  "transcribe.models.compat.unknown": "Compatibility unknown",
  "transcribe.models.install": "Install",
  "transcribe.models.installed": "Installed",
  "transcribe.models.install.starting": "Starting installation…",
  "transcribe.models.install.queued": "Installation queued…",
  "transcribe.models.install.downloading": "Downloading model…",
  "transcribe.models.install.working": "Installing model…",
  "transcribe.models.install.failedFallback":
    "The model installation failed.",
  "transcribe.models.retry": "Try again",
  "video.output.mode.label": "Output mode",
  "video.output.mode.resolution": "Target resolution",
  "video.output.mode.resolution.description":
    "Choose the output height; width follows the source aspect ratio.",
  "video.output.mode.multiplier": "Multiplier",
  "video.output.mode.multiplier.description":
    "Use the scale from the selected profile.",
  "video.output.target.label": "Target height",
  // Dice QUE hacer, no solo que algo esta mal: los modelos de IA solo amplian, asi que
  // bajar de resolucion necesita el grupo "No AI" del selector de modelo.
  "video.output.target.sourceAlreadyReaches":
    "Source already meets or exceeds this — pick a No AI model to downscale",
  "video.output.preview.label": "Expected output",
  "video.output.preview.dimensions": "{{width}} × {{height}} px",
  "video.output.preview.targetOnly":
    "{{height}}p high · width follows the source aspect ratio",
  "video.output.preview.scaleOnly":
    "{{scale}}× source dimensions · exact size unavailable",
  "video.output.preview.unavailable":
    "Add an upscale step to calculate the output.",
  "video.output.warnings.label": "Output size warnings",
  "video.output.warning.target":
    "This would produce {{width}} × {{height}} ({{megapixels}} MP) per frame. A {{suggestedHeight}}p target is a more proportionate alternative.",
  "video.output.warning.keepSource":
    "This would produce {{width}} × {{height}} ({{megapixels}} MP) per frame. The source is already {{sourceHeight}}p; keeping its current resolution is a more proportionate alternative.",
} as const;
