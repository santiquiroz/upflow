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
  // --- avisos de instalacion de modelos ---------------------------------
  "warning.degraded": "Could not evaluate this model. You can install it anyway.",
  "warning.gated":
    "Restricted repo: you need a Hugging Face token and to accept the license.",
  "warning.incompatible.fallback": "This does not look like a diffusers pipeline.",
  "warning.diskLow":
    "{{free}} free on {{path}} and {{needed}} required.",
  "warning.diskLow.singleFile":
    "Converting this checkpoint needs about {{peak}} peak on {{path}} " +
    "(the checkpoint, the pipeline and the ONNX all sit on disk at once) " +
    "and {{free}} are free.",
  "warning.ramLow":
    "Conversion loads the whole checkpoint into RAM: {{needed}} required and {{free}} free.",
  "warning.deviceWontFit":
    "{{device}}: does not fit. Needs about {{needed}} estimated at " +
    "{{width}}x{{height}} and has {{free}} free.",
  "warning.cpuOnly": "No compatible GPU: generating on CPU takes several minutes per image.",
  "warning.cpuSlow": "On CPU this takes several minutes per image.",

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
} as const;
