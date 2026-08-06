import type { CapabilityResponse } from "../../lib/apiTypes";

// A que superficie lleva cada capacidad. Los modulos NO se reescriben: esto es
// solo la puerta de entrada, y el estado preseleccionado viaja en la URL para que
// la pantalla abra en la pestaña correcta.
const SURFACE_BY_CAPABILITY: Record<string, string> = {
  "video.upscale": "/enhance/video",
  "video.interpolate": "/enhance/video",
  "image.upscale": "/enhance/image",
  "audio.denoise": "/audio",
  "audio.restore": "/audio",
  "audio.voice": "/audio",
  "audio.transcribe": "/transcribe",
  // Sincronicas: devuelven el audio en la misma peticion, sin encolar. Igual
  // tienen pantalla, que es lo que decide si van a algun lado.
  "audio.speak": "/voice",
  "audio.voiceConvert": "/voice",
  "generate.textToImage": "/generate",
  "generate.textToVideo": "/generate",
};

export function surfaceFor(capability: CapabilityResponse): string | null {
  // El mapa ES la decision. Antes habia ademas un porton por `jobKind === null`,
  // como sustituto de "no tiene a donde mandar trabajo"; resulto falso para las
  // capacidades sincronicas —la voz devuelve el audio en la misma peticion— que
  // quedaban listadas en Tareas sin llevar a ninguna pantalla.
  return SURFACE_BY_CAPABILITY[capability.id] ?? null;
}
