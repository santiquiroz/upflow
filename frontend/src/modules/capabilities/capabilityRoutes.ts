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
  "generate.textToImage": "/generate",
  "generate.textToVideo": "/generate",
};

export function surfaceFor(capability: CapabilityResponse): string | null {
  // Una capacidad sin job_kind no tiene a donde mandar trabajo, asi que no puede
  // tener superficie: chequear eso ademas del mapa evita que agregar una entrada
  // al mapa por error abra una pantalla que no hace nada.
  if (capability.jobKind === null) {
    return null;
  }
  return SURFACE_BY_CAPABILITY[capability.id] ?? null;
}
