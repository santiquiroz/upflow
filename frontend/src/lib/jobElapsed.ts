import { formatDuration } from "./formatDuration";

// Cuanto lleva corriendo, no cuanto tardo: `formatDuration` ya sabe formatear un
// intervalo, asi que el "ahora" se pasa como el segundo extremo en vez de
// escribir un segundo formateador que se desincronice del primero.
export function formatElapsed(startedAt: string | null, nowMs: number): string | null {
  if (!startedAt) {
    return null;
  }
  const started = new Date(startedAt).getTime();
  if (!Number.isFinite(started) || nowMs < started) {
    return null;
  }
  return formatDuration(startedAt, new Date(nowMs).toISOString());
}
