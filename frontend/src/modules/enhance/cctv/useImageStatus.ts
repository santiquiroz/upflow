import { useState } from "react";

export type ImageStatus = "loading" | "ready" | "failed";

export interface ImageStatusHandlers {
  status: ImageStatus;
  onLoad: () => void;
  onError: () => void;
}

interface SettledImage {
  src: string;
  ok: boolean;
}

function statusFor(settled: SettledImage | null, src: string): ImageStatus {
  if (settled === null || settled.src !== src) {
    return "loading";
  }
  return settled.ok ? "ready" : "failed";
}

// El estado queda atado al src que lo produjo: cambiar de cuadro vuelve a "loading" sin un efecto.
export function useImageStatus(src: string): ImageStatusHandlers {
  const [settled, setSettled] = useState<SettledImage | null>(null);
  return {
    status: statusFor(settled, src),
    onLoad: () => setSettled({ src, ok: true }),
    onError: () => setSettled({ src, ok: false }),
  };
}
