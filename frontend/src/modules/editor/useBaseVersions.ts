import { useCallback, useEffect, useRef, useState } from "react";
import type { InitImageResponse } from "../../lib/apiTypes";
import { uploadGenerationInitImage } from "../../services/generation";

export const MAX_BASE_VERSIONS = 10;

export interface BaseVersion {
  blob: Blob;
  info: InitImageResponse;
}

// La versión 0 es siempre la foto abierta en la sesión: el tope descarta la
// más vieja que no sea esa, así el original sobrevive a cualquier cantidad de retoques.
function appendVersion(versions: BaseVersion[], next: BaseVersion): BaseVersion[] {
  const grown = [...versions, next];
  if (grown.length > MAX_BASE_VERSIONS) grown.splice(1, 1);
  return grown;
}

export function useBaseVersions() {
  const [versions, setVersions] = useState<BaseVersion[]>([]);
  const versionsRef = useRef<BaseVersion[]>([]);
  // Cada foto nueva (o cerrar la imagen) abre otra sesión: una subida que termina
  // después de eso ya no pertenece a la imagen que se ve, y se descarta.
  const sessionRef = useRef(0);

  const commit = useCallback((next: BaseVersion[]) => {
    versionsRef.current = next;
    setVersions(next);
  }, []);

  const reset = useCallback(async (file: File) => {
    const session = ++sessionRef.current;
    const info = await uploadGenerationInitImage(file);
    if (session !== sessionRef.current) return;
    commit([{ blob: file, info }]);
  }, [commit]);

  // Devuelve true si el tope descartó una versión vieja.
  const push = useCallback(async (blob: Blob): Promise<boolean> => {
    const session = sessionRef.current;
    const info = await uploadGenerationInitImage(new File([blob], "edited.png", { type: "image/png" }));
    if (session !== sessionRef.current) return false;
    const evicted = versionsRef.current.length >= MAX_BASE_VERSIONS;
    commit(appendVersion(versionsRef.current, { blob, info }));
    return evicted;
  }, [commit]);

  const undo = useCallback((): boolean => {
    const current = versionsRef.current;
    if (current.length <= 1) return false;
    commit(current.slice(0, -1));
    return true;
  }, [commit]);

  const clear = useCallback(() => {
    sessionRef.current += 1;
    commit([]);
  }, [commit]);

  const original = versions[0] ?? null;
  const latest = versions[versions.length - 1] ?? null;
  return {
    original,
    current: latest?.info ?? null,
    currentBlob: latest?.blob ?? null,
    push,
    undo,
    reset,
    clear,
  };
}

export function useObjectUrl(blob: Blob | null): string | null {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!blob || typeof URL.createObjectURL !== "function") {
      setUrl(null);
      return;
    }
    const created = URL.createObjectURL(blob);
    setUrl(created);
    return () => URL.revokeObjectURL(created);
  }, [blob]);
  return url;
}
