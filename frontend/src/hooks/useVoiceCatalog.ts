import { useQuery } from "@tanstack/react-query";
import type { VoiceCatalog } from "../lib/apiTypes";
import { fetchVoiceCatalog } from "../services/audio";

// El catalogo es estatico dentro de una version del backend: los pasos, su orden
// causal y los numeros de loudness no cambian en runtime. Vale la pena no volver
// a pedirlo en cada visita al panel.
const ONE_HOUR_MS = 60 * 60 * 1000;

export function useVoiceCatalog() {
  return useQuery<VoiceCatalog>({
    queryKey: ["voice-catalog"],
    queryFn: fetchVoiceCatalog,
    staleTime: ONE_HOUR_MS,
  });
}
