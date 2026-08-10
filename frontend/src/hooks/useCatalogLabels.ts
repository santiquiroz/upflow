import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "../i18n/LocaleProvider";
import type { AudioCapabilities, VoiceCatalog } from "../lib/apiTypes";
import type { CatalogName } from "../lib/jobDetails";
import { fetchAudioCapabilities, fetchVoiceCatalog } from "../services/audio";

// Los catalogos son estaticos dentro de una version del backend: sus ids, su
// orden y su copia no cambian en runtime.
const ONE_HOUR_MS = 60 * 60 * 1000;

export type CatalogLabelResolver = (catalog: CatalogName, id: string) => string;

function pickLabel(
  catalog: CatalogName,
  id: string,
  audio: AudioCapabilities | undefined,
  voice: VoiceCatalog | undefined,
  t: (key: string) => string,
): string | null {
  if (catalog === "cleanupStep") {
    // `name` es nombre propio del modelo: se muestra tal cual, no se traduce.
    return audio?.cleanupSteps?.find((step) => step.id === id)?.name ?? null;
  }
  if (catalog === "separationModel") {
    return audio?.separationModels?.find((model) => model.id === id)?.name ?? null;
  }
  if (catalog === "masteringPreset") {
    const preset = audio?.masteringPresets?.find((entry) => entry.id === id);
    return preset ? t(preset.labelKey) : null;
  }
  if (catalog === "voiceStep") {
    const step = voice?.steps.find((entry) => entry.id === id);
    return step ? t(step.labelKey) : null;
  }
  const delivery = voice?.deliveries.find((entry) => entry.id === id);
  return delivery ? t(delivery.labelKey) : null;
}

/**
 * El job guarda ids y la copia vive en los catalogos. Cuando el catalogo no
 * llego (todavia, o porque el endpoint fallo) se muestra el id: es feo pero
 * cierto, y no deja la fila vacia.
 */
export function useCatalogLabels(enabled: boolean): CatalogLabelResolver {
  const audioQuery = useQuery<AudioCapabilities>({
    queryKey: ["audioCapabilities"],
    queryFn: fetchAudioCapabilities,
    staleTime: ONE_HOUR_MS,
    enabled,
  });
  const voiceQuery = useQuery<VoiceCatalog>({
    queryKey: ["voice-catalog"],
    queryFn: fetchVoiceCatalog,
    staleTime: ONE_HOUR_MS,
    enabled,
  });
  const { t } = useTranslation();

  return (catalog, id) => pickLabel(catalog, id, audioQuery.data, voiceQuery.data, t) ?? id;
}
