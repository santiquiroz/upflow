import { useMemo, useState } from "react";
import type { SeparationModel } from "../../lib/apiTypes";

/**
 * Nunca transcribible: batería no tiene altura (decisión #10 del contrato
 * F3a). Mismo criterio que `_validate_transcribe_selection` en
 * audio_job_manager.py, del lado del backend.
 */
const UNTRANSCRIBABLE_STEM_IDS = new Set(["drums"]);

export interface TranscribeSelection {
  /** El modelo elegido tiene al menos un stem con altura para transcribir. */
  available: boolean;
  /** La subsección está encendida. Opt-in explícito, independiente de minus-one. */
  active: boolean;
  setActive: (active: boolean) => void;
  /** Ids con altura del modelo elegido, en el orden del catálogo. */
  transcribableStemIds: string[];
  /** Los stems elegidos a transcribir, en el orden del modelo. */
  enabledStems: string[];
  toggleStem: (id: string, enabled: boolean) => void;
  isEnabled: (id: string) => boolean;
}

function orderedByModel(stemIds: string[], selected: Set<string>): string[] {
  return stemIds.filter((id) => selected.has(id));
}

export function useTranscribeSelection(
  model: SeparationModel | undefined,
): TranscribeSelection {
  const [active, setActive] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);

  const transcribableStemIds = useMemo(
    () =>
      (model?.stems ?? [])
        .map((stem) => stem.id)
        .filter((id) => !UNTRANSCRIBABLE_STEM_IDS.has(id)),
    [model],
  );
  const available = transcribableStemIds.length > 0;

  // Derivar SIEMPRE desde las pistas del modelo, igual que minus-one: cambiar
  // de modelo no arrastra elecciones que ya no existen (o que pasaron a ser
  // batería en el modelo nuevo).
  const enabledStems = useMemo(
    () => (active && available ? orderedByModel(transcribableStemIds, new Set(selected)) : []),
    [active, available, transcribableStemIds, selected],
  );

  function toggleStem(id: string, enabled: boolean): void {
    if (!enabled) {
      setSelected((current) => current.filter((entry) => entry !== id));
      return;
    }
    setSelected((current) => [...current.filter((entry) => entry !== id), id]);
  }

  return {
    available,
    active,
    setActive,
    transcribableStemIds,
    enabledStems,
    toggleStem,
    isEnabled: (id: string) => enabledStems.includes(id),
  };
}
