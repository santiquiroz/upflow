import { useMemo, useState } from "react";
import type { SeparationModel } from "../../lib/apiTypes";

/** Escalones del volumen de guía. El backend valida 0-30; el control no ofrece
 *  valores intermedios porque a esta escala no se distinguen de oído. */
export const GUIDE_PERCENT_LEVELS: readonly number[] = [0, 10, 20, 30];

/** Con dos stems, "la mezcla sin uno" es el otro stem tal cual: eso ya lo da el
 *  modo karaoke. Minus-one solo tiene sentido desde tres pistas. */
const MIN_STEMS = 3;

export interface RehearsalSelection {
  /** El modelo elegido puede armar minus-one: separa tres pistas o más. */
  available: boolean;
  /** La sección entera está encendida. Opt-in explícito. */
  active: boolean;
  setActive: (active: boolean) => void;
  /** Los stems a quitar, en el orden de pistas del modelo. */
  enabledStems: string[];
  toggleStem: (id: string, enabled: boolean) => void;
  isEnabled: (id: string) => boolean;
  /** Volumen de guía del instrumento quitado (0/10/20/30). */
  guidePercent: number;
  setGuidePercent: (percent: number) => void;
}

function orderedByModel(stemIds: string[], selected: Set<string>): string[] {
  return stemIds.filter((id) => selected.has(id));
}

export function useRehearsalSelection(
  model: SeparationModel | undefined,
): RehearsalSelection {
  const [active, setActive] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const [guidePercent, setGuidePercentState] = useState(0);

  const stemIds = useMemo(() => (model?.stems ?? []).map((stem) => stem.id), [model]);
  const available = stemIds.length >= MIN_STEMS;

  // Derivar SIEMPRE desde las pistas del modelo: cambiar de modelo no arrastra
  // elecciones que ya no existen, igual que la cadena de limpieza con su
  // catálogo.
  const enabledStems = useMemo(
    () => (active && available ? orderedByModel(stemIds, new Set(selected)) : []),
    [active, available, stemIds, selected],
  );

  function toggleStem(id: string, enabled: boolean): void {
    if (!enabled) {
      setSelected((current) => current.filter((entry) => entry !== id));
      return;
    }
    setSelected((current) => [...current.filter((entry) => entry !== id), id]);
  }

  function setGuidePercent(percent: number): void {
    // El control es escalonado: un valor sin botón no tiene forma legítima de
    // llegar acá, así que se ignora en vez de viajar al backend.
    if (GUIDE_PERCENT_LEVELS.includes(percent)) {
      setGuidePercentState(percent);
    }
  }

  return {
    available,
    active,
    setActive,
    enabledStems,
    toggleStem,
    isEnabled: (id: string) => enabledStems.includes(id),
    guidePercent,
    setGuidePercent,
  };
}
