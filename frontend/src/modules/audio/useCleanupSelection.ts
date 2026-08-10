import { useMemo, useState } from "react";
import type { CleanupStep } from "../../lib/apiTypes";

export interface CleanupSelection {
  /** La cadena de limpieza entera está encendida. Opt-in explícito. */
  active: boolean;
  setActive: (active: boolean) => void;
  /** Los ids elegidos, en orden de catálogo (= orden de ejecución). */
  enabledIds: string[];
  toggleStep: (id: string, enabled: boolean) => void;
  isEnabled: (id: string) => boolean;
  /** Encender este paso apagaría estos otros, porque hacen la misma tarea. */
  conflictsOf: (id: string) => string[];
  /** Hay tres pasadas encadenadas: cada una es con pérdida. */
  isOverprocessing: boolean;
}

/**
 * Dos pasos se excluyen cuando resuelven alguna familia en común. La regla sale
 * del catálogo (`covers`), no de una lista de ids acá: `deecho_dereverb` hace
 * eco y reverb en una pasada, así que excluye a los de-echo Y a `reverb_hq` sin
 * que la UI tenga que saber cuáles son.
 */
function conflictsWith(step: CleanupStep, other: CleanupStep): boolean {
  if (step.id === other.id) {
    return false;
  }
  return step.covers.some((family) => other.covers.includes(family));
}

function stepsById(steps: CleanupStep[]): Map<string, CleanupStep> {
  return new Map(steps.map((step) => [step.id, step]));
}

function orderedByCatalog(steps: CleanupStep[], selected: Set<string>): string[] {
  return steps.filter((step) => selected.has(step.id)).map((step) => step.id);
}

export function useCleanupSelection(
  steps: CleanupStep[] | undefined,
  overprocessingThreshold = 3,
): CleanupSelection {
  const [active, setActive] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);

  const catalog = useMemo(() => steps ?? [], [steps]);
  const byId = useMemo(() => stepsById(catalog), [catalog]);

  // El orden lo fija el catálogo, igual que en el backend. La lista viaja
  // ordenada aunque el backend la reordene igual: así lo que el usuario ve
  // enumerado en pantalla es lo que se va a correr, en ese orden.
  const enabledIds = useMemo(
    () => (active ? orderedByCatalog(catalog, new Set(selected)) : []),
    [active, catalog, selected],
  );

  function conflictsOf(id: string): string[] {
    const step = byId.get(id);
    if (!step) {
      return [];
    }
    return catalog.filter((other) => conflictsWith(step, other)).map((other) => other.id);
  }

  function toggleStep(id: string, enabled: boolean): void {
    if (!enabled) {
      setSelected((current) => current.filter((entry) => entry !== id));
      return;
    }
    // Encender uno APAGA a los que hacen su misma tarea: el backend rechaza la
    // combinación redundante, así que dejarla armar sería ofrecer un estado que
    // ya se sabe inválido.
    const excluded = new Set(conflictsOf(id));
    setSelected((current) => [...current.filter((entry) => !excluded.has(entry) && entry !== id), id]);
  }

  return {
    active,
    setActive,
    enabledIds,
    toggleStep,
    isEnabled: (id: string) => enabledIds.includes(id),
    conflictsOf,
    isOverprocessing: enabledIds.length >= overprocessingThreshold,
  };
}
