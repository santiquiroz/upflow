import { useMemo, useState } from "react";
import type { VoiceCatalog } from "../../lib/apiTypes";
import { PRESENCE_DEFAULT_DB } from "./PresenceControl";

// Ajustar loudness sin destino no tiene numero al que ajustar, y el backend lo
// rechaza. Encender el paso preselecciona el destino mas comun en vez de dejar
// la seleccion en un estado que se sabe invalido.
const DEFAULT_DELIVERY = "streaming";
const LOUDNESS_STEP_ID = "loudness";
const PRESENCE_STEP_ID = "presence";

export interface VoiceSelection {
  /** La mejora de voz entera esta encendida. Opt-in explicito. */
  active: boolean;
  setActive: (active: boolean) => void;
  enabledIds: string[];
  delivery: string | null;
  presenceDb: number;
  toggleStep: (id: string, enabled: boolean) => void;
  setDelivery: (id: string) => void;
  setPresenceDb: (db: number) => void;
  isEnabled: (id: string) => boolean;
  /** Falta elegir destino para el paso de loudness que esta encendido. */
  needsDelivery: boolean;
}

function defaultsFrom(catalog: VoiceCatalog | undefined): string[] {
  return (catalog?.steps ?? []).filter((step) => step.defaultEnabled).map((s) => s.id);
}

export function useVoiceSelection(catalog: VoiceCatalog | undefined): VoiceSelection {
  // null = el usuario no toco nada todavia, asi que valen los defaults del
  // catalogo. Se deriva en vez de sincronizar con un efecto: el catalogo llega
  // async y un efecto de inicializacion pisaria la eleccion del usuario en cada
  // refetch.
  // La mejora de voz arranca APAGADA. `defaultEnabled` del catalogo describe la
  // forma sensata de la cadena una vez que alguien la pide, no que este pedida:
  // encenderla sola aplicaria procesado de voz a jobs donde solo se pidio quitar
  // ruido.
  const [active, setActive] = useState(false);
  const [touched, setTouched] = useState<string[] | null>(null);
  const [delivery, setDeliveryState] = useState<string | null>(null);
  const [presenceDb, setPresenceDb] = useState(PRESENCE_DEFAULT_DB);

  const enabledIds = useMemo(
    () => (active ? (touched ?? defaultsFrom(catalog)) : []),
    [active, touched, catalog],
  );

  function toggleStep(id: string, enabled: boolean): void {
    const next = enabled
      ? [...enabledIds.filter((current) => current !== id), id]
      : enabledIds.filter((current) => current !== id);
    setTouched(next);
    if (id === LOUDNESS_STEP_ID && enabled && delivery === null) {
      setDeliveryState(DEFAULT_DELIVERY);
    }
  }

  const loudnessOn = enabledIds.includes(LOUDNESS_STEP_ID);

  return {
    active,
    setActive,
    // El ORDEN de esta lista es irrelevante para el backend, que reordena segun
    // el catalogo; se manda tal cual para no duplicar esa regla aca.
    enabledIds,
    delivery: loudnessOn ? delivery : null,
    presenceDb,
    toggleStep,
    setDelivery: setDeliveryState,
    setPresenceDb,
    isEnabled: (id: string) => enabledIds.includes(id),
    needsDelivery: loudnessOn && delivery === null,
  };
}

export { LOUDNESS_STEP_ID, PRESENCE_STEP_ID };
