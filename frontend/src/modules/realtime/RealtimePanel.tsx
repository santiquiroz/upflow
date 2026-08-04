import { useQuery } from "@tanstack/react-query";
import { Zap } from "lucide-react";
import { useState } from "react";
import type { RealtimePreset } from "../../lib/apiTypes";
import { fetchRealtimeCapabilities, startRealtime } from "../../services/realtime";

// El overlay lo presenta Magpie, que corre como proceso aparte. Esta pantalla
// es el plano de control: elegís el preset, se escribe su config y se lanza.
// No hace falta ningún driver: el overlay es una ventana normal, no engancha el
// swapchain del juego.

const FRAME_RATE_OPTIONS = [
  { value: null, label: "Sin límite" },
  { value: 60, label: "60 fps" },
  { value: 120, label: "120 fps" },
  { value: 144, label: "144 fps" },
] as const;

function PresetOption({
  preset,
  checked,
  onSelect,
}: {
  preset: RealtimePreset;
  checked: boolean;
  onSelect: () => void;
}) {
  return (
    <label className="flex cursor-pointer items-start gap-3 rounded border border-border bg-surface p-3 transition-[border-color] duration-fast hover:border-accent">
      <input
        type="radio"
        name="realtime-preset"
        value={preset.id}
        checked={checked}
        onChange={onSelect}
        className="mt-0.5 h-3.5 w-3.5 accent-accent"
      />
      <span className="flex flex-col gap-0.5">
        <span className="text-sm font-medium text-text">{preset.label}</span>
        <span className="text-xs text-text-dim">{preset.description}</span>
      </span>
    </label>
  );
}

export function RealtimePanel() {
  const [preset, setPreset] = useState<string | null>(null);
  const [maxFrameRate, setMaxFrameRate] = useState<number | null>(null);
  const [launched, setLaunched] = useState<number | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);

  const capabilities = useQuery({
    queryKey: ["realtimeCapabilities"],
    queryFn: fetchRealtimeCapabilities,
  });

  if (capabilities.data && !capabilities.data.available) {
    return (
      <div role="alert" className="flex flex-col gap-2 rounded border border-border bg-surface p-4">
        <p className="text-sm text-text-dim">{capabilities.data.reason}</p>
        <p className="text-xs text-text-faint">
          Instalalo tildando <strong>Tiempo real</strong> en el instalador, o corriendo{" "}
          <code>scripts/download-magpie.ps1</code>.
        </p>
      </div>
    );
  }

  const presets = capabilities.data?.presets ?? [];
  const selected = preset ?? presets[0]?.id ?? null;

  async function handleStart() {
    if (!selected) {
      return;
    }
    setIsStarting(true);
    setErrorMessage(null);
    try {
      const started = await startRealtime(selected, maxFrameRate);
      setLaunched(started.pid);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "No se pudo iniciar el overlay.");
    } finally {
      setIsStarting(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <fieldset className="flex flex-col gap-2">
        <legend className="text-xs font-medium text-text-dim">Modo de reescalado</legend>
        <div className="grid grid-cols-2 gap-2 max-[700px]:grid-cols-1">
          {presets.map((item) => (
            <PresetOption
              key={item.id}
              preset={item}
              checked={selected === item.id}
              onSelect={() => setPreset(item.id)}
            />
          ))}
        </div>
      </fieldset>

      <div className="flex flex-col gap-2">
        <label htmlFor="realtime-fps" className="text-xs font-medium text-text-dim">
          Límite de cuadros
        </label>
        <select
          id="realtime-fps"
          value={maxFrameRate === null ? "" : String(maxFrameRate)}
          onChange={(event) =>
            setMaxFrameRate(event.target.value === "" ? null : Number(event.target.value))
          }
          className="w-fit rounded border border-border bg-surface p-2 text-sm text-text"
        >
          {FRAME_RATE_OPTIONS.map((option) => (
            <option key={option.label} value={option.value === null ? "" : String(option.value)}>
              {option.label}
            </option>
          ))}
        </select>
      </div>

      <div className="flex flex-col gap-2">
        <button
          type="button"
          onClick={() => void handleStart()}
          disabled={!selected || isStarting}
          className="inline-flex w-fit items-center gap-2 rounded bg-accent px-4 py-2 text-sm font-medium text-bg transition-[background-color,opacity] duration-fast hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
        >
          <Zap aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
          {isStarting ? "Abriendo…" : "Abrir overlay"}
        </button>
        {launched !== null && (
          // No se nombra un atajo concreto: Magpie lo guarda como un codigo
          // empaquetado y su ventana ya lo muestra. Inventarlo seria peor que
          // mandar a mirarlo donde de verdad esta.
          <p role="status" className="text-xs text-ok">
            Overlay abierto. Poné el foco en la ventana que querés escalar y usá el atajo que
            muestra la ventana de Magpie. El mismo atajo lo cierra.
          </p>
        )}
        {errorMessage && (
          <p role="alert" className="text-xs text-danger">
            {errorMessage}
          </p>
        )}
      </div>
    </div>
  );
}
