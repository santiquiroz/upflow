import { useQuery } from "@tanstack/react-query";
import { getModels } from "../lib/api";
import type { ModelResponse } from "../lib/apiTypes";

interface ModelPickerProps {
  value: string | null;
  onChange: (model: ModelResponse) => void;
  // El reescalado sin IA lo hace swscale DENTRO del encode de video, asi que hoy solo
  // existe para video. Default false (fail-closed): ofrecerlo donde el pipeline no lo
  // sabe ejecutar seria una opcion que falla al enviar.
  allowNoAi?: boolean;
}

interface ModelGroup {
  label: string;
  models: ModelResponse[];
}

// Explicit kind === "onnx" (not "everything non-builtin") so kinds like
// diffusion-onnx never leak into the upscale picker's Installed group.
//
// "No AI" goes LAST and named for what it is, not for how it works: someone who picks
// it must know no model runs, and someone who wants AI must not land on it by accident.
function groupModels(models: ModelResponse[], allowNoAi: boolean): ModelGroup[] {
  const groups: ModelGroup[] = [
    { label: "Builtin", models: models.filter((model) => model.kind === "builtin-ncnn") },
    { label: "ONNX", models: models.filter((model) => model.kind === "onnx") },
    {
      label: "No AI (classic resize)",
      models: allowNoAi ? models.filter((model) => model.kind === "classic") : [],
    },
  ];
  return groups.filter((group) => group.models.length > 0);
}

function formatModelMeta(model: ModelResponse): string {
  // El clasico no tiene escala fija (va a cualquier medida en una pasada) ni
  // arquitectura: mostrar "—x · classic" no diria nada util.
  if (model.kind === "classic") {
    return "any scale · CPU, no model";
  }
  const scale = model.scale ? `${model.scale}x` : "—";
  const arch = model.arch ?? model.kind;
  return `${scale} · ${arch}`;
}

function isModelSelectable(model: ModelResponse): boolean {
  return model.status === "installed";
}

function modelOptionClassName(isSelected: boolean, isDisabled: boolean): string {
  const base =
    "flex cursor-pointer flex-col gap-1 rounded border px-3 py-2 transition-[background-color,border-color] duration-fast focus-within:outline focus-within:outline-2 focus-within:outline-accent";
  if (isDisabled) {
    return `${base} cursor-not-allowed border-border bg-surface opacity-50`;
  }
  if (isSelected) {
    return `${base} border-accent bg-surface-2`;
  }
  return `${base} border-border bg-surface hover:border-text-faint`;
}

function ModelOption({
  model,
  isSelected,
  onChange,
}: {
  model: ModelResponse;
  isSelected: boolean;
  onChange: (model: ModelResponse) => void;
}) {
  const isDisabled = !isModelSelectable(model);
  return (
    <label className={modelOptionClassName(isSelected, isDisabled)}>
      <span className="flex items-center gap-2">
        <input
          type="radio"
          name="model"
          value={model.id}
          checked={isSelected}
          disabled={isDisabled}
          onChange={() => onChange(model)}
          className="h-3.5 w-3.5 accent-accent"
        />
        <span className="text-sm text-text">{model.name}</span>
      </span>
      <span className="font-mono-tabular pl-[22px] text-xs text-text-dim">{formatModelMeta(model)}</span>
      {isDisabled && (
        <span className="pl-[22px] text-xs text-warn">
          {model.status === "converting" ? "Converting…" : (model.error ?? "Not ready")}
        </span>
      )}
    </label>
  );
}

export function ModelPicker({ value, onChange, allowNoAi = false }: ModelPickerProps) {
  const modelsQuery = useQuery({ queryKey: ["models"], queryFn: getModels });

  if (modelsQuery.isLoading) {
    return <p className="text-sm text-text-dim">Loading models…</p>;
  }

  if (modelsQuery.isError) {
    return <p className="text-sm text-danger">Could not load models.</p>;
  }

  const groups = groupModels(modelsQuery.data?.models ?? [], allowNoAi);

  return (
    <fieldset className="flex flex-col gap-4">
      <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">Model</legend>
      {groups.map((group) => (
        <div key={group.label} role="group" aria-label={group.label} className="flex flex-col gap-2">
          <h3 className="text-xs font-medium text-text-faint">{group.label}</h3>
          {group.models.map((model) => (
            <ModelOption key={model.id} model={model} isSelected={model.id === value} onChange={onChange} />
          ))}
        </div>
      ))}
    </fieldset>
  );
}
