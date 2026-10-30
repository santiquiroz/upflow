import type { GenerationModelSummary } from "../../lib/apiTypes";

export type EditMode = "erase" | "replace";

export function InpaintModelSelect({
  models,
  value,
  onChange,
  label,
  mode,
  hint,
}: {
  models: GenerationModelSummary[];
  value: string | null;
  onChange: (id: string | null) => void;
  label: string;
  mode: EditMode;
  hint?: string;
}) {
  const usable = models.filter(
    (model) =>
      model.status === "installed" &&
      model.supportsInpaint !== false &&
      model.speed !== "turbo" &&
      (mode === "erase" || !model.eraseOnly),
  );
  return (
    <label className="flex flex-col gap-1 text-sm text-text">
      {label}
      <select
        className="rounded border border-border bg-surface-2 px-2 py-1.5 text-sm text-text"
        value={value ?? ""}
        onChange={(event) => onChange(event.target.value || null)}
      >
        <option value="" disabled>
          —
        </option>
        {usable.map((model) => (
          <option key={model.id} value={model.id}>
            {model.name}
          </option>
        ))}
      </select>
      {hint && <span className="text-xs text-text-faint">{hint}</span>}
    </label>
  );
}
