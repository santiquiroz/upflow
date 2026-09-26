import { useId } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { RestorePreset } from "../../../lib/restoreApiTypes";

interface RestorePresetPickerProps {
  presets: RestorePreset[];
  selectedId: string;
  suggestedIds: string[];
  customized: boolean;
  onSelect: (presetId: string) => void;
}

interface PresetOptionProps {
  preset: RestorePreset;
  name: string;
  checked: boolean;
  suggested: boolean;
  onSelect: () => void;
}

function PresetOption({ preset, name, checked, suggested, onSelect }: PresetOptionProps) {
  const { t } = useTranslation();
  const radioId = useId();
  const descriptionId = useId();

  return (
    <div
      className={`flex items-start gap-2.5 rounded border px-3 py-2 transition-[background-color,border-color] duration-fast motion-reduce:transition-none ${
        checked ? "border-accent bg-surface-2" : "border-border bg-surface hover:bg-surface-2"
      }`}
    >
      <input
        id={radioId}
        type="radio"
        name={name}
        value={preset.id}
        checked={checked}
        aria-describedby={descriptionId}
        onChange={onSelect}
        className="mt-0.5 h-3.5 w-3.5 shrink-0 cursor-pointer accent-accent"
      />
      <div className="flex min-w-0 flex-col gap-0.5">
        <div className="flex flex-wrap items-center gap-2">
          <label htmlFor={radioId} className="cursor-pointer text-sm text-text">
            {t(preset.labelKey)}
          </label>
          {suggested && (
            <span className="rounded-sm border border-accent/60 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-accent">
              {t("restore.preset.suggested")}
            </span>
          )}
        </div>
        <p id={descriptionId} className="text-xs leading-relaxed text-text-dim">
          {t(preset.descriptionKey)}
        </p>
      </div>
    </div>
  );
}

export function RestorePresetPicker({
  presets,
  selectedId,
  suggestedIds,
  customized,
  onSelect,
}: RestorePresetPickerProps) {
  const { t } = useTranslation();
  const name = useId();

  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="mb-1 text-xs font-medium uppercase tracking-wide text-text-faint">
        {t("restore.preset.title")}
      </legend>
      {presets.map((preset) => (
        <PresetOption
          key={preset.id}
          preset={preset}
          name={name}
          checked={preset.id === selectedId}
          suggested={suggestedIds.includes(preset.id)}
          onSelect={() => onSelect(preset.id)}
        />
      ))}
      {customized && (
        <p className="flex flex-wrap items-center gap-2 text-xs text-text-faint">
          {t("restore.preset.customized")}
          <button
            type="button"
            onClick={() => onSelect(selectedId)}
            className="text-text-dim underline underline-offset-2 hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
          >
            {t("restore.preset.reset")}
          </button>
        </p>
      )}
    </fieldset>
  );
}
