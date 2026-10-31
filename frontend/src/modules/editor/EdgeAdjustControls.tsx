import { useId } from "react";
import { useTranslation } from "../../i18n/LocaleProvider";
import type { EdgeAdjust } from "./editorMaskCanvas";

const EXPAND_MIN = -50;
const EXPAND_MAX = 50;
const SOFTEN_MAX = 30;

export function EdgeAdjustControls({ value, onChange }: { value: EdgeAdjust; onChange: (value: EdgeAdjust) => void }) {
  const { t } = useTranslation();
  return (
    <div className="mt-3 flex flex-col gap-3">
      <EdgeSlider
        label={t("editor.edge.expand")}
        min={EXPAND_MIN}
        max={EXPAND_MAX}
        value={value.growPx}
        onChange={(growPx) => onChange({ ...value, growPx })}
      />
      <EdgeSlider
        label={t("editor.edge.soften")}
        min={0}
        max={SOFTEN_MAX}
        value={value.featherPx}
        onChange={(featherPx) => onChange({ ...value, featherPx })}
      />
      <p className="text-xs text-text-faint">{t("editor.edge.hint")}</p>
    </div>
  );
}

function EdgeSlider({ label, min, max, value, onChange }: {
  label: string;
  min: number;
  max: number;
  value: number;
  onChange: (value: number) => void;
}) {
  const id = useId();
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between text-sm text-text">
        <label htmlFor={id}>{label}</label>
        <span className="font-mono-tabular text-text-dim">{formatPx(value)}</span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={1}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </div>
  );
}

function formatPx(value: number): string {
  return `${value > 0 ? "+" : ""}${value} px`;
}
