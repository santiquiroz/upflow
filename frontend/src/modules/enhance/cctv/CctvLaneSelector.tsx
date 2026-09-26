import { useTranslation } from "../../../i18n/LocaleProvider";
import type { CctvLane } from "../../../services/cctv";
import type { AiLaneState } from "./cctvLanes";

function laneCardClassName(isActive: boolean): string {
  const base =
    "flex flex-1 flex-col gap-1 rounded border p-3 text-left transition-[border-color,background-color] duration-fast focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-50";
  return isActive ? `${base} border-accent bg-surface-2` : `${base} border-border bg-surface hover:border-text-faint`;
}

function LaneCard({
  lane,
  active,
  disabled,
  note,
  onSelect,
}: {
  lane: CctvLane;
  active: boolean;
  disabled: boolean;
  note: string | null;
  onSelect: (lane: CctvLane) => void;
}) {
  const { t } = useTranslation();
  return (
    <button
      type="button"
      role="radio"
      aria-checked={active}
      disabled={disabled}
      onClick={() => onSelect(lane)}
      className={laneCardClassName(active)}
    >
      <span className="text-sm font-medium text-text">{t(`cctv.lane.${lane}`)}</span>
      <span className="text-xs text-text-dim">{t(`cctv.lane.${lane}.hint`)}</span>
      {note && <span className="text-xs text-warn">{note}</span>}
    </button>
  );
}

export function CctvLaneSelector({
  value,
  ai,
  onChange,
}: {
  value: CctvLane;
  ai: AiLaneState;
  onChange: (lane: CctvLane) => void;
}) {
  const { t } = useTranslation();
  const aiNote = ai.reason ? t(ai.reason.key, ai.reason.params) : null;
  return (
    <div role="radiogroup" aria-label={t("cctv.lane.legend")} className="flex gap-3 max-[640px]:flex-col">
      <LaneCard lane="classic" active={value === "classic"} disabled={false} note={null} onSelect={onChange} />
      <LaneCard lane="ai" active={value === "ai"} disabled={!ai.available} note={aiNote} onSelect={onChange} />
    </div>
  );
}
