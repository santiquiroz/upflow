import { useTranslation } from "../../../i18n/LocaleProvider";
import type { CctvFrameIndex } from "../../../services/cctv";
import { frameTimecode, fullRange, isTrimValid, trimFrameCount, type TrimRange } from "./cctvFrames";

interface TrimControlsProps {
  frameCount: number;
  currentFrame: number;
  index: CctvFrameIndex | null;
  trim: TrimRange | null;
  onChange: (trim: TrimRange | null) => void;
}

type TrimEnd = "start" | "end";

const FIELD_CLASS =
  "font-mono-tabular w-24 rounded-sm border border-border bg-surface px-2 py-1 text-sm text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";
const LINK_BUTTON_CLASS =
  "rounded-sm px-1 text-xs text-accent underline-offset-2 hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";

function parseFrame(raw: string): number {
  const value = Math.trunc(Number(raw));
  return Number.isFinite(value) ? value : 0;
}

function withEnd(trim: TrimRange, end: TrimEnd, frame: number): TrimRange {
  return end === "start" ? [frame, trim[1]] : [trim[0], frame];
}

function TrimField({
  end,
  trim,
  currentFrame,
  index,
  onChange,
}: {
  end: TrimEnd;
  trim: TrimRange;
  currentFrame: number;
  index: CctvFrameIndex | null;
  onChange: (trim: TrimRange) => void;
}) {
  const { t } = useTranslation();
  const value = end === "start" ? trim[0] : trim[1];
  const id = `cctv-trim-${end}`;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <label htmlFor={id} className="w-24 text-xs text-text-dim">
        {t(`cctv.trim.${end}`)}
      </label>
      <input
        id={id}
        type="number"
        min={0}
        step={1}
        value={value}
        onChange={(event) => onChange(withEnd(trim, end, parseFrame(event.target.value)))}
        className={FIELD_CLASS}
      />
      <span className="font-mono-tabular text-xs text-text-dim" title={t("cctv.frames.timecode")}>
        {frameTimecode(index, value) ?? ""}
      </span>
      <button type="button" onClick={() => onChange(withEnd(trim, end, currentFrame))} className={LINK_BUTTON_CLASS}>
        {t(`cctv.trim.${end}.useCurrent`)}
      </button>
    </div>
  );
}

function countKey(count: number): string {
  return count === 1 ? "cctv.trim.count.one" : "cctv.trim.count";
}

function TrimSummary({ trim, frameCount }: { trim: TrimRange; frameCount: number }) {
  const { t } = useTranslation();
  if (!isTrimValid(trim, frameCount)) {
    return (
      <p role="alert" className="text-xs text-danger">
        {t("cctv.trim.invalid")}
      </p>
    );
  }
  const count = trimFrameCount(trim);
  return <p className="text-xs text-text-dim">{t(countKey(count), { count })}</p>;
}

export function TrimControls({ frameCount, currentFrame, index, trim, onChange }: TrimControlsProps) {
  const { t } = useTranslation();
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="sr-only">{t("cctv.trim.toggle")}</legend>
      <label className="flex items-center gap-2 text-sm text-text">
        <input
          type="checkbox"
          checked={trim !== null}
          onChange={(event) => onChange(event.target.checked ? fullRange(frameCount) : null)}
          className="h-3.5 w-3.5 accent-accent"
        />
        {t("cctv.trim.toggle")}
      </label>
      {trim && (
        <div className="flex flex-col gap-2 pl-5">
          <TrimField end="start" trim={trim} currentFrame={currentFrame} index={index} onChange={onChange} />
          <TrimField end="end" trim={trim} currentFrame={currentFrame} index={index} onChange={onChange} />
          <TrimSummary trim={trim} frameCount={frameCount} />
        </div>
      )}
    </fieldset>
  );
}
