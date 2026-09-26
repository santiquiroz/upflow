import { TriangleAlert } from "lucide-react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { CctvFrameIndex } from "../../../services/cctv";
import { clampFrame, frameTimecode } from "./cctvFrames";
import { CctvOptionGroup } from "./CctvOptionGroup";
import { MAX_ROI_FRAMES, ROI_KINDS, ROI_METHODS, ROI_SCALES, roiDensityNotice, roiFrameCount, type RoiChoice } from "./cctvRoi";

type RangeEnd = "first" | "last";

// Mismos textos que el recorte: primer y ultimo cuadro del rango.
const TRIM_KEYS: Readonly<Record<RangeEnd, string>> = { first: "cctv.trim.start", last: "cctv.trim.end" };

interface RoiFusionPanelProps {
  roi: RoiChoice;
  frame: number;
  frameCount: number;
  index: CctvFrameIndex | null;
  onChange: (roi: RoiChoice) => void;
  onShowFrame: (frame: number) => void;
}

const FIELD_CLASS =
  "font-mono-tabular w-24 rounded-sm border border-border bg-surface px-2 py-1 text-sm text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";
const LINK_BUTTON_CLASS =
  "rounded-sm px-1 text-xs text-accent underline-offset-2 hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";

function parsedFrame(raw: string, frameCount: number): number | null {
  return raw.trim() === "" ? null : clampFrame(Number(raw), frameCount);
}

function RangeField({
  end,
  roi,
  frame,
  frameCount,
  index,
  onChange,
}: {
  end: RangeEnd;
  roi: RoiChoice;
  frame: number;
  frameCount: number;
  index: CctvFrameIndex | null;
  onChange: (roi: RoiChoice) => void;
}) {
  const { t } = useTranslation();
  const value = roi[end];
  const id = `cctv-roi-${end}`;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <label htmlFor={id} className="w-24 text-xs text-text-dim">
        {t(TRIM_KEYS[end])}
      </label>
      <input
        id={id}
        type="number"
        min={0}
        step={1}
        value={value ?? ""}
        onChange={(event) => onChange({ ...roi, [end]: parsedFrame(event.target.value, frameCount) })}
        className={FIELD_CLASS}
      />
      <span className="font-mono-tabular text-xs text-text-dim" title={t("cctv.frames.timecode")}>
        {value === null ? "" : frameTimecode(index, value) ?? ""}
      </span>
      <button type="button" onClick={() => onChange({ ...roi, [end]: frame })} className={LINK_BUTTON_CLASS}>
        {t(`${TRIM_KEYS[end]}.useCurrent`)}
      </button>
    </div>
  );
}

function RangeSummary({ roi }: { roi: RoiChoice }) {
  const { t } = useTranslation();
  const count = roiFrameCount(roi);
  const text =
    count === null
      ? t("cctv.roi.range.empty", { max: MAX_ROI_FRAMES })
      : t("cctv.roi.range.count", { count, max: MAX_ROI_FRAMES });
  const tone = count !== null && count > MAX_ROI_FRAMES ? "text-danger" : "text-text-dim";
  return <p className={`text-xs ${tone}`}>{text}</p>;
}

function ReferenceLine({ reference, onShowFrame }: { reference: number | null; onShowFrame: (frame: number) => void }) {
  const { t } = useTranslation();
  if (reference === null) {
    return <p className="text-xs text-text-dim">{t("cctv.roi.reference.none")}</p>;
  }
  return (
    <p className="flex flex-wrap items-center gap-2 text-xs text-text">
      <span className="font-mono-tabular">{t("cctv.roi.reference", { frame: reference })}</span>
      <button type="button" onClick={() => onShowFrame(reference)} className={LINK_BUTTON_CLASS}>
        {t("cctv.roi.reference.show")}
      </button>
    </p>
  );
}

function DensityNotice({ roi }: { roi: RoiChoice }) {
  const { t } = useTranslation();
  const notice = roiDensityNotice(roi.kind, roi.box);
  if (!notice) {
    return null;
  }
  return (
    <p role="note" className="flex items-start gap-1.5 text-xs text-warn">
      <TriangleAlert aria-hidden="true" className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
      {t(notice.key, notice.params)}
    </p>
  );
}

function RoiSettings({ roi, onChange }: { roi: RoiChoice; onChange: (roi: RoiChoice) => void }) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-3">
      <CctvOptionGroup
        legend={t("cctv.roi.kind.legend")}
        options={ROI_KINDS.map((kind) => ({ value: kind, label: t(`cctv.roi.kind.${kind}`) }))}
        value={roi.kind}
        onChange={(kind) => onChange({ ...roi, kind })}
      />
      <CctvOptionGroup
        legend={t("cctv.roi.scale.legend")}
        options={ROI_SCALES.map((scale) => ({ value: scale, label: t("cctv.scaleOption", { scale }) }))}
        value={roi.scale}
        onChange={(scale) => onChange({ ...roi, scale })}
      />
      <CctvOptionGroup
        legend={t("cctv.roi.method.legend")}
        options={ROI_METHODS.map((method) => ({ value: method, label: t(`cctv.roi.method.${method}`) }))}
        value={roi.method}
        onChange={(method) => onChange({ ...roi, method })}
      />
    </div>
  );
}

export function RoiFusionPanel({ roi, frame, frameCount, index, onChange, onShowFrame }: RoiFusionPanelProps) {
  const { t } = useTranslation();
  return (
    <section aria-label={t("cctv.roi.legend")} className="flex flex-col gap-3 rounded border border-border bg-surface p-3">
      <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">{t("cctv.roi.legend")}</span>
      <p className="text-xs text-text-dim">{t("cctv.roi.intro")}</p>
      <RangeField end="first" roi={roi} frame={frame} frameCount={frameCount} index={index} onChange={onChange} />
      <RangeField end="last" roi={roi} frame={frame} frameCount={frameCount} index={index} onChange={onChange} />
      <RangeSummary roi={roi} />
      <ReferenceLine reference={roi.reference} onShowFrame={onShowFrame} />
      <RoiSettings roi={roi} onChange={onChange} />
      <DensityNotice roi={roi} />
    </section>
  );
}
