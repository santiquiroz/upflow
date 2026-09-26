import { EyeOff } from "lucide-react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { CctvBox } from "../../../services/cctv";
import { BoxEditor } from "./BoxEditor";
import type { FrameSize } from "./cctvBoxes";
import type { TrimRange } from "./cctvFrames";
import { CctvOptionGroup } from "./CctvOptionGroup";
import {
  activeBoxes,
  hasKeyframeAt,
  MAX_REDACTION_TRACKS,
  overlapsSpan,
  rangesText,
  REDACTION_STYLES,
  uncoveredRanges,
  withEditedBoxes,
  withKeyframeRemoved,
  withStyle,
  withTrackEnd,
  withTrackRemoved,
  withTrackStart,
  type RedactionChoice,
  type RedactionTrack,
} from "./cctvRedaction";

const LINK_BUTTON_CLASS =
  "rounded-sm px-1 text-xs text-accent underline-offset-2 hover:underline disabled:cursor-not-allowed disabled:text-text-faint disabled:no-underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";
const DANGER_BUTTON_CLASS =
  "rounded-sm px-1 text-xs text-danger underline-offset-2 hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";

interface RedactionBoxEditorProps {
  redaction: RedactionChoice;
  frame: number;
  frameSize: FrameSize;
  span: TrimRange;
  onChange: (redaction: RedactionChoice) => void;
}

// Solo las cajas que tapan este cuadro: moverla aca agrega una keyframe, dibujar una nueva la crea para todo el tramo.
export function RedactionBoxEditor({ redaction, frame, frameSize, span, onChange }: RedactionBoxEditorProps) {
  const active = activeBoxes(redaction, frame);
  return (
    <BoxEditor
      kind="redact"
      frameSize={frameSize}
      boxes={active.map((item) => item.box)}
      labels={active.map((item) => item.number)}
      onChange={(boxes: CctvBox[]) => onChange(withEditedBoxes(redaction, active, boxes, frame, span))}
    />
  );
}

interface RedactionPanelProps {
  redaction: RedactionChoice;
  frame: number;
  span: TrimRange;
  onChange: (redaction: RedactionChoice) => void;
  onShowFrame: (frame: number) => void;
}

function TrackRow({
  track,
  number,
  frame,
  span,
  redaction,
  onChange,
  onShowFrame,
}: {
  track: RedactionTrack;
  number: number;
  frame: number;
  span: TrimRange;
  redaction: RedactionChoice;
  onChange: (redaction: RedactionChoice) => void;
  onShowFrame: (frame: number) => void;
}) {
  const { t } = useTranslation();
  const canRemoveKeyframe = hasKeyframeAt(track, frame) && track.keyframes.length > 1;
  const isOutside = !overlapsSpan(track, span);
  const rowClass = isOutside ? "border-danger" : "border-border";
  return (
    <li className={`flex flex-col gap-1 rounded-sm border bg-surface px-3 py-2 ${rowClass}`}>
      <span className="font-mono-tabular text-xs text-text">
        {t("cctv.redact.track", {
          index: number,
          first: track.firstFrame,
          last: track.lastFrame,
          count: track.keyframes.length,
        })}
      </span>
      {isOutside && <span className="text-xs text-danger">{t("cctv.redact.outsideTrim", { first: span[0], last: span[1] })}</span>}
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" onClick={() => onShowFrame(track.firstFrame)} className={LINK_BUTTON_CLASS}>
          {t("cctv.redact.showStart")}
        </button>
        <button type="button" onClick={() => onChange(withTrackStart(redaction, track.id, frame))} className={LINK_BUTTON_CLASS}>
          {t("cctv.redact.startHere")}
        </button>
        <button type="button" onClick={() => onChange(withTrackEnd(redaction, track.id, frame))} className={LINK_BUTTON_CLASS}>
          {t("cctv.redact.endHere")}
        </button>
        <button
          type="button"
          disabled={!canRemoveKeyframe}
          onClick={() => onChange(withKeyframeRemoved(redaction, track.id, frame))}
          className={LINK_BUTTON_CLASS}
        >
          {t("cctv.redact.removeKeyframe")}
        </button>
        <button
          type="button"
          onClick={() => onChange(withTrackRemoved(redaction, track.id))}
          aria-label={t("cctv.redact.remove", { index: number })}
          className={DANGER_BUTTON_CLASS}
        >
          {t("cctv.redact.removeShort")}
        </button>
      </div>
    </li>
  );
}

function TrackList({ redaction, frame, span, onChange, onShowFrame }: RedactionPanelProps) {
  const { t } = useTranslation();
  if (redaction.tracks.length === 0) {
    return <p className="text-xs text-text-dim">{t("cctv.redact.empty")}</p>;
  }
  return (
    <ol aria-label={t("cctv.redact.boxes")} className="flex flex-col gap-2">
      {redaction.tracks.map((track, position) => (
        <TrackRow
          key={track.id}
          track={track}
          number={position + 1}
          frame={frame}
          span={span}
          redaction={redaction}
          onChange={onChange}
          onShowFrame={onShowFrame}
        />
      ))}
    </ol>
  );
}

// Los cuadros de la copia que ninguna caja tapa salen como se grabaron: se nombran para que nadie lo descubra al compartir.
function UncoveredNotice({ redaction, span }: { redaction: RedactionChoice; span: TrimRange }) {
  const { t } = useTranslation();
  const gaps = uncoveredRanges(redaction, span);
  if (redaction.tracks.length === 0 || gaps.length === 0) {
    return null;
  }
  return (
    <p role="status" className="text-xs text-warn">
      {t("cctv.redact.uncovered", { frames: rangesText(gaps) })}
    </p>
  );
}

export function RedactionPanel({ redaction, frame, span, onChange, onShowFrame }: RedactionPanelProps) {
  const { t } = useTranslation();
  const styles = REDACTION_STYLES.map((style) => ({ value: style, label: t(`cctv.redact.style.${style}`) }));
  return (
    <div className="flex flex-col gap-3">
      <p role="note" className="flex items-start gap-2 text-xs text-text">
        <EyeOff aria-hidden="true" className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
        {t("cctv.redact.notice")}
      </p>
      <p className="text-xs text-text-dim">{t("cctv.redact.help", { max: MAX_REDACTION_TRACKS })}</p>
      <CctvOptionGroup
        legend={t("cctv.redact.style.legend")}
        options={styles}
        value={redaction.style}
        onChange={(style) => onChange(withStyle(redaction, style))}
      />
      <p className="text-xs text-text-dim">{t(`cctv.redact.style.${redaction.style}Hint`)}</p>
      <TrackList redaction={redaction} frame={frame} span={span} onChange={onChange} onShowFrame={onShowFrame} />
      <UncoveredNotice redaction={redaction} span={span} />
      <p className="text-xs text-warn">{t("cctv.redact.noDetector")}</p>
    </div>
  );
}
