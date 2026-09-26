import { useState, type KeyboardEvent, type PointerEvent } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { CctvBox } from "../../../services/cctv";
import { MAX_REDACTION_TRACKS } from "./cctvRedaction";
import {
  boxAfterKey,
  boxFromCorners,
  boxPercentStyle,
  MAX_OSD_BOXES,
  MAX_ROI_BOXES,
  pointOnFrame,
  withBoxAdded,
  withBoxRemoved,
  withBoxReplaced,
  type FramePoint,
  type FrameSize,
} from "./cctvBoxes";

export type BoxKind = "osd" | "roi" | "redact";

interface BoxEditorProps {
  kind: BoxKind;
  frameSize: FrameSize;
  boxes: readonly CctvBox[];
  onChange: (boxes: CctvBox[]) => void;
  flagged?: readonly number[];
  disabled?: boolean;
  // Numero que se muestra en cada caja; por defecto, su posicion. La anonimizacion numera por caja, no por cuadro.
  labels?: readonly number[];
}

interface Draft {
  start: FramePoint;
  end: FramePoint;
}

const REMOVE_KEYS: ReadonlySet<string> = new Set(["Delete", "Backspace"]);

const MAX_BOXES: Readonly<Record<BoxKind, number>> = { osd: MAX_OSD_BOXES, roi: MAX_ROI_BOXES, redact: MAX_REDACTION_TRACKS };
const SURFACE_LABELS: Readonly<Record<BoxKind, string>> = {
  osd: "cctv.box.surfaceOsd",
  roi: "cctv.box.surfaceRoi",
  redact: "cctv.box.surfaceRedact",
};

function maxBoxesOf(kind: BoxKind): number {
  return MAX_BOXES[kind];
}

function boxClassName(isFlagged: boolean): string {
  const tone = isFlagged ? "border-warn" : "border-accent";
  return `absolute border-2 ${tone} bg-transparent focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-1 focus-visible:outline-text`;
}

function SizeReading({ box, kind }: { box: CctvBox | null; kind: BoxKind }) {
  const { t } = useTranslation();
  const hint = kind === "roi" ? t("cctv.roi.tight") : null;
  const size = box ? t("cctv.roi.size", { w: box[2], h: box[3] }) : null;
  if (!hint && !size) {
    return null;
  }
  return (
    <p role="status" className="pointer-events-none absolute bottom-2 left-2 rounded-sm bg-surface px-2 py-1 font-mono-tabular text-xs text-text">
      {[size, hint].filter(Boolean).join(" · ")}
    </p>
  );
}

// Lectura en vivo: mientras se dibuja, la caja en curso; en una ROI ya dibujada, su tamano.
function readingBox(draftBox: CctvBox | null, kind: BoxKind, boxes: readonly CctvBox[]): CctvBox | null {
  if (draftBox) {
    return draftBox;
  }
  return kind === "roi" ? boxes[0] ?? null : null;
}

export function BoxEditor({ kind, frameSize, boxes, onChange, flagged = [], disabled = false, labels }: BoxEditorProps) {
  const { t } = useTranslation();
  const [draft, setDraft] = useState<Draft | null>(null);
  const draftBox = draft ? boxFromCorners(draft.start, draft.end, frameSize) : null;

  function pointOf(event: PointerEvent<HTMLDivElement>): FramePoint {
    return pointOnFrame(event.clientX, event.clientY, event.currentTarget.getBoundingClientRect(), frameSize);
  }

  function handlePointerDown(event: PointerEvent<HTMLDivElement>): void {
    if (disabled || event.target !== event.currentTarget) {
      return;
    }
    event.currentTarget.setPointerCapture?.(event.pointerId);
    const point = pointOf(event);
    setDraft({ start: point, end: point });
  }

  function handlePointerMove(event: PointerEvent<HTMLDivElement>): void {
    if (draft) {
      setDraft({ start: draft.start, end: pointOf(event) });
    }
  }

  function handlePointerUp(): void {
    if (draftBox) {
      onChange(withBoxAdded(boxes, draftBox, maxBoxesOf(kind)));
    }
    setDraft(null);
  }

  function handleBoxKey(event: KeyboardEvent<HTMLButtonElement>, index: number): void {
    if (REMOVE_KEYS.has(event.key)) {
      event.preventDefault();
      onChange(withBoxRemoved(boxes, index));
      return;
    }
    const next = boxAfterKey(boxes[index], event.key, event.shiftKey, frameSize);
    if (next) {
      event.preventDefault();
      onChange(withBoxReplaced(boxes, index, next));
    }
  }

  return (
    <div
      role="group"
      aria-label={t(SURFACE_LABELS[kind])}
      aria-disabled={disabled}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={() => setDraft(null)}
      className={`absolute inset-0 touch-none ${disabled ? "cursor-not-allowed" : "cursor-crosshair"}`}
    >
      {/* La clave es la posicion: con las coordenadas, cada flecha remontaria el boton y perderia el foco. */}
      {boxes.map((box, index) => (
        <button
          key={index}
          type="button"
          disabled={disabled}
          aria-label={t("cctv.box.label", { index: labels?.[index] ?? index + 1, x: box[0], y: box[1], w: box[2], h: box[3] })}
          title={t("cctv.box.keyboard")}
          onKeyDown={(event) => handleBoxKey(event, index)}
          style={boxPercentStyle(box, frameSize)}
          className={boxClassName(flagged.includes(index))}
        >
          <span aria-hidden="true" className="absolute -top-5 left-0 rounded-sm bg-surface px-1 font-mono-tabular text-xs text-text">
            {labels?.[index] ?? index + 1}
          </span>
        </button>
      ))}
      {draftBox && (
        <div aria-hidden="true" style={boxPercentStyle(draftBox, frameSize)} className="pointer-events-none absolute border-2 border-dashed border-text" />
      )}
      <SizeReading box={readingBox(draftBox, kind, boxes)} kind={kind} />
    </div>
  );
}
