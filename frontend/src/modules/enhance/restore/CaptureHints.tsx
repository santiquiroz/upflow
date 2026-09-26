import { LayoutGrid, Move3d, ScanLine, type LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { RestoreCapture, RestoreGeometry } from "../../../lib/restoreApiTypes";
import { sameGeometry, type Point, type Size } from "./geometryMath";
import { suggestionOutline } from "./perspectiveMath";

interface CaptureHintsProps {
  capture: RestoreCapture;
  geometry: RestoreGeometry;
  busy: boolean;
  canAdjustCorners: boolean;
  onApply: (geometry: RestoreGeometry) => void;
  onAdjustCorners: () => void;
}

const HINT_BUTTON =
  "inline-flex items-center rounded-sm border border-border bg-surface px-2.5 py-1 text-xs font-medium text-text transition-[border-color,background-color,opacity] duration-fast hover:border-accent disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent aria-pressed:border-accent aria-pressed:bg-accent aria-pressed:text-bg";

function HintRow({ icon: Icon, text, children }: { icon: LucideIcon; text: string; children: ReactNode }) {
  return (
    <li className="flex flex-wrap items-center gap-x-3 gap-y-2">
      <span className="flex min-w-0 flex-1 basis-56 items-start gap-2 text-xs text-text-dim">
        <Icon aria-hidden="true" className="mt-px h-4 w-4 shrink-0 text-accent" strokeWidth={1.75} />
        {text}
      </span>
      <span className="flex flex-wrap gap-1.5">{children}</span>
    </li>
  );
}

function SuggestionButton({
  suggestion,
  geometry,
  busy,
  label,
  onApply,
}: {
  suggestion: RestoreGeometry;
  geometry: RestoreGeometry;
  busy: boolean;
  label: string;
  onApply: (geometry: RestoreGeometry) => void;
}) {
  const applied = sameGeometry(suggestion, geometry);
  return (
    <button type="button" className={HINT_BUTTON} disabled={busy} aria-pressed={applied} onClick={() => !applied && onApply(suggestion)}>
      {label}
    </button>
  );
}

export function hasCaptureHints(capture: RestoreCapture | undefined): capture is RestoreCapture {
  return Boolean(capture && (capture.autoCrop || capture.photos.length > 1 || capture.perspective));
}

export function CaptureHints({ capture, geometry, busy, canAdjustCorners, onApply, onAdjustCorners }: CaptureHintsProps) {
  const { t } = useTranslation();
  const { autoCrop, photos, perspective } = capture;
  return (
    <ul aria-label={t("restore.capture.title")} className="flex flex-col gap-2 rounded border border-border bg-surface-2 px-3 py-2.5">
      {autoCrop && (
        <HintRow icon={ScanLine} text={t("restore.capture.autoCropHint")}>
          <SuggestionButton suggestion={autoCrop} geometry={geometry} busy={busy} label={t("restore.capture.autoCrop")} onApply={onApply} />
        </HintRow>
      )}
      {photos.length > 1 && (
        <HintRow icon={LayoutGrid} text={t("restore.capture.photosHint", { count: photos.length })}>
          {photos.map((photo, index) => (
            <SuggestionButton
              key={index}
              suggestion={photo}
              geometry={geometry}
              busy={busy}
              label={t("restore.capture.photo", { number: index + 1 })}
              onApply={onApply}
            />
          ))}
        </HintRow>
      )}
      {perspective && (
        <HintRow icon={Move3d} text={t("restore.capture.perspectiveHint")}>
          <SuggestionButton
            suggestion={perspective}
            geometry={geometry}
            busy={busy}
            label={t("restore.capture.fixPerspective")}
            onApply={onApply}
          />
          {canAdjustCorners && (
            <button type="button" className={HINT_BUTTON} disabled={busy} onClick={onAdjustCorners}>
              {t("restore.capture.adjustCorners")}
            </button>
          )}
        </HintRow>
      )}
    </ul>
  );
}

function centroid(points: Point[]): Point {
  const total = points.reduce((sum, point) => ({ x: sum.x + point.x, y: sum.y + point.y }), { x: 0, y: 0 });
  return { x: total.x / points.length, y: total.y / points.length };
}

function outlinePoints(points: Point[]): string {
  return points.map((point) => `${point.x * 100},${point.y * 100}`).join(" ");
}

export function PhotoOutlines({
  photos,
  geometry,
  working,
  frame,
}: {
  photos: RestoreGeometry[];
  geometry: RestoreGeometry;
  working: Size;
  frame: Size;
}) {
  const outlines = photos.map((photo) => suggestionOutline(photo, geometry, working, frame));
  if (photos.length < 2 || outlines.some((outline) => outline === null)) {
    return null;
  }
  return (
    <div aria-hidden="true" data-testid="restore-photo-outlines" className="pointer-events-none absolute inset-0">
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="absolute inset-0 h-full w-full">
        {outlines.map((outline, index) => (
          <polygon
            key={index}
            points={outlinePoints(outline!)}
            fill="none"
            stroke="var(--accent)"
            strokeWidth={2}
            strokeDasharray="6 4"
            vectorEffect="non-scaling-stroke"
          />
        ))}
      </svg>
      {outlines.map((outline, index) => {
        const center = centroid(outline!);
        return (
          <span
            key={index}
            className="absolute flex h-6 w-6 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-accent font-mono-tabular text-xs font-semibold text-bg shadow"
            style={{ left: `${center.x * 100}%`, top: `${center.y * 100}%` }}
          >
            {index + 1}
          </span>
        );
      })}
    </div>
  );
}
