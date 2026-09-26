import { ScanFace, TriangleAlert } from "lucide-react";
import { useId, useState } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { RestoreFace } from "../../../lib/restoreApiTypes";
import {
  bySize,
  effectiveFaceBlend,
  faceTier,
  isFaceSelected,
  isSelectable,
  needsConfirmation,
  tierLabelKey,
  type FaceTier,
} from "./faceSelection";
import { SliderField } from "./RestoreStepControls";
import type { SliderControl } from "./restoreSteps";
import type { FacePatch } from "./restoreSessionState";
import { versionedUrl } from "./restoreUrls";

export interface FaceGridProps {
  faces: RestoreFace[];
  proposed: RestoreFace[];
  stepBlend: number;
  imageRevision: number;
  onChange: (index: number, patch: FacePatch) => void;
}

const PERCENT = 100;
const FACE_BLEND: SliderControl = {
  kind: "slider",
  option: "blend",
  labelKey: "restore.control.faceBlend",
  min: 0,
  max: 1,
  step: 0.05,
  format: "percent",
};

function faceNumber(face: RestoreFace): number {
  return face.index + 1;
}

function FaceThumbnail({ face, imageRevision }: { face: RestoreFace; imageRevision: number }) {
  const { t } = useTranslation();
  const name = t("restore.face.name", { number: faceNumber(face) });
  if (face.thumbnailUrl === null) {
    return (
      <div role="img" aria-label={name} className="flex aspect-square w-full items-center justify-center rounded-sm bg-surface">
        <ScanFace aria-hidden="true" className="h-6 w-6 text-text-faint" strokeWidth={1.5} />
      </div>
    );
  }
  return (
    <img
      src={versionedUrl(face.thumbnailUrl, imageRevision)}
      alt={name}
      className="aspect-square w-full rounded-sm bg-surface object-cover"
    />
  );
}

function FaceMeasure({ face }: { face: RestoreFace }) {
  const { t } = useTranslation();
  if (face.eyePx === null) {
    return null;
  }
  return (
    <p className="font-mono-tabular text-[10px] text-text-faint">
      {t("restore.face.measure", {
        px: Math.round(face.eyePx),
        confidence: Math.round((face.confidence ?? 0) * PERCENT),
      })}
    </p>
  );
}

function TierLabel({ tier }: { tier: FaceTier }) {
  const { t } = useTranslation();
  const cautious = tier !== "restore";
  return <p className={`text-xs leading-snug ${cautious ? "text-warn" : "text-text-dim"}`}>{t(tierLabelKey(tier))}</p>;
}

interface ConfirmSmallFaceProps {
  onAccept: () => void;
  onCancel: () => void;
}

function ConfirmSmallFace({ onAccept, onCancel }: ConfirmSmallFaceProps) {
  const { t } = useTranslation();
  return (
    <div role="group" className="flex flex-col gap-2 rounded-sm border border-warn bg-surface p-2">
      <p className="text-xs leading-snug text-text">{t("restore.face.confirmSmall")}</p>
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={onAccept}
          className="rounded border border-warn px-2 py-1 text-xs text-warn transition hover:bg-warn hover:text-bg focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          {t("restore.face.confirmSmall.accept")}
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="rounded border border-border px-2 py-1 text-xs text-text-dim transition hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          {t("common.cancel")}
        </button>
      </div>
    </div>
  );
}

interface FaceTileProps {
  face: RestoreFace;
  proposed: RestoreFace | undefined;
  stepBlend: number;
  imageRevision: number;
  onChange: (index: number, patch: FacePatch) => void;
}

function FaceTile({ face, proposed, stepBlend, imageRevision, onChange }: FaceTileProps) {
  const { t } = useTranslation();
  const checkboxId = useId();
  const [confirming, setConfirming] = useState(false);
  const tier = faceTier(face);
  const selected = isFaceSelected(face);
  const number = faceNumber(face);

  function handleToggle(checked: boolean) {
    if (checked && needsConfirmation(tier)) {
      setConfirming(true);
      return;
    }
    onChange(face.index, { enabled: checked });
  }

  function acceptSmallFace() {
    setConfirming(false);
    onChange(face.index, { enabled: true });
  }

  return (
    <li
      aria-label={t("restore.face.name", { number })}
      className={`flex flex-col gap-2 rounded border p-2 transition-[border-color,opacity] duration-fast motion-reduce:transition-none ${
        selected ? "border-accent bg-surface-2" : "border-border bg-surface opacity-90"
      }`}
    >
      <FaceThumbnail face={face} imageRevision={imageRevision} />
      <label htmlFor={checkboxId} className="flex items-center gap-2 text-xs font-medium text-text">
        <input
          id={checkboxId}
          type="checkbox"
          checked={selected}
          disabled={!isSelectable(tier)}
          onChange={(event) => handleToggle(event.target.checked)}
          className="h-3.5 w-3.5 shrink-0 accent-accent enabled:cursor-pointer disabled:cursor-not-allowed"
        />
        {t("restore.face.select", { number })}
      </label>
      <TierLabel tier={tier} />
      <FaceMeasure face={face} />
      {confirming && <ConfirmSmallFace onAccept={acceptSmallFace} onCancel={() => setConfirming(false)} />}
      {selected && (
        <SliderField
          control={FACE_BLEND}
          value={effectiveFaceBlend(face, proposed, stepBlend)}
          onChange={(value) => onChange(face.index, { blend: Number(value) })}
        />
      )}
    </li>
  );
}

export function FaceGrid({ faces, proposed, stepBlend, imageRevision, onChange }: FaceGridProps) {
  const { t } = useTranslation();
  const titleId = useId();
  const noneSelected = !faces.some(isFaceSelected);

  return (
    <section aria-labelledby={titleId} className="flex flex-col gap-3">
      <h3 id={titleId} className="font-heading text-sm font-semibold text-text">
        {t("restore.faces.title")}
      </h3>
      <p className="flex items-start gap-2 text-xs leading-relaxed text-warn">
        <TriangleAlert aria-hidden="true" className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
        <span>{t("restore.warning.faces")}</span>
      </p>
      <p className="text-xs text-text-faint">{t("restore.faces.hint")}</p>
      <ul className="grid grid-cols-[repeat(auto-fill,minmax(9rem,1fr))] gap-2">
        {bySize(faces).map((face) => (
          <FaceTile
            key={face.index}
            face={face}
            proposed={proposed.find((candidate) => candidate.index === face.index)}
            stepBlend={stepBlend}
            imageRevision={imageRevision}
            onChange={onChange}
          />
        ))}
      </ul>
      {noneSelected && (
        <p role="status" className="text-xs text-text-dim">
          {t("restore.faces.noneSelected")}
        </p>
      )}
    </section>
  );
}
