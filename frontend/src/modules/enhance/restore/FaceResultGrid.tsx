import { Check, RefreshCw } from "lucide-react";
import { useId, useState } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { RecomposeFaceChoice, RecomposeResponse } from "../../../lib/restoreApiTypes";
import { recomposeFaces, restoreArtifactUrl } from "../../../services/restore";
import {
  draftOf,
  isDraftChanged,
  withFaceChoice,
  type FaceDraft,
  type RestoredFace,
} from "./faceResultModel";
import { SliderField } from "./RestoreStepControls";
import type { SliderControl } from "./restoreSteps";

type Recompose = (jobId: string, faces: Record<number, RecomposeFaceChoice>) => Promise<RecomposeResponse>;

export interface FaceResultGridProps {
  jobId: string;
  faces: RestoredFace[];
  recomposeAvailable: boolean;
  onRecomposed: (sidecar: Record<string, unknown>) => void;
  recompose?: Recompose;
}

type ApplyState = { kind: "idle" } | { kind: "applying" } | { kind: "applied" } | { kind: "failed"; message: string };

const FACE_BLEND: SliderControl = {
  kind: "slider",
  option: "blend",
  labelKey: "restore.control.faceBlend",
  min: 0,
  max: 1,
  step: 0.05,
  format: "percent",
};

function errorText(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

// La mezcla es lineal (alfa por cara), asi que la capa restaurada con esa
// opacidad sobre la original anticipa lo que va a dar "Apply".
function FaceComparison({ jobId, index, choice }: { jobId: string; index: number; choice: RecomposeFaceChoice }) {
  const { t } = useTranslation();
  const number = index + 1;
  const opacity = choice.enabled ? choice.blend : 0;
  return (
    <div className="grid grid-cols-2 gap-1.5">
      <figure className="flex flex-col gap-1">
        <img
          src={restoreArtifactUrl(jobId, `face:${index}:before`)}
          alt={t("restore.face.beforeAlt", { number })}
          className="aspect-square w-full rounded-sm bg-surface object-cover"
        />
        <figcaption className="text-[10px] text-text-faint">{t("restore.compare.before")}</figcaption>
      </figure>
      <figure className="flex flex-col gap-1">
        <div className="relative aspect-square w-full overflow-hidden rounded-sm bg-surface">
          <img
            src={restoreArtifactUrl(jobId, `face:${index}:before`)}
            alt=""
            aria-hidden="true"
            className="absolute inset-0 h-full w-full object-cover"
          />
          <img
            src={restoreArtifactUrl(jobId, `face:${index}:after`)}
            alt={t("restore.face.afterAlt", { number })}
            style={{ opacity }}
            className="absolute inset-0 h-full w-full object-cover transition-opacity duration-fast motion-reduce:transition-none"
          />
        </div>
        <figcaption className="text-[10px] text-text-faint">{t("restore.compare.after")}</figcaption>
      </figure>
    </div>
  );
}

interface FaceResultTileProps {
  jobId: string;
  index: number;
  choice: RecomposeFaceChoice;
  disabled: boolean;
  onChange: (patch: Partial<RecomposeFaceChoice>) => void;
}

function FaceResultTile({ jobId, index, choice, disabled, onChange }: FaceResultTileProps) {
  const { t } = useTranslation();
  return (
    <li
      aria-label={t("restore.face.name", { number: index + 1 })}
      className="flex flex-col gap-2 rounded border border-border bg-surface-2 p-2"
    >
      <FaceComparison jobId={jobId} index={index} choice={choice} />
      <label className="flex items-center gap-2 text-xs text-text">
        <input
          type="checkbox"
          checked={!choice.enabled}
          disabled={disabled}
          onChange={(event) => onChange({ enabled: !event.target.checked })}
          className="h-3.5 w-3.5 shrink-0 accent-accent enabled:cursor-pointer disabled:cursor-not-allowed"
        />
        {t("restore.face.showOriginal")}
      </label>
      <SliderField
        control={FACE_BLEND}
        value={choice.blend}
        disabled={disabled || !choice.enabled}
        onChange={(value) => onChange({ blend: Number(value) })}
      />
      {choice.enabled && <p className="text-xs leading-snug text-warn">{t("restore.warning.faceEach")}</p>}
    </li>
  );
}

function ApplyStatus({ state }: { state: ApplyState }) {
  const { t } = useTranslation();
  if (state.kind === "failed") {
    return (
      <p role="alert" className="text-xs text-danger">
        {state.message}
      </p>
    );
  }
  if (state.kind === "applied") {
    return (
      <p role="status" className="inline-flex items-center gap-1 text-xs text-ok">
        <Check aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
        {t("restore.faces.applied")}
      </p>
    );
  }
  return null;
}

interface ApplyBarProps {
  state: ApplyState;
  changed: boolean;
  onApply: () => void;
}

function ApplyBar({ state, changed, onApply }: ApplyBarProps) {
  const { t } = useTranslation();
  const applying = state.kind === "applying";
  return (
    <div className="flex flex-wrap items-center gap-3">
      <button
        type="button"
        onClick={onApply}
        disabled={!changed || applying}
        className="inline-flex items-center gap-1.5 rounded border border-accent px-3 py-1.5 text-xs font-medium text-accent transition hover:bg-accent hover:text-bg disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent disabled:hover:text-accent focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        <RefreshCw aria-hidden="true" className={`h-3.5 w-3.5 ${applying ? "animate-spin motion-reduce:animate-none" : ""}`} strokeWidth={1.75} />
        {t(applying ? "restore.faces.applying" : "restore.faces.apply")}
      </button>
      <ApplyStatus state={state} />
    </div>
  );
}

export function FaceResultGrid({
  jobId,
  faces,
  recomposeAvailable,
  onRecomposed,
  recompose = recomposeFaces,
}: FaceResultGridProps) {
  const { t } = useTranslation();
  const titleId = useId();
  const [draft, setDraft] = useState<FaceDraft>(() => draftOf(faces));
  const [applyState, setApplyState] = useState<ApplyState>({ kind: "idle" });
  const locked = !recomposeAvailable || applyState.kind === "applying";

  function changeFace(index: number, patch: Partial<RecomposeFaceChoice>) {
    setDraft((current) => withFaceChoice(current, index, patch));
    setApplyState({ kind: "idle" });
  }

  // Se mandan todas las caras: una que falte vuelve a su mezcla original en el backend.
  async function apply() {
    setApplyState({ kind: "applying" });
    try {
      const response = await recompose(jobId, { ...draft });
      setApplyState({ kind: "applied" });
      onRecomposed(response.sidecar);
    } catch (error) {
      setApplyState({ kind: "failed", message: errorText(error) });
    }
  }

  return (
    <section aria-labelledby={titleId} className="flex flex-col gap-3">
      <h4 id={titleId} className="font-heading text-sm font-semibold text-text">
        {t("restore.faces.result.title")}
      </h4>
      <p className="text-xs text-text-dim">
        {t(recomposeAvailable ? "restore.faces.result.hint" : "restore.faces.recomposeUnavailable")}
      </p>
      <ul className="grid grid-cols-[repeat(auto-fill,minmax(12rem,1fr))] gap-2">
        {faces.map((face) => (
          <FaceResultTile
            key={face.index}
            jobId={jobId}
            index={face.index}
            choice={draft[face.index] ?? face}
            disabled={locked}
            onChange={(patch) => changeFace(face.index, patch)}
          />
        ))}
      </ul>
      {recomposeAvailable && (
        <ApplyBar state={applyState} changed={isDraftChanged(faces, draft)} onApply={() => void apply()} />
      )}
    </section>
  );
}
