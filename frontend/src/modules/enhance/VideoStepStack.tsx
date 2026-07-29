import { Plus, X } from "lucide-react";
import { useId } from "react";
import type { TranslationParams } from "../../i18n";
import { useTranslation } from "../../i18n/LocaleProvider";
import { denoiseLabel, restoreLabel } from "../../lib/audioLabels";
import type { VideoStep, VideoStepId } from "./videoSteps";

interface VideoStepStackProps {
  steps: VideoStep[];
  addableStepIds: VideoStepId[];
  onRemove: (stepId: VideoStepId) => void;
  onAdd: (stepId: VideoStepId) => void;
}

const STEP_LABEL_KEYS: Record<VideoStepId, string> = {
  upscale: "video.steps.upscale.label",
  interpolate: "video.steps.interpolate.label",
  audio: "video.steps.audio.label",
  subtitles: "video.steps.subtitles.label",
};

type Translate = (key: string, params?: TranslationParams) => string;

function stepLabel(stepId: VideoStepId, t: Translate): string {
  return t(STEP_LABEL_KEYS[stepId]);
}

function formatTargetFps(targetFps: string): string {
  if (targetFps === "60000/1001") {
    return "59.94 fps";
  }
  if (targetFps === "60/1") {
    return "60 fps";
  }
  return targetFps;
}

function stepDescription(step: VideoStep, t: Translate): string {
  switch (step.id) {
    case "upscale":
      return t("video.steps.upscale.description", {
        model: step.modelName ?? t("video.steps.modelUnknown"),
        scale: step.scale,
      });
    case "interpolate":
      if (step.targetFps !== null) {
        return t("video.steps.interpolate.targetDescription", {
          engine: step.interpEngine.toUpperCase(),
          target: formatTargetFps(step.targetFps),
        });
      }
      return t("video.steps.interpolate.multiplierDescription", {
        engine: step.interpEngine.toUpperCase(),
        multiplier: step.fpsMultiplier,
      });
    case "audio": {
      const modes = [
        step.audioEnhance === null ? null : denoiseLabel(step.audioEnhance),
        step.audioRestore === null ? null : restoreLabel(step.audioRestore),
      ].filter((mode): mode is string => mode !== null);
      return t("video.steps.audio.description", { modes: modes.join(" + ") });
    }
    case "subtitles":
      return t("video.steps.subtitles.description");
  }
}

export function VideoStepStack({
  steps,
  addableStepIds,
  onRemove,
  onAdd,
}: VideoStepStackProps) {
  const { t } = useTranslation();
  const headingId = useId();

  return (
    <section
      aria-labelledby={headingId}
      className="flex flex-col gap-4 rounded border border-border bg-surface px-4 py-4"
    >
      <div className="flex flex-col gap-1">
        <h3 id={headingId} className="font-heading text-sm font-semibold text-text">
          {t("video.steps.title")}
        </h3>
        <p className="text-xs leading-relaxed text-text-dim">
          {t("video.steps.description")}
        </p>
      </div>

      {steps.length > 0 ? (
        <ol aria-label={t("video.steps.listLabel")} className="flex flex-col">
          {steps.map((step, index) => {
            const label = stepLabel(step.id, t);
            const isLast = index === steps.length - 1;
            return (
              <li key={step.id} className="flex gap-3">
                <div className="flex flex-col items-center gap-1 pt-2.5">
                  <span
                    aria-hidden="true"
                    className="font-mono-tabular flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-accent bg-accent text-[10px] text-bg"
                  >
                    {index + 1}
                  </span>
                  {!isLast && <span aria-hidden="true" className="w-px flex-1 bg-border" />}
                </div>

                <div className={`flex-1 ${isLast ? "" : "pb-3"}`}>
                  <div className="flex min-h-16 items-start gap-3 rounded border border-border bg-surface-2 px-3 py-2.5">
                    <div className="flex min-w-0 flex-1 flex-col gap-1.5">
                      <span className="text-sm font-medium text-text">{label}</span>
                      <p className="text-xs leading-relaxed text-text-dim">
                        {stepDescription(step, t)}
                      </p>
                    </div>
                    <button
                      type="button"
                      aria-label={t("video.steps.removeStep", { step: label })}
                      title={t("video.steps.removeStep", { step: label })}
                      onClick={() => onRemove(step.id)}
                      className="rounded-sm p-1 text-text-faint transition-colors duration-fast hover:bg-surface hover:text-danger focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
                    >
                      <X aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
                    </button>
                  </div>
                </div>
              </li>
            );
          })}
        </ol>
      ) : (
        <p className="rounded border border-dashed border-border px-3 py-3 text-xs text-text-faint">
          {t("video.steps.empty")}
        </p>
      )}

      {addableStepIds.length > 0 && (
        <div className="flex flex-col gap-2">
          <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
            {t("video.steps.add")}
          </span>
          <div
            role="group"
            aria-label={t("video.steps.add")}
            className="flex flex-wrap gap-2"
          >
            {addableStepIds.map((stepId) => {
              const label = stepLabel(stepId, t);
              return (
                <button
                  key={stepId}
                  type="button"
                  aria-label={t("video.steps.addStep", { step: label })}
                  onClick={() => onAdd(stepId)}
                  className="inline-flex items-center gap-1.5 rounded-sm border border-border bg-surface-2 px-2.5 py-1.5 text-xs text-text-dim transition-[border-color,color] duration-fast hover:border-accent hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
                >
                  <Plus aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
                  {label}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </section>
  );
}
