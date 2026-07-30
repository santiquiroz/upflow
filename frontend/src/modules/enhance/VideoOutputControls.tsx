import { AlertTriangle } from "lucide-react";
import { useTranslation } from "../../i18n/LocaleProvider";
import {
  TARGET_HEIGHT_PRESETS,
  buildVideoOutputWarnings,
  outputDimensions,
  type VideoDimensions,
  type VideoOutputMode,
} from "./videoOutputResolution";

interface VideoOutputControlsProps {
  mode: VideoOutputMode;
  onModeChange: (mode: VideoOutputMode) => void;
  targetHeight: number;
  onTargetHeightChange: (height: number) => void;
  scale: number | null;
  sourceDimensions: VideoDimensions | null;
}

function optionClassName(selected: boolean, sourceAlreadyReaches: boolean): string {
  const base =
    "flex cursor-pointer flex-col gap-1 rounded border px-3 py-2 transition-[background-color,border-color] duration-fast focus-within:outline focus-within:outline-2 focus-within:outline-accent";
  if (selected) {
    return `${base} border-accent bg-surface-2`;
  }
  if (sourceAlreadyReaches) {
    return `${base} border-warn bg-surface text-text-dim`;
  }
  return `${base} border-border bg-surface hover:border-text-faint`;
}

export function VideoOutputControls({
  mode,
  onModeChange,
  targetHeight,
  onTargetHeightChange,
  scale,
  sourceDimensions,
}: VideoOutputControlsProps) {
  const { t } = useTranslation();
  const expected = outputDimensions(sourceDimensions, mode, scale, targetHeight);
  const warnings = buildVideoOutputWarnings(sourceDimensions, mode, scale);

  return (
    <section className="flex flex-col gap-4 rounded border border-border bg-surface px-4 py-4">
      <fieldset className="flex flex-col gap-2">
        <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
          {t("video.output.mode.label")}
        </legend>
        <div className="grid gap-2 sm:grid-cols-2">
          {(["resolution", "multiplier"] as const).map((option) => (
            <label
              key={option}
              className={optionClassName(mode === option, false)}
            >
              <span className="flex items-center gap-2 text-sm text-text">
                <input
                  type="radio"
                  name="video-output-mode"
                  value={option}
                  checked={mode === option}
                  onChange={() => onModeChange(option)}
                  className="h-3.5 w-3.5 accent-accent"
                />
                {t(`video.output.mode.${option}`)}
              </span>
              <span className="pl-[22px] text-xs text-text-dim">
                {t(`video.output.mode.${option}.description`)}
              </span>
            </label>
          ))}
        </div>
      </fieldset>

      {mode === "resolution" && (
        <fieldset className="flex flex-col gap-2">
          <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
            {t("video.output.target.label")}
          </legend>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
            {TARGET_HEIGHT_PRESETS.map((height) => {
              const sourceAlreadyReaches =
                sourceDimensions !== null && sourceDimensions.height >= height;
              return (
                <label
                  key={height}
                  className={optionClassName(targetHeight === height, sourceAlreadyReaches)}
                >
                  <span className="flex items-center gap-2 text-sm text-text">
                    <input
                      type="radio"
                      name="video-target-height"
                      value={height}
                      checked={targetHeight === height}
                      onChange={() => onTargetHeightChange(height)}
                      className="h-3.5 w-3.5 accent-accent"
                    />
                    <span className="font-mono-tabular">{height}p</span>
                  </span>
                  {sourceAlreadyReaches && (
                    <span className="pl-[22px] text-[10px] leading-tight text-warn">
                      {t("video.output.target.sourceAlreadyReaches")}
                    </span>
                  )}
                </label>
              );
            })}
          </div>
        </fieldset>
      )}

      <div className="rounded border border-border bg-surface-2 px-3 py-2">
        <p className="text-xs text-text-dim">{t("video.output.preview.label")}</p>
        <p className="font-mono-tabular mt-1 text-sm text-text">
          {expected
            ? t("video.output.preview.dimensions", {
                width: expected.width,
                height: expected.height,
              })
            : mode === "resolution"
              ? t("video.output.preview.targetOnly", { height: targetHeight })
              : scale === null
                ? t("video.output.preview.unavailable")
                : t("video.output.preview.scaleOnly", { scale })}
        </p>
      </div>

      {warnings.length > 0 && (
        <ul aria-label={t("video.output.warnings.label")} className="flex flex-col gap-2">
          {warnings.map((warning) => (
            <li
              key={warning.code}
              className="flex items-start gap-2 rounded border border-warn bg-surface-2 px-3 py-2 text-sm text-warn"
            >
              <AlertTriangle
                aria-hidden="true"
                className="mt-0.5 h-4 w-4 shrink-0"
                strokeWidth={1.75}
              />
              <span>{t(warning.key, warning.params)}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
