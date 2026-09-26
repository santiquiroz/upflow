import { ScanSearch } from "lucide-react";
import { useState, type PointerEvent } from "react";
import { CompareSlider } from "../../../components/CompareSlider";
import { JobCard } from "../../../components/JobCard";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { JobResponse } from "../../../lib/apiTypes";
import type { CropBox } from "../../../lib/restoreApiTypes";
import { restoreArtifactUrl } from "../../../services/restore";
import type { Point, Size } from "./geometryMath";
import { DEFAULT_PREVIEW_BEFORE, type PreviewBeforeDeps } from "./previewBefore";
import {
  beforeAtFullResolution,
  centeredPreviewCrop,
  cropFraction,
  previewCropFromPoints,
  readPreviewResult,
  withPreviewCrop,
} from "./previewCrop";
import { usePreviewBefore } from "./usePreviewBefore";
import { useRestoreJob, type RestoreJobDeps, type RestoreJobRequest } from "./useRestoreJob";

export interface PreviewCropToolProps {
  previewUrl: string;
  alt: string;
  originalName: string;
  workingSize: Size;
  canRun: boolean;
  // Lo que se mandaria ahora: si cambia despues de probar, la prueba quedo vieja.
  settingsKey: string;
  buildRequest: () => Promise<RestoreJobRequest | null>;
  jobDeps?: RestoreJobDeps;
  beforeDeps?: PreviewBeforeDeps;
}

const TOOL_BUTTON =
  "inline-flex w-fit items-center gap-1.5 rounded-sm border border-border bg-surface px-3 py-1.5 text-sm text-text transition-[border-color,opacity] duration-fast hover:border-accent disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";
const PRIMARY_BUTTON =
  "rounded-sm bg-accent px-3 py-1.5 text-sm font-medium text-bg transition-[background-color,opacity] duration-fast hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";

function percent(value: number): string {
  return `${value * 100}%`;
}

function pointOf(event: PointerEvent<HTMLElement>): Point {
  return { x: event.clientX, y: event.clientY };
}

function isActive(phase: string): boolean {
  return phase === "queued" || phase === "running";
}

interface AreaStageProps {
  previewUrl: string;
  alt: string;
  workingSize: Size;
  crop: CropBox;
  onCrop: (crop: CropBox) => void;
}

function AreaStage({ previewUrl, alt, workingSize, crop, onCrop }: AreaStageProps) {
  const [start, setStart] = useState<Point | null>(null);
  const shown = cropFraction(crop, workingSize);

  function select(event: PointerEvent<HTMLDivElement>, from: Point) {
    const next = previewCropFromPoints(from, pointOf(event), event.currentTarget.getBoundingClientRect(), workingSize);
    if (next !== null) {
      onCrop(next);
    }
  }

  function handleDown(event: PointerEvent<HTMLDivElement>) {
    event.currentTarget.setPointerCapture?.(event.pointerId);
    setStart(pointOf(event));
    select(event, pointOf(event));
  }

  return (
    <div className="relative mx-auto w-fit max-w-full overflow-hidden rounded border border-border bg-surface">
      <img src={previewUrl} alt={alt} className="block max-h-[480px] max-w-full select-none" draggable={false} />
      <div
        data-testid="restore-preview-area-overlay"
        onPointerDown={handleDown}
        onPointerMove={(event) => start && select(event, start)}
        onPointerUp={() => setStart(null)}
        className="absolute inset-0 cursor-crosshair touch-none"
      >
        <div
          className="absolute border-2 border-accent shadow-[0_0_0_9999px_rgb(0_0_0/0.45)]"
          style={{ left: percent(shown.x), top: percent(shown.y), width: percent(shown.width), height: percent(shown.height) }}
        />
      </div>
    </div>
  );
}

interface AreaResultProps {
  job: JobResponse;
  previewUrl: string;
  originalName: string;
  workingSize: Size;
  stale: boolean;
  beforeDeps: PreviewBeforeDeps;
}

function AreaResult({ job, previewUrl, originalName, workingSize, stale, beforeDeps }: AreaResultProps) {
  const { t } = useTranslation();
  const result = readPreviewResult(job);
  const beforeUrl = usePreviewBefore(previewUrl, result?.crop ?? null, workingSize, beforeDeps);
  if (result === null) {
    return null;
  }
  const afterUrl = restoreArtifactUrl(job.jobId, "preview");
  const afterAlt = t("restore.previewArea.afterAlt", { name: originalName });
  return (
    <div className="flex flex-col gap-2">
      {beforeUrl === null ? (
        <img src={afterUrl} alt={afterAlt} className="mx-auto block max-h-[480px] max-w-full rounded border border-border" />
      ) : (
        <CompareSlider
          beforeSrc={beforeUrl}
          afterSrc={afterUrl}
          beforeAlt={t("restore.previewArea.beforeAlt", { name: originalName })}
          afterAlt={afterAlt}
          fullResolution={beforeAtFullResolution(workingSize)}
        />
      )}
      {stale && <p className="text-xs text-text-dim">{t("restore.previewArea.stale")}</p>}
    </div>
  );
}

export function PreviewCropTool({
  previewUrl,
  alt,
  originalName,
  workingSize,
  canRun,
  settingsKey,
  buildRequest,
  jobDeps,
  beforeDeps = DEFAULT_PREVIEW_BEFORE,
}: PreviewCropToolProps) {
  const { t } = useTranslation();
  const previewJob = useRestoreJob(jobDeps);
  const [open, setOpen] = useState(false);
  const [crop, setCrop] = useState<CropBox>(() => centeredPreviewCrop(workingSize));
  const [preparing, setPreparing] = useState(false);
  const [ranWith, setRanWith] = useState<string | null>(null);
  const busy = preparing || isActive(previewJob.phase);

  async function run() {
    setPreparing(true);
    const request = await buildRequest();
    setPreparing(false);
    if (request === null) {
      return;
    }
    setRanWith(settingsKey);
    previewJob.submit({
      params: { ...request.params, options: withPreviewCrop(request.params.options, crop) },
      fileName: t("restore.previewArea.jobName", { name: originalName }),
    });
  }

  return (
    <section aria-labelledby="restore-preview-area-title" className="flex flex-col gap-3">
      <div>
        <h2 id="restore-preview-area-title" className="text-sm font-medium text-text">
          {t("restore.previewArea.title")}
        </h2>
        <p className="text-xs text-text-dim">{t("restore.previewArea.hint")}</p>
      </div>
      {open ? (
        <>
          <AreaStage previewUrl={previewUrl} alt={alt} workingSize={workingSize} crop={crop} onCrop={setCrop} />
          <div className="flex flex-wrap items-center gap-3">
            <p className="text-sm text-text-dim">{t("restore.previewArea.pick")}</p>
            <span className="font-mono-tabular text-sm text-text-dim">
              {t("restore.previewArea.size", { width: crop[2], height: crop[3] })}
            </span>
            <button type="button" className={PRIMARY_BUTTON} disabled={!canRun || busy} onClick={() => void run()}>
              {t("restore.previewArea.run")}
            </button>
            <button type="button" className={TOOL_BUTTON} onClick={() => setOpen(false)}>
              {t("restore.previewArea.close")}
            </button>
          </div>
        </>
      ) : (
        <button type="button" className={TOOL_BUTTON} onClick={() => setOpen(true)}>
          <ScanSearch aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
          {t("restore.previewArea.choose")}
        </button>
      )}
      {(previewJob.phase !== "idle" || previewJob.errorMessage) && (
        <JobCard
          phase={previewJob.phase}
          job={previewJob.job}
          fileName={t("restore.previewArea.jobName", { name: originalName })}
          errorMessage={previewJob.errorMessage}
          onCancel={previewJob.cancel}
        />
      )}
      {previewJob.job && (
        <AreaResult
          key={previewJob.job.jobId}
          job={previewJob.job}
          previewUrl={previewUrl}
          originalName={originalName}
          workingSize={workingSize}
          stale={ranWith !== null && ranWith !== settingsKey}
          beforeDeps={beforeDeps}
        />
      )}
    </section>
  );
}
