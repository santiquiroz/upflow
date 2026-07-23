import { AlertTriangle, Ban, CheckCircle2, Circle, Loader2 } from "lucide-react";
import { Fragment, useEffect, useState } from "react";
import type { JobQueueEntry } from "../hooks/useJobQueue";
import type { AudioJob, GenerationJob, JobResponse, JobStage, VideoJobResponse } from "../lib/apiTypes";
import { denoiseLabel, restoreLabel } from "../lib/audioLabels";
import { estimateEta, formatEta, type EtaSample } from "../lib/eta";
import { formatDuration } from "../lib/formatDuration";
import { formatFps } from "../lib/formatFps";
import {
  areFramesReportable,
  deriveStepper,
  isProgressDeterminate,
  resolveFramesDenominator,
  toMonotonicProgressPct,
} from "../lib/jobProgress";
import { isCancellableJobStatus, jobKindLabel } from "../lib/jobStatus";
import { DeterminateProgressBar } from "./DeterminateProgressBar";
import { IndeterminateProgressBar } from "./IndeterminateProgressBar";
import { Modal } from "./Modal";

interface JobDetailModalProps {
  entry: JobQueueEntry;
  onClose: () => void;
  onCancel?: (id: string) => void;
}

type AnyJobResponse = JobResponse | VideoJobResponse | AudioJob | GenerationJob;

const MAX_ETA_SAMPLES = 5;

const AUDIO_ENHANCE_LABELS: Record<string, string> = {
  rnnoise: "RNNoise",
  deepfilter: "DeepFilterNet",
};

function isVideoJob(job: AnyJobResponse): job is VideoJobResponse {
  return "videoCodec" in job;
}

function isAudioJob(job: AnyJobResponse): job is AudioJob {
  return "denoise" in job;
}

function isGenerationJob(job: AnyJobResponse): job is GenerationJob {
  return "prompt" in job;
}

// Audio and generation jobs carry stages at the top level (no `metadata`),
// image/video jobs nest them under metadata -- normalize both to the same list.
function resolveStages(job: AnyJobResponse | undefined): JobStage[] | undefined {
  if (!job) {
    return undefined;
  }
  if (isAudioJob(job) || isGenerationJob(job)) {
    return job.stages ?? undefined;
  }
  return job.metadata.stages;
}

function titleIdFor(jobId: string): string {
  return `job-detail-title-${jobId}`;
}

function readAudioLabel(job: VideoJobResponse): string {
  if (!job.keepAudio) {
    return "Disabled";
  }
  if (job.audioEnhance) {
    return AUDIO_ENHANCE_LABELS[job.audioEnhance] ?? job.audioEnhance;
  }
  return "Kept";
}

function readFpsLabel(job: VideoJobResponse): string | null {
  const outputFps = job.metadata.outputFps;
  if (outputFps) {
    return formatFps(outputFps);
  }
  if (job.targetFps) {
    return formatFps(job.targetFps);
  }
  if (job.fpsMultiplier > 1) {
    return `${job.fpsMultiplier}x`;
  }
  return null;
}

// Progress must never appear to move backward in the UI (a stage-transition
// recompute can transiently report a lower fraction than what was already
// shown) -- this is React's documented "adjust state during render" pattern
// for resetting derived state on prop change, so it stays synchronous and
// avoids an extra render versus doing the reset in an effect.
function useMonotonicProgressPct(jobId: string, rawProgressPct: number | null): number | null {
  const [trackedJobId, setTrackedJobId] = useState(jobId);
  const [maxPct, setMaxPct] = useState<number | null>(null);

  if (jobId !== trackedJobId) {
    setTrackedJobId(jobId);
    setMaxPct(null);
    return null;
  }

  if (rawProgressPct === null) {
    return maxPct;
  }

  const nextMaxPct = toMonotonicProgressPct(maxPct ?? 0, rawProgressPct);
  if (nextMaxPct !== maxPct) {
    setMaxPct(nextMaxPct);
  }
  return nextMaxPct;
}

// Date.now() is an impure read, so unlike the monotonic-progress adjustment
// above this buffer is built in an effect rather than during render.
function useEtaSampleBuffer(jobId: string, monotonicProgressPct: number | null): EtaSample[] {
  const [state, setState] = useState<{ jobId: string; samples: EtaSample[] }>({ jobId, samples: [] });

  useEffect(() => {
    if (monotonicProgressPct === null) {
      return;
    }
    const progress = monotonicProgressPct / 100;
    setState((previous) => {
      const samples = previous.jobId === jobId ? previous.samples : [];
      const last = samples[samples.length - 1];
      if (last && last.progress === progress) {
        return previous.jobId === jobId ? previous : { jobId, samples };
      }
      return { jobId, samples: [...samples, { progress, t: Date.now() }].slice(-MAX_ETA_SAMPLES) };
    });
  }, [jobId, monotonicProgressPct]);

  return state.jobId === jobId ? state.samples : [];
}

interface DetailItem {
  label: string;
  value: string;
  isNumeric?: boolean;
}

function JobTypeSummary({ entry, job }: { entry: JobQueueEntry; job: AnyJobResponse | undefined }) {
  const items: DetailItem[] = [{ label: "Type", value: jobKindLabel(entry.kind) }];
  if (!job) {
    return <DetailList items={items} />;
  }
  if (isAudioJob(job)) {
    items.push({ label: "Denoise", value: denoiseLabel(job.denoise) });
    items.push({ label: "Restore", value: restoreLabel(job.restore) });
    if (job.device) {
      items.push({ label: "Device", value: job.device });
    }
    pushDurationItem(items, job);
    return <DetailList items={items} />;
  }
  if (isGenerationJob(job)) {
    items.push({ label: "Prompt", value: job.prompt });
    if (job.negativePrompt) {
      items.push({ label: "Negative prompt", value: job.negativePrompt });
    }
    items.push({ label: "Model", value: job.modelId });
    items.push({ label: "Steps", value: String(job.steps), isNumeric: true });
    items.push({ label: "Guidance", value: String(job.guidance), isNumeric: true });
    items.push({ label: "Size", value: `${job.width}x${job.height}` });
    if (job.seed !== null) {
      items.push({ label: "Seed", value: String(job.seed), isNumeric: true });
    }
    if (job.device) {
      items.push({ label: "Device", value: job.device });
    }
    pushDurationItem(items, job);
    return <DetailList items={items} />;
  }
  items.push({ label: "Model", value: job.modelName });
  if (job.device) {
    items.push({ label: "Device", value: job.device });
  }
  items.push({ label: "Scale", value: `${job.scale}x`, isNumeric: true });
  if (isVideoJob(job)) {
    items.push({ label: "Container", value: job.outputContainer });
    const fps = readFpsLabel(job);
    if (fps) {
      items.push({ label: "FPS", value: fps, isNumeric: true });
    }
    items.push({ label: "Audio", value: readAudioLabel(job) });
  } else {
    items.push({ label: "Format", value: job.outputFormat.toUpperCase() });
  }
  pushDurationItem(items, job);
  return <DetailList items={items} />;
}

// Only meaningful once the job has actually finished (completed/failed/cancelled) --
// while running, finishedAt is still null so the row is omitted.
function pushDurationItem(items: DetailItem[], job: AnyJobResponse): void {
  if (!job.finishedAt) {
    return;
  }
  items.push({ label: "Duration", value: formatDuration(job.startedAt, job.finishedAt) });
}

function detailValueClassName(isNumeric: boolean | undefined): string {
  return isNumeric ? "font-mono-tabular text-right text-text" : "text-right text-text";
}

function DetailList({ items }: { items: DetailItem[] }) {
  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs text-text-dim">
      {items.map((item) => (
        <Fragment key={item.label}>
          <dt className="text-text-faint">{item.label}</dt>
          <dd className={detailValueClassName(item.isNumeric)}>{item.value}</dd>
        </Fragment>
      ))}
    </dl>
  );
}

function StepIcon({ state }: { state: "done" | "active" | "pending" }) {
  if (state === "done") {
    return <CheckCircle2 aria-hidden="true" className="h-4 w-4 shrink-0 text-ok" strokeWidth={1.75} />;
  }
  if (state === "active") {
    return <Loader2 aria-hidden="true" className="h-4 w-4 shrink-0 animate-spin text-accent" strokeWidth={1.75} />;
  }
  return <Circle aria-hidden="true" className="h-4 w-4 shrink-0 text-text-faint" strokeWidth={1.75} />;
}

function stepTextClassName(state: "done" | "active" | "pending"): string {
  if (state === "pending") {
    return "text-text-faint";
  }
  return state === "active" ? "text-text" : "text-text-dim";
}

function Stepper({ job }: { job: AnyJobResponse | undefined }) {
  const steps = deriveStepper(resolveStages(job));
  if (steps.length === 0) {
    return null;
  }
  return (
    <ol className="flex flex-col gap-2">
      {steps.map((step) => (
        <li key={step.key} className="flex items-center gap-2 text-xs">
          <StepIcon state={step.iconState} />
          <span className={stepTextClassName(step.iconState)}>{step.label}</span>
        </li>
      ))}
    </ol>
  );
}

function ProgressSection({ job, monotonicProgressPct }: { job: AnyJobResponse | undefined; monotonicProgressPct: number | null }) {
  const label = "Progress";
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between text-xs text-text-dim">
        <span>{label}</span>
        {isProgressDeterminate(monotonicProgressPct) && (
          <span className="font-mono-tabular text-text">{Math.round(monotonicProgressPct)}%</span>
        )}
      </div>
      {isProgressDeterminate(monotonicProgressPct) ? (
        <DeterminateProgressBar label={label} percent={monotonicProgressPct} />
      ) : (
        <IndeterminateProgressBar label={label} />
      )}
      <FramesReadout job={job} />
    </div>
  );
}

function FramesReadout({ job }: { job: AnyJobResponse | undefined }) {
  if (!job || isAudioJob(job) || isGenerationJob(job)) {
    return null;
  }
  const framesDone = job.metadata.framesDone;
  const framesTotal = resolveFramesDenominator(job.metadata);
  if (!areFramesReportable(framesDone, framesTotal)) {
    return null;
  }
  return (
    <p className="text-xs text-text-dim">
      <span className="font-mono-tabular">{framesDone}</span>
      {" / "}
      <span className="font-mono-tabular">{framesTotal}</span>
      {" frames"}
    </p>
  );
}

function EtaReadout({ samples }: { samples: EtaSample[] }) {
  const etaSeconds = estimateEta(samples);
  if (etaSeconds === null) {
    return null;
  }
  return <p className="text-xs text-text-dim">ETA {formatEta(etaSeconds)}</p>;
}

function ErrorNotice({ message }: { message: string }) {
  return (
    <div role="alert" className="flex items-start gap-2 rounded border border-danger bg-surface-2 px-3 py-2 text-sm text-danger">
      <AlertTriangle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" strokeWidth={1.75} />
      <span>{message}</span>
    </div>
  );
}

function CancelledNotice() {
  return (
    <div className="flex items-center gap-2 rounded border border-border bg-surface-2 px-3 py-2 text-sm text-text-dim">
      <Ban aria-hidden="true" className="h-4 w-4 shrink-0" strokeWidth={1.75} />
      <span>Cancelled</span>
    </div>
  );
}

function ModalActions({ entry, onClose, onCancel }: JobDetailModalProps) {
  return (
    <div className="ml-auto flex w-fit gap-2">
      {isCancellableJobStatus(entry.status) && onCancel && (
        <button
          type="button"
          onClick={() => onCancel(entry.id)}
          className="inline-flex items-center gap-1.5 rounded-sm border border-danger px-3 py-1.5 text-sm text-danger transition-[background-color,color] duration-fast hover:bg-danger hover:text-bg focus-visible:outline focus-visible:outline-2 focus-visible:outline-danger"
        >
          <Ban aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
          Cancel
        </button>
      )}
      <button
        type="button"
        onClick={onClose}
        className="rounded-sm border border-border bg-surface px-3 py-1.5 text-sm text-text-dim transition-[border-color,color] duration-fast hover:border-text-faint hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        Close
      </button>
    </div>
  );
}

function showsProgress(status: JobQueueEntry["status"]): boolean {
  return status !== "failed" && status !== "cancelled";
}

export function JobDetailModal({ entry, onClose, onCancel }: JobDetailModalProps) {
  const titleId = titleIdFor(entry.id);
  const rawProgressPct = entry.job?.progressPct ?? null;
  const monotonicProgressPct = useMonotonicProgressPct(entry.id, rawProgressPct);
  const etaSamples = useEtaSampleBuffer(entry.id, monotonicProgressPct);

  return (
    <Modal titleId={titleId} onClose={onClose}>
      <h2 id={titleId} className="truncate font-heading text-sm font-semibold text-text" title={entry.fileName}>
        {entry.fileName}
      </h2>
      <JobTypeSummary entry={entry} job={entry.job} />
      <Stepper job={entry.job} />
      {showsProgress(entry.status) && <ProgressSection job={entry.job} monotonicProgressPct={monotonicProgressPct} />}
      {entry.status === "running" && <EtaReadout samples={etaSamples} />}
      {entry.status === "cancelled" && <CancelledNotice />}
      {entry.errorMessage && <ErrorNotice message={entry.errorMessage} />}
      <ModalActions entry={entry} onClose={onClose} onCancel={onCancel} />
    </Modal>
  );
}
