import { AlertTriangle, Ban, CheckCircle2, Clock, Download, ImageIcon, Loader2, UploadCloud } from "lucide-react";
import type { AudioJob, GenerationJob, JobResponse, VideoJobResponse } from "../lib/apiTypes";
import { denoiseLabel, restoreLabel } from "../lib/audioLabels";
import { formatDuration } from "../lib/formatDuration";
import { formatFps } from "../lib/formatFps";
import { isProgressDeterminate } from "../lib/jobProgress";
import { isGenerationJob, type AnyJobResponse } from "../lib/jobTypeGuards";
import { DeterminateProgressBar } from "./DeterminateProgressBar";
import { IndeterminateProgressBar } from "./IndeterminateProgressBar";

export type JobCardPhase = "idle" | "uploading" | "queued" | "running" | "completed" | "failed" | "cancelled";

interface JobCardProps {
  phase: JobCardPhase;
  job?: AnyJobResponse | null;
  fileName?: string | null;
  errorMessage?: string | null;
  onCancel?: () => void;
}

function isVideoJob(job: AnyJobResponse): job is VideoJobResponse {
  return "videoCodec" in job;
}

function isAudioJob(job: AnyJobResponse): job is AudioJob {
  return "denoise" in job;
}

function readOutputFps(job: VideoJobResponse): string | null {
  const raw = job.metadata.outputFps;
  return typeof raw === "string" ? raw : null;
}

function readStage(job: VideoJobResponse): string | null {
  const raw = job.metadata.stage;
  return typeof raw === "string" ? raw : null;
}

function humanizeStage(stage: string): string {
  return stage.replace(/_/g, " ");
}

function IdleState() {
  return (
    <div className="flex flex-col items-center gap-2 py-8 text-center">
      <ImageIcon aria-hidden="true" className="h-6 w-6 text-text-faint" strokeWidth={1.5} />
      <p className="text-sm text-text-faint">Select a file to begin.</p>
    </div>
  );
}

function UploadingState({ fileName }: { fileName?: string | null }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 text-sm text-text">
        <UploadCloud aria-hidden="true" className="h-4 w-4 text-accent" strokeWidth={1.75} />
        <span>Uploading</span>
        {fileName && <span className="text-text-dim">{fileName}</span>}
      </div>
      <IndeterminateProgressBar label="Uploading" />
    </div>
  );
}

function readProgressPct(job?: AnyJobResponse | null): number | null {
  return job?.progressPct ?? null;
}

function JobProgressBar({ label, job }: { label: string; job?: AnyJobResponse | null }) {
  const progressPct = readProgressPct(job);
  if (isProgressDeterminate(progressPct)) {
    return (
      <div className="flex flex-col gap-1">
        <DeterminateProgressBar label={label} percent={progressPct} />
        <span className="font-mono-tabular self-end text-xs text-text-dim">{Math.round(progressPct)}%</span>
      </div>
    );
  }
  return <IndeterminateProgressBar label={label} />;
}

function CancelButton({ onCancel }: { onCancel: () => void }) {
  return (
    <button
      type="button"
      onClick={onCancel}
      className="inline-flex w-fit items-center gap-1.5 rounded-sm border border-danger px-2.5 py-1 text-xs font-medium text-danger transition-[background-color,color] duration-fast hover:bg-danger hover:text-bg focus-visible:outline focus-visible:outline-2 focus-visible:outline-danger"
    >
      <Ban aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
      Cancel
    </button>
  );
}

function QueuedState({ job, onCancel }: { job?: AnyJobResponse | null; onCancel?: () => void }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 text-sm text-text">
        <Clock aria-hidden="true" className="h-4 w-4 text-accent" strokeWidth={1.75} />
        <span>Queued</span>
      </div>
      <JobProgressBar label="Queued" job={job} />
      {onCancel && <CancelButton onCancel={onCancel} />}
    </div>
  );
}

function RunningState({ job, onCancel }: { job?: AnyJobResponse | null; onCancel?: () => void }) {
  const stage = job && isVideoJob(job) ? readStage(job) : null;
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 text-sm text-text">
        <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin text-accent" strokeWidth={1.75} />
        <span>Processing</span>
        {stage && <span className="text-text-dim">— {humanizeStage(stage)}</span>}
      </div>
      <JobProgressBar label="Processing" job={job} />
      {onCancel && <CancelButton onCancel={onCancel} />}
    </div>
  );
}

function CancelledState() {
  return (
    <div className="flex items-center gap-2 text-sm text-text-dim">
      <Ban aria-hidden="true" className="h-4 w-4 shrink-0" strokeWidth={1.75} />
      <span>Cancelled</span>
    </div>
  );
}

function DurationDetailItem({ job }: { job: AnyJobResponse }) {
  return (
    <div className="flex items-center gap-1">
      <dt className="text-text-faint">Duration</dt>
      <dd className="font-mono-tabular text-text">{formatDuration(job.startedAt, job.finishedAt)}</dd>
    </div>
  );
}

function ImageCompletedDetails({ job }: { job: JobResponse }) {
  return (
    <>
      {job.downloadUrl && (
        <img
          src={job.downloadUrl}
          alt={job.originalFilename}
          className="max-h-48 w-full rounded border border-border bg-bg object-contain"
        />
      )}
      <dl className="flex gap-4 text-xs text-text-dim">
        <div className="flex items-center gap-1">
          <dt className="sr-only">Scale</dt>
          <dd className="font-mono-tabular text-text">{job.scale}x</dd>
        </div>
        <div className="flex items-center gap-1">
          <dt className="sr-only">Format</dt>
          <dd className="uppercase text-text">{job.outputFormat}</dd>
        </div>
        <DurationDetailItem job={job} />
      </dl>
    </>
  );
}

function VideoCompletedDetails({ job }: { job: VideoJobResponse }) {
  const outputFps = readOutputFps(job);
  return (
    <dl className="flex gap-4 text-xs text-text-dim">
      <div className="flex items-center gap-1">
        <dt className="sr-only">Scale</dt>
        <dd className="font-mono-tabular text-text">{job.scale}x</dd>
      </div>
      <div className="flex items-center gap-1">
        <dt className="sr-only">Container</dt>
        <dd className="uppercase text-text">{job.outputContainer}</dd>
      </div>
      {outputFps && (
        <div className="flex items-center gap-1">
          <dt className="text-text-faint">FPS</dt>
          <dd className="font-mono-tabular text-text">{formatFps(outputFps)}</dd>
        </div>
      )}
      <DurationDetailItem job={job} />
    </dl>
  );
}

function AudioCompletedDetails({ job }: { job: AudioJob }) {
  return (
    <dl className="flex gap-4 text-xs text-text-dim">
      <div className="flex items-center gap-1">
        <dt className="text-text-faint">Denoise</dt>
        <dd className="text-text">{denoiseLabel(job.denoise)}</dd>
      </div>
      <div className="flex items-center gap-1">
        <dt className="text-text-faint">Restore</dt>
        <dd className="text-text">{restoreLabel(job.restore)}</dd>
      </div>
      <DurationDetailItem job={job} />
    </dl>
  );
}

function GenerationCompletedDetails({ job }: { job: GenerationJob }) {
  return (
    <>
      {job.downloadUrl && (
        <img
          src={job.downloadUrl}
          alt="Generated image"
          className="max-h-48 w-full rounded border border-border bg-bg object-contain"
        />
      )}
      <dl className="flex gap-4 text-xs text-text-dim">
        <div className="flex items-center gap-1">
          <dt className="sr-only">Dimensions</dt>
          <dd className="font-mono-tabular text-text">
            {job.width}x{job.height}
          </dd>
        </div>
        <DurationDetailItem job={job} />
      </dl>
    </>
  );
}

function CompletedDetails({ job }: { job: AnyJobResponse }) {
  if (isVideoJob(job)) {
    return <VideoCompletedDetails job={job} />;
  }
  if (isAudioJob(job)) {
    return <AudioCompletedDetails job={job} />;
  }
  if (isGenerationJob(job)) {
    return <GenerationCompletedDetails job={job} />;
  }
  return <ImageCompletedDetails job={job} />;
}

function CompletedState({ job }: { job: AnyJobResponse }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2 text-sm text-ok">
        <CheckCircle2 aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
        <span>Completed</span>
      </div>
      <CompletedDetails job={job} />
      {job.downloadUrl && (
        <a
          href={job.downloadUrl}
          download
          className="inline-flex items-center justify-center gap-2 rounded border border-accent bg-surface-2 px-3 py-2 text-sm font-medium text-accent transition-[background-color,color] duration-fast hover:bg-accent hover:text-bg focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          <Download aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
          Download
        </a>
      )}
    </div>
  );
}

function FailedState({ message }: { message: string }) {
  return (
    <div role="alert" className="flex items-start gap-2 rounded border border-danger bg-surface-2 px-3 py-2 text-sm text-danger">
      <AlertTriangle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" strokeWidth={1.75} />
      <span>{message}</span>
    </div>
  );
}

function resolveErrorMessage(job: AnyJobResponse | null | undefined, errorMessage?: string | null): string {
  if (errorMessage) {
    return errorMessage;
  }
  if (job?.error) {
    return job.error;
  }
  return "The job failed.";
}

// An upload-level rejection (400/429/500 from POST /jobs) never produces a
// job, so the hook's phase stays "idle" -- without this, that error would
// silently vanish instead of reaching the user.
function resolveDisplayPhase(phase: JobCardPhase, errorMessage?: string | null): JobCardPhase {
  if (phase === "idle" && errorMessage) {
    return "failed";
  }
  return phase;
}

export function JobCard({ phase, job, fileName, errorMessage, onCancel }: JobCardProps) {
  const displayPhase = resolveDisplayPhase(phase, errorMessage);

  return (
    <div aria-live="polite" className="rounded border border-border bg-surface p-4">
      {displayPhase === "idle" && <IdleState />}
      {displayPhase === "uploading" && <UploadingState fileName={fileName} />}
      {displayPhase === "queued" && <QueuedState job={job} onCancel={onCancel} />}
      {displayPhase === "running" && <RunningState job={job} onCancel={onCancel} />}
      {displayPhase === "completed" && job && <CompletedState job={job} />}
      {displayPhase === "failed" && <FailedState message={resolveErrorMessage(job, errorMessage)} />}
      {displayPhase === "cancelled" && <CancelledState />}
    </div>
  );
}
