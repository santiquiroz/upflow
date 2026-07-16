import { AlertTriangle, CheckCircle2, Clock, Download, ImageIcon, Loader2, UploadCloud } from "lucide-react";
import type { JobResponse, VideoJobResponse } from "../lib/apiTypes";
import { formatFps } from "../lib/formatFps";
import { IndeterminateProgressBar } from "./IndeterminateProgressBar";

export type JobCardPhase = "idle" | "uploading" | "queued" | "running" | "completed" | "failed";

type AnyJobResponse = JobResponse | VideoJobResponse;

interface JobCardProps {
  phase: JobCardPhase;
  job?: AnyJobResponse | null;
  fileName?: string | null;
  errorMessage?: string | null;
}

function isVideoJob(job: AnyJobResponse): job is VideoJobResponse {
  return "videoCodec" in job;
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

function QueuedState() {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 text-sm text-text">
        <Clock aria-hidden="true" className="h-4 w-4 text-accent" strokeWidth={1.75} />
        <span>Queued</span>
      </div>
      <IndeterminateProgressBar label="Queued" />
    </div>
  );
}

function RunningState({ job }: { job?: AnyJobResponse | null }) {
  const stage = job && isVideoJob(job) ? readStage(job) : null;
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 text-sm text-text">
        <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin text-accent" strokeWidth={1.75} />
        <span>Processing</span>
        {stage && <span className="text-text-dim">— {humanizeStage(stage)}</span>}
      </div>
      <IndeterminateProgressBar label="Processing" />
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
    </dl>
  );
}

function CompletedState({ job }: { job: AnyJobResponse }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2 text-sm text-ok">
        <CheckCircle2 aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
        <span>Completed</span>
      </div>
      {isVideoJob(job) ? <VideoCompletedDetails job={job} /> : <ImageCompletedDetails job={job} />}
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

export function JobCard({ phase, job, fileName, errorMessage }: JobCardProps) {
  const displayPhase = resolveDisplayPhase(phase, errorMessage);

  return (
    <div aria-live="polite" className="rounded border border-border bg-surface p-4">
      {displayPhase === "idle" && <IdleState />}
      {displayPhase === "uploading" && <UploadingState fileName={fileName} />}
      {displayPhase === "queued" && <QueuedState />}
      {displayPhase === "running" && <RunningState job={job} />}
      {displayPhase === "completed" && job && <CompletedState job={job} />}
      {displayPhase === "failed" && <FailedState message={resolveErrorMessage(job, errorMessage)} />}
    </div>
  );
}
