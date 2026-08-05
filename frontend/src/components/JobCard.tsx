import { AlertTriangle, Ban, CheckCircle2, Clock, Download, ImageIcon, Loader2, UploadCloud } from "lucide-react";
import { useTranslation } from "../i18n/LocaleProvider";
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
  uploadPercent?: number | null;
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

function readUpscaleEngine(job: VideoJobResponse): string | null {
  const backend = job.metadata.upscaleBackend;
  if (typeof backend !== "string") {
    return null;
  }
  const precision = job.metadata.upscalePrecision;
  const label = backend === "onnx" && typeof precision === "string" ? `onnx ${precision}` : backend;
  return job.metadata.upscaleTiled === true ? `${label} · en mosaicos` : label;
}

function readStage(job: VideoJobResponse): string | null {
  const raw = job.metadata.stage;
  return typeof raw === "string" ? raw : null;
}

function humanizeStage(stage: string): string {
  return stage.replace(/_/g, " ");
}

function IdleState() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center gap-2 py-8 text-center">
      <ImageIcon aria-hidden="true" className="h-6 w-6 text-text-faint" strokeWidth={1.5} />
      <p className="text-sm text-text-faint">{t("job.card.selectFile")}</p>
    </div>
  );
}

function UploadingState({
  fileName,
  percent,
}: {
  fileName?: string | null;
  // `null` cuando el navegador no puede saber el total. Ahi va la barra
  // indeterminada, en vez de un porcentaje inventado.
  percent?: number | null;
}) {
  const { t } = useTranslation();
  const label = t("job.card.uploading");
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2 text-sm text-text">
        <UploadCloud aria-hidden="true" className="h-4 w-4 text-accent" strokeWidth={1.75} />
        <span>{label}</span>
        {fileName && <span className="text-text-dim">{fileName}</span>}
        {percent !== null && percent !== undefined && (
          <span className="font-mono-tabular text-text-dim">{percent}%</span>
        )}
      </div>
      {percent !== null && percent !== undefined ? (
        <DeterminateProgressBar label={label} percent={percent} />
      ) : (
        <IndeterminateProgressBar label={label} />
      )}
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
  // Sin esto, "en mi maquina va lentisimo" no se puede diagnosticar: el motor y
  // la precision ya se registraban, pero no se veian en ninguna parte.
  const engine = readUpscaleEngine(job);
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
      {engine && (
        <div className="flex items-center gap-1">
          <dt className="text-text-faint">Motor</dt>
          <dd className="text-text">{engine}</dd>
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

function GenerationPreview({ job }: { job: GenerationJob }) {
  const { t } = useTranslation();
  if (!job.downloadUrl) {
    return null;
  }
  const className = "max-h-48 w-full rounded border border-border bg-bg object-contain";
  // Un .webm pintado como <img> sale roto: el navegador no lo decodifica. La URL
  // de descarga no lleva extension, asi que la bandera del backend es lo unico
  // que distingue un clip de una imagen.
  if (job.isVideo) {
    return <video src={job.downloadUrl} controls loop muted playsInline className={className} />;
  }
  return <img src={job.downloadUrl} alt={t("job.card.generatedImage")} className={className} />;
}

function GenerationCompletedDetails({ job }: { job: GenerationJob }) {
  return (
    <>
      <GenerationPreview job={job} />
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

// Recibe `t`: es una funcion pura, no un componente.
function resolveErrorMessage(
  job: AnyJobResponse | null | undefined,
  errorMessage: string | null | undefined,
  t: (key: string) => string,
): string {
  if (errorMessage) {
    return errorMessage;
  }
  if (job?.error) {
    return job.error;
  }
  return t("job.failed");
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

// Decisiones que el backend tomo SOLO y que cambian lo que el usuario recibe.
// Quedaban anotadas en la metadata y no salian a ningun lado: el trabajo
// terminaba "bien" con un resultado distinto del pedido.
const SILENT_DECISIONS: { key: string; labelKey: string }[] = [
  { key: "videoEncoderFallback", labelKey: "job.notice.encoderFallback" },
  { key: "containerUpgradedReason", labelKey: "job.notice.containerUpgraded" },
  { key: "streamPipelineFallback", labelKey: "job.notice.slowerPipeline" },
  { key: "rawPipeFallback", labelKey: "job.notice.slowerPipeline" },
  { key: "masteringSkipped", labelKey: "job.notice.masteringSkipped" },
];

function readNotices(job: AnyJobResponse): { labelKey: string; detail: string }[] {
  const avisos: { labelKey: string; detail: string }[] = [];
  const upscaleError = isGenerationJob(job) ? job.upscaleError : null;
  if (typeof upscaleError === "string" && upscaleError) {
    avisos.push({ labelKey: "job.notice.upscaleFailed", detail: upscaleError });
  }
  const metadata = "metadata" in job ? job.metadata : undefined;
  if (metadata) {
    for (const { key, labelKey } of SILENT_DECISIONS) {
      const valor = metadata[key];
      if (typeof valor === "string" && valor) {
        avisos.push({ labelKey, detail: valor });
      }
    }
  }
  return avisos;
}

function JobNotices({ job }: { job: AnyJobResponse }) {
  const { t } = useTranslation();
  const avisos = readNotices(job);
  if (avisos.length === 0) {
    return null;
  }
  return (
    <div role="status" className="flex flex-col gap-1 rounded border border-warn bg-surface-2 p-2">
      {avisos.map(({ labelKey, detail }) => (
        <div key={`${labelKey}-${detail}`} className="flex flex-col">
          <span className="text-sm text-text">{t(labelKey)}</span>
          {/* El detalle viene tal cual de ffmpeg o del motor: es diagnostico,
              no copia, y traducirlo lo volveria irrastreable. */}
          <span className="text-xs text-text-dim">{detail}</span>
        </div>
      ))}
    </div>
  );
}

export function JobCard({
  phase,
  job,
  fileName,
  errorMessage,
  onCancel,
  uploadPercent,
}: JobCardProps) {
  const { t } = useTranslation();
  const displayPhase = resolveDisplayPhase(phase, errorMessage);

  return (
    <div aria-live="polite" className="rounded border border-border bg-surface p-4">
      {displayPhase === "idle" && <IdleState />}
      {displayPhase === "uploading" && <UploadingState fileName={fileName} percent={uploadPercent} />}
      {displayPhase === "queued" && <QueuedState job={job} onCancel={onCancel} />}
      {displayPhase === "running" && <RunningState job={job} onCancel={onCancel} />}
      {displayPhase === "completed" && job && (
        <div className="flex flex-col gap-3">
          <CompletedState job={job} />
          <JobNotices job={job} />
        </div>
      )}
      {displayPhase === "failed" && <FailedState message={resolveErrorMessage(job, errorMessage, t)} />}
      {displayPhase === "cancelled" && <CancelledState />}
    </div>
  );
}
