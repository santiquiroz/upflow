import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Ban, CheckCircle2, Circle, Loader2 } from "lucide-react";
import { useTranslation } from "../i18n/LocaleProvider";
import { Fragment, useEffect, useState } from "react";
import { getDevices } from "../lib/api";
import type { JobQueueEntry } from "../hooks/useJobQueue";
import { useCatalogLabels } from "../hooks/useCatalogLabels";
import type { JobStage } from "../lib/apiTypes";
import { estimateEta, formatEta, type EtaSample } from "../lib/eta";
import { buildJobDetailSections, type DetailItem, type JobDetailSections } from "../lib/jobDetails";
import {
  areFramesReportable,
  deriveStepper,
  isProgressDeterminate,
  resolveFramesDenominator,
  toMonotonicProgressPct,
} from "../lib/jobProgress";
import { translateStageLabel } from "../lib/jobStageLabels";
import { isCancellableJobStatus } from "../lib/jobStatus";
import { isVideoJob, type AnyQueuedJob } from "../lib/jobTypeGuards";
import { DeterminateProgressBar } from "./DeterminateProgressBar";
import { IndeterminateProgressBar } from "./IndeterminateProgressBar";
import { Modal } from "./Modal";

interface JobDetailModalProps {
  entry: JobQueueEntry;
  onClose: () => void;
  onCancel?: (id: string) => void;
}

const MAX_ETA_SAMPLES = 5;
const ELAPSED_TICK_MS = 1000;

// Audio and generation jobs carry stages at the top level, image/video jobs nest
// them under metadata, and transcribe/download/shape3d have none -- normalize.
function resolveStages(job: AnyQueuedJob | undefined): JobStage[] | undefined {
  if (!job) {
    return undefined;
  }
  if ("stages" in job) {
    return job.stages ?? undefined;
  }
  return "metadata" in job ? job.metadata?.stages : undefined;
}

function titleIdFor(jobId: string): string {
  return `job-detail-title-${jobId}`;
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

// El "lleva corriendo" solo avanza si algo lo empuja. Se tickea unicamente
// mientras el trabajo corre: un modal abierto sobre un job terminado no tiene
// por que re-renderizarse cada segundo.
function useTickingNow(isRunning: boolean): number {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!isRunning) {
      return;
    }
    const timer = setInterval(() => setNow(Date.now()), ELAPSED_TICK_MS);
    return () => clearInterval(timer);
  }, [isRunning]);

  return now;
}

// Nombre real de la placa en vez del id crudo: "AMD Radeon RX 7900 XT" dice
// algo; "dml:0" no. El id queda entre paréntesis para diagnóstico.
function useDeviceInfo(): { label: (deviceId: string) => string; defaultId: string | null } {
  const devicesQuery = useQuery({ queryKey: ["devices"], queryFn: getDevices, staleTime: 60_000 });
  return {
    label: (deviceId: string) => {
      const match = devicesQuery.data?.devices.find((device) => device.id === deviceId);
      return match ? `${match.name} (${deviceId})` : deviceId;
    },
    defaultId: devicesQuery.data?.defaultDeviceId ?? null,
  };
}

function useJobDetailSections(entry: JobQueueEntry): JobDetailSections {
  const { t } = useTranslation();
  const device = useDeviceInfo();
  const labelFor = useCatalogLabels(entry.kind === "audio");
  const nowMs = useTickingNow(entry.status === "running");
  return buildJobDetailSections(entry.job, {
    t,
    deviceLabel: device.label,
    defaultDeviceId: device.defaultId,
    labelFor,
    nowMs,
  });
}

function detailValueClassName(item: DetailItem): string {
  if (item.isLong) {
    // Un prompt, una ruta o un bloque de código no pueden estirar el modal:
    // ocupan las dos columnas y se parten donde haga falta.
    return "col-span-2 break-words whitespace-pre-wrap text-text";
  }
  return item.isNumeric ? "font-mono-tabular break-words text-right text-text" : "break-words text-right text-text";
}

function DetailList({ items }: { items: DetailItem[] }) {
  const { t } = useTranslation();
  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs text-text-dim">
      {items.map((item) => (
        <Fragment key={`${item.labelKey}-${item.value}`}>
          <dt className={item.isLong ? "col-span-2 text-text-faint" : "text-text-faint"}>
            {t(item.labelKey)}
          </dt>
          <dd className={detailValueClassName(item)}>{item.value}</dd>
        </Fragment>
      ))}
    </dl>
  );
}

function DetailSection({ titleKey, items }: { titleKey: string; items: DetailItem[] }) {
  const { t } = useTranslation();
  if (items.length === 0) {
    return null;
  }
  return (
    <section className="flex flex-col gap-2">
      <h3 className="font-heading text-[10px] font-semibold uppercase tracking-wide text-text-faint">
        {t(titleKey)}
      </h3>
      <DetailList items={items} />
    </section>
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

function Stepper({ job }: { job: AnyQueuedJob | undefined }) {
  const { t } = useTranslation();
  const steps = deriveStepper(resolveStages(job));
  if (steps.length === 0) {
    return null;
  }
  return (
    <ol className="flex flex-col gap-2">
      {steps.map((step) => (
        <li key={step.key} className="flex items-center gap-2 text-xs">
          <StepIcon state={step.iconState} />
          <span className={stepTextClassName(step.iconState)}>{translateStageLabel(step, t)}</span>
        </li>
      ))}
    </ol>
  );
}

function ProgressSection({
  job,
  monotonicProgressPct,
}: {
  job: AnyQueuedJob | undefined;
  monotonicProgressPct: number | null;
}) {
  const { t } = useTranslation();
  const label = t("job.detail.progress");
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

function FramesReadout({ job }: { job: AnyQueuedJob | undefined }) {
  const { t } = useTranslation();
  if (!job || !isVideoJob(job)) {
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
      {` ${t("job.detail.frames")}`}
    </p>
  );
}

function EtaReadout({ samples }: { samples: EtaSample[] }) {
  const { t } = useTranslation();
  const etaSeconds = estimateEta(samples);
  if (etaSeconds === null) {
    return null;
  }
  return <p className="text-xs text-text-dim">{`${t("job.detail.eta")} ${formatEta(etaSeconds)}`}</p>;
}

function ErrorNotice({ message }: { message: string }) {
  return (
    <div
      role="alert"
      className="flex items-start gap-2 rounded border border-danger bg-surface-2 px-3 py-2 text-sm text-danger"
    >
      <AlertTriangle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" strokeWidth={1.75} />
      <span className="min-w-0 break-words whitespace-pre-wrap">{message}</span>
    </div>
  );
}

function ModalActions({ entry, onClose, onCancel }: JobDetailModalProps) {
  const { t } = useTranslation();
  return (
    <div className="ml-auto flex w-fit gap-2">
      {isCancellableJobStatus(entry.status) && onCancel && (
        <button
          type="button"
          onClick={() => onCancel(entry.id)}
          className="inline-flex items-center gap-1.5 rounded-sm border border-danger px-3 py-1.5 text-sm text-danger transition-[background-color,color] duration-fast hover:bg-danger hover:text-bg focus-visible:outline focus-visible:outline-2 focus-visible:outline-danger"
        >
          <Ban aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
          {t("job.detail.cancel")}
        </button>
      )}
      <button
        type="button"
        onClick={onClose}
        className="rounded-sm border border-border bg-surface px-3 py-1.5 text-sm text-text-dim transition-[border-color,color] duration-fast hover:border-text-faint hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        {t("job.detail.close")}
      </button>
    </div>
  );
}

function showsProgress(status: JobQueueEntry["status"]): boolean {
  return status !== "failed" && status !== "cancelled";
}

// La generacion 3D no reporta fraccion (es un hilo de CPU no interrumpible):
// ahi la barra queda indeterminada en vez de inventar un porcentaje.
function readProgressPct(job: AnyQueuedJob | undefined): number | null {
  if (!job || !("progressPct" in job)) {
    return null;
  }
  return job.progressPct ?? null;
}

function ProgressBlock({
  entry,
  monotonicProgressPct,
  etaSamples,
  timing,
}: {
  entry: JobQueueEntry;
  monotonicProgressPct: number | null;
  etaSamples: EtaSample[];
  timing: DetailItem[];
}) {
  const { t } = useTranslation();
  return (
    <section className="flex flex-col gap-2">
      <h3 className="font-heading text-[10px] font-semibold uppercase tracking-wide text-text-faint">
        {t("job.detail.group.progress")}
      </h3>
      <Stepper job={entry.job} />
      {showsProgress(entry.status) && (
        <ProgressSection job={entry.job} monotonicProgressPct={monotonicProgressPct} />
      )}
      {entry.status === "running" && <EtaReadout samples={etaSamples} />}
      <DetailList items={timing} />
    </section>
  );
}

export function JobDetailModal({ entry, onClose, onCancel }: JobDetailModalProps) {
  const titleId = titleIdFor(entry.id);
  const rawProgressPct = readProgressPct(entry.job);
  const monotonicProgressPct = useMonotonicProgressPct(entry.id, rawProgressPct);
  const etaSamples = useEtaSampleBuffer(entry.id, monotonicProgressPct);
  const sections = useJobDetailSections(entry);

  return (
    <Modal titleId={titleId} onClose={onClose} widthClassName="max-w-md">
      <h2 id={titleId} className="truncate font-heading text-sm font-semibold text-text" title={entry.fileName}>
        {entry.fileName}
      </h2>
      {/* Un trabajo de video llega a veinte filas: el cuerpo scrollea y el
          titulo y los botones quedan siempre a la vista, incluso en un celular. */}
      <div className="flex min-h-0 flex-col gap-4 overflow-y-auto">
        <DetailSection titleKey="job.detail.group.parameters" items={sections.parameters} />
        <ProgressBlock
          entry={entry}
          monotonicProgressPct={monotonicProgressPct}
          etaSamples={etaSamples}
          timing={sections.timing}
        />
        <DetailSection titleKey="job.detail.group.result" items={sections.result} />
        {/* El estado cancelado ya se lee en la fila de Estado: repetirlo en un
            cartel aparte era la misma palabra dos veces. El error si va aparte:
            es largo y tiene que interrumpir. */}
        {entry.errorMessage && <ErrorNotice message={entry.errorMessage} />}
      </div>
      <ModalActions entry={entry} onClose={onClose} onCancel={onCancel} />
    </Modal>
  );
}
