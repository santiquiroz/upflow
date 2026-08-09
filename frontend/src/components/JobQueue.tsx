import { AlertTriangle, Ban, CheckCircle2, Clock, Download, Loader2, X } from "lucide-react";
import { useState, useEffect } from "react";
import { useAllJobsView, type AllJobsEntry } from "../hooks/useAllJobs";
import { useTranslation } from "../i18n/LocaleProvider";
import { useAuth } from "../hooks/useAuth";
import { type JobQueueEntry, useJobQueue } from "../hooks/useJobQueue";
import { rehydrateJobQueue } from "../lib/jobQueueRehydrate";
import { jobQueueStore } from "../lib/jobQueueStore";
import { isCancellableJobStatus, isTerminalJobStatus, jobKindLabel } from "../lib/jobStatus";
import { IndeterminateProgressBar } from "./IndeterminateProgressBar";
import { JobDetailModal } from "./JobDetailModal";

function QueuedStatus() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-1.5 text-xs text-text">
        <Clock aria-hidden="true" className="h-3.5 w-3.5 text-accent" strokeWidth={1.75} />
        <span>{t("job.status.queued")}</span>
      </div>
      <IndeterminateProgressBar label={t("job.status.queued")} />
    </div>
  );
}

function RunningStatus() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-1.5 text-xs text-text">
        <Loader2 aria-hidden="true" className="h-3.5 w-3.5 animate-spin text-accent" strokeWidth={1.75} />
        <span>{t("job.status.processing")}</span>
      </div>
      <IndeterminateProgressBar label={t("job.status.processing")} />
    </div>
  );
}

const QUEUE_DOWNLOAD_LINK_CLASS =
  "inline-flex items-center gap-1 rounded-sm border border-accent px-2 py-1 text-xs font-medium text-accent transition-[background-color,color] duration-fast hover:bg-accent hover:text-bg focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";

function readVocalsUrl(entry: JobQueueEntry): string | null {
  // Solo los jobs de audio en modo karaoke traen el segundo stem.
  const job = entry.job;
  return job && "vocalsDownloadUrl" in job ? job.vocalsDownloadUrl ?? null : null;
}

function CompletedDownloadLinks({ entry }: { entry: JobQueueEntry }) {
  const { t } = useTranslation();
  if (!entry.downloadUrl) {
    return null;
  }
  const vocalsUrl = readVocalsUrl(entry);
  if (vocalsUrl) {
    // Dos links etiquetados: el generico bajaba SOLO la instrumental y dejaba
    // la voz inalcanzable desde la cola tras un refresh.
    return (
      <div className="flex flex-wrap justify-end gap-1.5">
        <a href={entry.downloadUrl} download className={QUEUE_DOWNLOAD_LINK_CLASS}>
          <Download aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
          {t("audio.karaoke.download.instrumental")}
        </a>
        <a href={vocalsUrl} download className={QUEUE_DOWNLOAD_LINK_CLASS}>
          <Download aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
          {t("audio.karaoke.download.vocals")}
        </a>
      </div>
    );
  }
  return (
    <a
      href={entry.downloadUrl}
      download
      aria-label={t("job.download.aria", { name: entry.fileName })}
      className={QUEUE_DOWNLOAD_LINK_CLASS}
    >
      <Download aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
      {t("job.download")}
    </a>
  );
}

function CompletedStatus({ entry }: { entry: JobQueueEntry }) {
  const { t } = useTranslation();
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="flex items-center gap-1.5 text-xs text-ok">
        <CheckCircle2 aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
        {t("job.status.completed")}
      </span>
      <CompletedDownloadLinks entry={entry} />
    </div>
  );
}

function FailedStatus({ entry }: { entry: JobQueueEntry }) {
  const { t } = useTranslation();
  return (
    <p role="alert" className="flex items-start gap-1.5 text-xs text-danger">
      <AlertTriangle aria-hidden="true" className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
      <span>{entry.errorMessage ?? t("job.failed")}</span>
    </p>
  );
}

function CancelledStatus() {
  const { t } = useTranslation();
  return (
    <span className="flex items-center gap-1.5 text-xs text-text-dim">
      <Ban aria-hidden="true" className="h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
      {t("job.status.cancelled")}
    </span>
  );
}

function QueueEntryStatus({ entry }: { entry: JobQueueEntry }) {
  if (entry.status === "queued") {
    return <QueuedStatus />;
  }
  if (entry.status === "running") {
    return <RunningStatus />;
  }
  if (entry.status === "completed") {
    return <CompletedStatus entry={entry} />;
  }
  if (entry.status === "cancelled") {
    return <CancelledStatus />;
  }
  return <FailedStatus entry={entry} />;
}

function DismissButton({ entry, onDismiss }: { entry: JobQueueEntry; onDismiss: (id: string) => void }) {
  const { t } = useTranslation();
  return (
    <button
      type="button"
      aria-label={t("job.dismiss.aria", { name: entry.fileName })}
      onClick={() => onDismiss(entry.id)}
      className="shrink-0 rounded-sm p-1 text-text-faint transition-colors duration-fast hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
    >
      <X aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
    </button>
  );
}

function CancelButton({ entry, onCancel }: { entry: JobQueueEntry; onCancel: (id: string) => void }) {
  const { t } = useTranslation();
  return (
    <button
      type="button"
      aria-label={t("job.cancel.aria", { name: entry.fileName })}
      onClick={() => onCancel(entry.id)}
      className="shrink-0 rounded-sm p-1 text-text-faint transition-colors duration-fast hover:text-danger focus-visible:outline focus-visible:outline-2 focus-visible:outline-danger"
    >
      <Ban aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
    </button>
  );
}

function QueueEntryAction({
  entry,
  onDismiss,
  onCancel,
}: {
  entry: JobQueueEntry;
  onDismiss: (id: string) => void;
  onCancel: (id: string) => void;
}) {
  if (isCancellableJobStatus(entry.status)) {
    return <CancelButton entry={entry} onCancel={onCancel} />;
  }
  if (isTerminalJobStatus(entry.status)) {
    return <DismissButton entry={entry} onDismiss={onDismiss} />;
  }
  return null;
}

function QueueEntryRow({
  entry,
  onDismiss,
  onCancel,
  onOpenDetail,
}: {
  entry: JobQueueEntry;
  onDismiss: (id: string) => void;
  onCancel: (id: string) => void;
  onOpenDetail: (id: string) => void;
}) {
  const { t } = useTranslation();
  return (
    <li className="flex flex-col gap-2 rounded border border-border bg-surface-2 p-3">
      <div className="flex items-start justify-between gap-2">
        <button
          type="button"
          onClick={() => onOpenDetail(entry.id)}
          aria-label={t("job.viewDetails.aria", { name: entry.fileName })}
          className="flex min-w-0 flex-col text-left transition-colors duration-fast hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          <span className="truncate text-xs text-text" title={entry.fileName}>
            {entry.fileName}
          </span>
          <span className="text-[10px] uppercase tracking-wide text-text-faint">{jobKindLabel(entry.kind)}</span>
        </button>
        <QueueEntryAction entry={entry} onDismiss={onDismiss} onCancel={onCancel} />
      </div>
      <QueueEntryStatus entry={entry} />
    </li>
  );
}

function QueueCount({ count }: { count: number }) {
  if (count === 0) {
    return null;
  }
  return (
    <span className="font-mono-tabular rounded-sm bg-surface-2 px-1.5 py-0.5 text-[10px] text-text-dim">{count}</span>
  );
}

function EmptyQueueState() {
  const { t } = useTranslation();
  return <p className="text-sm text-text-faint">{t("job.queue.empty")}</p>;
}

function AllJobsRow({ entry }: { entry: AllJobsEntry }) {
  const { t } = useTranslation();
  return (
    <li className="flex flex-col gap-1 rounded border border-border bg-surface-2 p-3">
      <div className="flex items-start justify-between gap-2">
        <span className="truncate text-xs text-text" title={entry.fileName}>
          {entry.fileName}
        </span>
        <span className="text-[10px] uppercase tracking-wide text-text-faint">{jobKindLabel(entry.kind)}</span>
      </div>
      <div className="flex items-center justify-between text-[10px] text-text-faint">
        <span>{t("job.owner", { owner: entry.ownerId ?? "—" })}</span>
        <span>{entry.status}</span>
      </div>
    </li>
  );
}

export function JobQueue() {
  // Recargar el navegador borraba la cola aunque los trabajos siguieran vivos en
  // el servidor. Se rehidrata desde el servidor una sola vez al montar: es la
  // fuente real, asi que no puede mostrar trabajos que ya no existen.
  useEffect(() => {
    void rehydrateJobQueue(jobQueueStore);
  }, []);

  const { t } = useTranslation();
  const { entries, dismiss, cancel, clearCompleted } = useJobQueue();
  const { hasPermission } = useAuth();
  const [viewAll, setViewAll] = useState(false);
  const allJobsEntries = useAllJobsView(viewAll);
  const canViewAll = hasPermission("jobs:read_all");
  const hasCompletedOrFailed = entries.some((entry) => isTerminalJobStatus(entry.status));
  const [detailJobId, setDetailJobId] = useState<string | null>(null);
  const detailEntry = entries.find((entry) => entry.id === detailJobId);

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">{t("job.queue.title")}</h2>
        <QueueCount count={viewAll ? allJobsEntries.length : entries.length} />
      </div>
      {canViewAll && (
        <label className="flex items-center gap-1.5 text-xs text-text-dim">
          <input type="checkbox" checked={viewAll} onChange={(event) => setViewAll(event.target.checked)} />
          {t("job.queue.showAll")}
        </label>
      )}
      {viewAll ? (
        allJobsEntries.length === 0 ? (
          <EmptyQueueState />
        ) : (
          <ul className="flex flex-col gap-2 overflow-y-auto">
            {allJobsEntries.map((entry) => (
              <AllJobsRow key={`${entry.kind}-${entry.id}`} entry={entry} />
            ))}
          </ul>
        )
      ) : entries.length === 0 ? (
        <EmptyQueueState />
      ) : (
        <ul className="flex flex-col gap-2 overflow-y-auto">
          {entries.map((entry) => (
            <QueueEntryRow
              key={entry.id}
              entry={entry}
              onDismiss={dismiss}
              onCancel={cancel}
              onOpenDetail={setDetailJobId}
            />
          ))}
        </ul>
      )}
      {!viewAll && hasCompletedOrFailed && (
        <button
          type="button"
          onClick={clearCompleted}
          className="mt-auto text-left text-xs text-text-dim transition-colors duration-fast hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          {t("job.queue.clearCompleted")}
        </button>
      )}
      {detailEntry && <JobDetailModal entry={detailEntry} onClose={() => setDetailJobId(null)} onCancel={cancel} />}
    </div>
  );
}
