import { AlertTriangle, CheckCircle2, Clock, Download, Loader2, X } from "lucide-react";
import { useState } from "react";
import { type JobQueueEntry, useJobQueue } from "../hooks/useJobQueue";
import { isTerminalJobStatus, jobKindLabel } from "../lib/jobStatus";
import { IndeterminateProgressBar } from "./IndeterminateProgressBar";
import { JobDetailModal } from "./JobDetailModal";

function QueuedStatus() {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-1.5 text-xs text-text">
        <Clock aria-hidden="true" className="h-3.5 w-3.5 text-accent" strokeWidth={1.75} />
        <span>Queued</span>
      </div>
      <IndeterminateProgressBar label="Queued" />
    </div>
  );
}

function RunningStatus() {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-1.5 text-xs text-text">
        <Loader2 aria-hidden="true" className="h-3.5 w-3.5 animate-spin text-accent" strokeWidth={1.75} />
        <span>Processing</span>
      </div>
      <IndeterminateProgressBar label="Processing" />
    </div>
  );
}

function CompletedStatus({ entry }: { entry: JobQueueEntry }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="flex items-center gap-1.5 text-xs text-ok">
        <CheckCircle2 aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
        Completed
      </span>
      {entry.downloadUrl && (
        <a
          href={entry.downloadUrl}
          download
          aria-label={`Download ${entry.fileName}`}
          className="inline-flex items-center gap-1 rounded-sm border border-accent px-2 py-1 text-xs font-medium text-accent transition-[background-color,color] duration-fast hover:bg-accent hover:text-bg focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          <Download aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
          Download
        </a>
      )}
    </div>
  );
}

function FailedStatus({ entry }: { entry: JobQueueEntry }) {
  return (
    <p role="alert" className="flex items-start gap-1.5 text-xs text-danger">
      <AlertTriangle aria-hidden="true" className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
      <span>{entry.errorMessage ?? "The job failed."}</span>
    </p>
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
  return <FailedStatus entry={entry} />;
}

function DismissButton({ entry, onDismiss }: { entry: JobQueueEntry; onDismiss: (id: string) => void }) {
  return (
    <button
      type="button"
      aria-label={`Dismiss ${entry.fileName}`}
      onClick={() => onDismiss(entry.id)}
      className="shrink-0 rounded-sm p-1 text-text-faint transition-colors duration-fast hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
    >
      <X aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
    </button>
  );
}

function QueueEntryRow({
  entry,
  onDismiss,
  onOpenDetail,
}: {
  entry: JobQueueEntry;
  onDismiss: (id: string) => void;
  onOpenDetail: (id: string) => void;
}) {
  const isTerminal = isTerminalJobStatus(entry.status);
  return (
    <li className="flex flex-col gap-2 rounded border border-border bg-surface-2 p-3">
      <div className="flex items-start justify-between gap-2">
        <button
          type="button"
          onClick={() => onOpenDetail(entry.id)}
          aria-label={`View details for ${entry.fileName}`}
          className="flex min-w-0 flex-col text-left transition-colors duration-fast hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          <span className="truncate text-xs text-text" title={entry.fileName}>
            {entry.fileName}
          </span>
          <span className="text-[10px] uppercase tracking-wide text-text-faint">{jobKindLabel(entry.kind)}</span>
        </button>
        {isTerminal && <DismissButton entry={entry} onDismiss={onDismiss} />}
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
  return <p className="text-sm text-text-faint">No active jobs.</p>;
}

export function JobQueue() {
  const { entries, dismiss, clearCompleted } = useJobQueue();
  const hasCompletedOrFailed = entries.some((entry) => isTerminalJobStatus(entry.status));
  const [detailJobId, setDetailJobId] = useState<string | null>(null);
  const detailEntry = entries.find((entry) => entry.id === detailJobId);

  return (
    <div className="flex h-full flex-col gap-3">
      <div className="flex items-center justify-between gap-2">
        <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">Job Queue</h2>
        <QueueCount count={entries.length} />
      </div>
      {entries.length === 0 ? (
        <EmptyQueueState />
      ) : (
        <ul className="flex flex-col gap-2 overflow-y-auto">
          {entries.map((entry) => (
            <QueueEntryRow key={entry.id} entry={entry} onDismiss={dismiss} onOpenDetail={setDetailJobId} />
          ))}
        </ul>
      )}
      {hasCompletedOrFailed && (
        <button
          type="button"
          onClick={clearCompleted}
          className="mt-auto text-left text-xs text-text-dim transition-colors duration-fast hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          Clear completed
        </button>
      )}
      {detailEntry && <JobDetailModal entry={detailEntry} onClose={() => setDetailJobId(null)} />}
    </div>
  );
}
