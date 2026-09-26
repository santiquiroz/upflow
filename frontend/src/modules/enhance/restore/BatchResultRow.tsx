import { ChevronDown, Download, ImageIcon, ScanFace } from "lucide-react";
import { useId, useState } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import { jobStatusLabelKey } from "../../../lib/jobStatus";
import { restoreArtifactUrl } from "../../../services/restore";
import {
  enabledFaceCount,
  faceCountKey,
  hasRestoredFaces,
  ranFaces,
  type BatchRowState,
} from "./batchResultModel";
import { FaceResultGrid } from "./FaceResultGrid";
import { withRecomposedSidecar, type RestoreResultSummary } from "./restoreResultModel";
import { revisedUrl } from "./restoreUrls";
import type { BatchResultDeps } from "./useBatchJobs";
import type { BatchEntry } from "./useRestoreBatch";

export interface BatchResultRowProps {
  entry: BatchEntry;
  state: BatchRowState;
  reviewed: boolean;
  onReviewed: (jobId: string) => void;
  recompose: BatchResultDeps["recompose"];
}

const ACTION_CLASS =
  "inline-flex items-center gap-1.5 rounded border px-2.5 py-1 text-xs transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";

function Thumbnail({ state, name, revision }: { state: BatchRowState; name: string; revision: number }) {
  const { t } = useTranslation();
  if (state.kind !== "done") {
    return (
      <div aria-hidden="true" className="flex h-12 w-12 shrink-0 items-center justify-center rounded-sm bg-surface-2">
        <ImageIcon className="h-5 w-5 text-text-faint" strokeWidth={1.5} />
      </div>
    );
  }
  return (
    <img
      src={revisedUrl(restoreArtifactUrl(state.job.jobId, "preview"), revision)}
      alt={t("restore.batch.results.thumbAlt", { name })}
      className="h-12 w-12 shrink-0 rounded-sm bg-surface-2 object-cover"
    />
  );
}

function StatusLine({ state, summary }: { state: BatchRowState; summary: RestoreResultSummary | null }) {
  const { t } = useTranslation();
  const job = state.job;
  if (summary !== null && job !== undefined && ranFaces(job)) {
    return <span>{t(faceCountKey(summary), { count: enabledFaceCount(summary) })}</span>;
  }
  if (job?.status === "running" && job.progressPct !== null) {
    return <span>{t("restore.batch.results.progress", { pct: Math.round(job.progressPct) })}</span>;
  }
  const failed = job?.status === "failed";
  return (
    <span className={failed ? "text-danger" : undefined}>
      {t(jobStatusLabelKey(job?.status ?? "queued"))}
      {failed && job?.error ? ` · ${job.error}` : ""}
    </span>
  );
}

function ReviewBadge({ reviewed }: { reviewed: boolean }) {
  const { t } = useTranslation();
  const tone = reviewed ? "border-ok text-ok" : "border-warn text-warn";
  return (
    <span className={`rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide ${tone}`}>
      {t(reviewed ? "restore.batch.results.reviewed" : "restore.batch.results.notReviewed")}
    </span>
  );
}

interface ReviewToggleProps {
  open: boolean;
  controls: string;
  onToggle: () => void;
}

function ReviewToggle({ open, controls, onToggle }: ReviewToggleProps) {
  const { t } = useTranslation();
  return (
    <button
      type="button"
      aria-expanded={open}
      aria-controls={controls}
      onClick={onToggle}
      className={`${ACTION_CLASS} border-accent font-medium text-accent hover:bg-accent hover:text-bg`}
    >
      <ScanFace aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
      {t(open ? "restore.batch.results.hideReview" : "restore.batch.results.review")}
      <ChevronDown
        aria-hidden="true"
        className={`h-3.5 w-3.5 transition-transform motion-reduce:transition-none ${open ? "rotate-180" : ""}`}
        strokeWidth={1.75}
      />
    </button>
  );
}

function DownloadRestored({ href, fileName }: { href: string; fileName: string }) {
  const { t } = useTranslation();
  return (
    <a href={href} download={fileName} className={`${ACTION_CLASS} border-border text-text hover:border-accent hover:text-accent`}>
      <Download aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
      {t("restore.result.download.restored")}
    </a>
  );
}

function currentSummary(state: BatchRowState, recomposed: RestoreResultSummary | null): RestoreResultSummary | null {
  if (state.kind !== "done") {
    return null;
  }
  return recomposed ?? state.summary;
}

export function BatchResultRow({ entry, state, reviewed, onReviewed, recompose }: BatchResultRowProps) {
  const reviewId = useId();
  const [open, setOpen] = useState(false);
  const [recomposed, setRecomposed] = useState<RestoreResultSummary | null>(null);
  const [revision, setRevision] = useState(0);
  const summary = currentSummary(state, recomposed);
  const reviewable = summary !== null && hasRestoredFaces(summary);

  function toggle() {
    setOpen((current) => !current);
    onReviewed(entry.jobId);
  }

  function handleRecomposed(sidecar: Record<string, unknown>) {
    if (summary !== null) {
      setRecomposed(withRecomposedSidecar(summary, sidecar));
      setRevision((current) => current + 1);
    }
  }

  return (
    <li className="flex flex-col gap-3 rounded border border-border bg-surface p-2.5">
      <div className="flex flex-wrap items-center gap-3">
        <Thumbnail state={state} name={entry.fileName} revision={revision} />
        <div className="flex min-w-0 flex-1 flex-col gap-0.5">
          <p className="truncate text-xs font-medium text-text" title={entry.fileName}>
            {entry.fileName}
          </p>
          <p className="text-xs text-text-dim">
            <StatusLine state={state} summary={summary} />
          </p>
        </div>
        {reviewable && <ReviewBadge reviewed={reviewed} />}
        <div className="flex flex-wrap items-center gap-2">
          {reviewable && <ReviewToggle open={open} controls={reviewId} onToggle={toggle} />}
          {summary !== null && state.job?.downloadUrl && (
            <DownloadRestored href={revisedUrl(state.job.downloadUrl, revision)} fileName={summary.downloadNames.restored} />
          )}
        </div>
      </div>
      {reviewable && open && state.job && (
        <div id={reviewId} className="border-t border-border pt-3">
          <FaceResultGrid
            jobId={state.job.jobId}
            faces={summary.faces}
            recomposeAvailable={summary.recomposeAvailable}
            onRecomposed={handleRecomposed}
            recompose={recompose}
          />
        </div>
      )}
    </li>
  );
}
