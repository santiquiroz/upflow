import { CircleCheck, TriangleAlert } from "lucide-react";
import { useId, useState } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import { batchRowState, countToReview, hasRestoredFaces, type BatchRowState } from "./batchResultModel";
import { BatchResultRow } from "./BatchResultRow";
import { useBatchJobs, type BatchResultDeps } from "./useBatchJobs";
import type { BatchEntry } from "./useRestoreBatch";

export interface RestoreBatchResultsProps {
  entries: BatchEntry[];
  deps: BatchResultDeps;
}

function anyRestoredFaces(states: readonly BatchRowState[]): boolean {
  return states.some((state) => state.kind === "done" && hasRestoredFaces(state.summary));
}

function ReviewSummary({ toReview, states }: { toReview: number; states: readonly BatchRowState[] }) {
  const { t } = useTranslation();
  if (toReview > 0) {
    return (
      <p role="status" className="flex items-center gap-1.5 text-xs font-medium text-warn">
        <TriangleAlert aria-hidden="true" className="h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
        {t(toReview === 1 ? "restore.batch.results.toReviewOne" : "restore.batch.results.toReview", { count: toReview })}
      </p>
    );
  }
  if (!anyRestoredFaces(states)) {
    return null;
  }
  return (
    <p role="status" className="flex items-center gap-1.5 text-xs text-ok">
      <CircleCheck aria-hidden="true" className="h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
      {t("restore.batch.results.allReviewed")}
    </p>
  );
}

export function RestoreBatchResults({ entries, deps }: RestoreBatchResultsProps) {
  const { t } = useTranslation();
  const titleId = useId();
  const [reviewed, setReviewed] = useState<ReadonlySet<string>>(() => new Set());
  const jobs = useBatchJobs(entries, deps);
  const states = jobs.map(batchRowState);

  function markReviewed(jobId: string) {
    setReviewed((current) => (current.has(jobId) ? current : new Set([...current, jobId])));
  }

  return (
    <section aria-labelledby={titleId} className="flex flex-col gap-2 rounded border border-border bg-surface-2 p-3">
      <h4 id={titleId} className="font-heading text-sm font-semibold text-text">
        {t("restore.batch.results.title")}
      </h4>
      <ReviewSummary toReview={countToReview(states, reviewed)} states={states} />
      <ul className="flex flex-col gap-2">
        {entries.map((entry, index) => (
          <BatchResultRow
            key={entry.jobId}
            entry={entry}
            state={states[index]}
            reviewed={reviewed.has(entry.jobId)}
            onReviewed={markReviewed}
            recompose={deps.recompose}
          />
        ))}
      </ul>
      <p className="text-xs text-text-faint">{t("restore.batch.results.keep")}</p>
    </section>
  );
}
