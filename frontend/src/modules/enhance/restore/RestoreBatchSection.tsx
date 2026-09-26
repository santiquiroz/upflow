import { Images, TriangleAlert } from "lucide-react";
import { useState, type ChangeEvent } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { CreateRestoreJobParams } from "../../../services/restore";
import { PHOTO_ACCEPT } from "./photoFiles";
import { batchJobParams, batchOffersFaces, hasBatchSteps } from "./restoreBatch";
import { RestoreBatchResults } from "./RestoreBatchResults";
import { DEFAULT_BATCH_RESULT_DEPS, type BatchResultDeps } from "./useBatchJobs";
import { useRestoreBatch, type RestoreBatchDeps } from "./useRestoreBatch";

export interface RestoreBatchSectionProps {
  base: CreateRestoreJobParams;
  deps?: RestoreBatchDeps;
  resultDeps?: BatchResultDeps;
}

const INPUT_ID = "restore-batch-input";

function BatchStatus({ pending, sent, failed }: { pending: number; sent: number; failed: number }) {
  const { t } = useTranslation();
  if (pending > 0) {
    return <p role="status" className="text-xs text-text-dim">{t("restore.batch.pending", { count: pending })}</p>;
  }
  return (
    <>
      {sent > 0 && (
        <p role="status" className="text-xs text-text-dim">
          {t(sent === 1 ? "restore.batch.sentOne" : "restore.batch.sent", { count: sent })}
        </p>
      )}
      {failed > 0 && (
        <p role="alert" className="text-xs text-danger">
          {t(failed === 1 ? "restore.batch.failedOne" : "restore.batch.failed", { count: failed })}
        </p>
      )}
    </>
  );
}

function BatchFacesChoice({ checked, onChange }: { checked: boolean; onChange: (checked: boolean) => void }) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-1.5 rounded border border-border bg-surface-2 p-2.5">
      <label className="inline-flex items-center gap-2 text-xs font-medium text-text">
        <input
          type="checkbox"
          checked={checked}
          onChange={(event) => onChange(event.target.checked)}
          className="h-3.5 w-3.5 shrink-0 cursor-pointer accent-accent"
        />
        {t("restore.batch.faces.toggle")}
      </label>
      {checked ? (
        <>
          <p className="text-xs leading-snug text-text-dim">{t("restore.batch.faces.hint")}</p>
          <p className="flex items-start gap-1.5 text-xs leading-snug text-warn">
            <TriangleAlert aria-hidden="true" className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
            <span>{t("restore.warning.faces")}</span>
          </p>
        </>
      ) : (
        <p className="text-xs text-text-dim">{t("restore.batch.noFaces")}</p>
      )}
    </div>
  );
}

export function RestoreBatchSection({ base, deps, resultDeps = DEFAULT_BATCH_RESULT_DEPS }: RestoreBatchSectionProps) {
  const { t } = useTranslation();
  const batch = useRestoreBatch(deps);
  const [withFaces, setWithFaces] = useState(false);
  const offersFaces = batchOffersFaces(base);
  const runnable = hasBatchSteps(base, withFaces);

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    // Vaciar el input deja volver a elegir las mismas fotos.
    event.target.value = "";
    if (files.length > 0) {
      batch.submitMany(batchJobParams(base, files, withFaces));
    }
  }

  return (
    <div className="flex flex-col gap-2">
      {offersFaces && <BatchFacesChoice checked={withFaces} onChange={setWithFaces} />}
      <label
        htmlFor={INPUT_ID}
        aria-disabled={!runnable}
        className="inline-flex w-fit cursor-pointer items-center gap-1.5 rounded border border-border px-3 py-1.5 text-xs text-text transition hover:border-accent hover:text-accent focus-within:outline focus-within:outline-2 focus-within:outline-accent aria-disabled:cursor-not-allowed aria-disabled:opacity-40 aria-disabled:hover:border-border aria-disabled:hover:text-text"
      >
        <Images aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
        {t("restore.batch.apply")}
        <input
          id={INPUT_ID}
          type="file"
          multiple
          accept={PHOTO_ACCEPT}
          onChange={handleChange}
          disabled={batch.pending > 0 || !runnable}
          className="sr-only"
        />
      </label>
      <p className="text-xs text-text-dim">{t(runnable ? "restore.batch.hint" : "restore.batch.facesOnly")}</p>
      <BatchStatus pending={batch.pending} sent={batch.sent} failed={batch.failed} />
      {batch.entries.length > 0 && <RestoreBatchResults entries={batch.entries} deps={resultDeps} />}
    </div>
  );
}
