import { Images } from "lucide-react";
import type { ChangeEvent } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { CreateRestoreJobParams } from "../../../services/restore";
import { PHOTO_ACCEPT } from "./photoFiles";
import { batchJobParams, batchSkipsFaces } from "./restoreBatch";
import { useRestoreBatch, type RestoreBatchDeps } from "./useRestoreBatch";

export interface RestoreBatchSectionProps {
  base: CreateRestoreJobParams;
  deps?: RestoreBatchDeps;
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

export function RestoreBatchSection({ base, deps }: RestoreBatchSectionProps) {
  const { t } = useTranslation();
  const batch = useRestoreBatch(deps);

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    // Vaciar el input deja volver a elegir las mismas fotos.
    event.target.value = "";
    if (files.length > 0) {
      batch.submitMany(batchJobParams(base, files));
    }
  }

  return (
    <div className="flex flex-col gap-1.5">
      <label
        htmlFor={INPUT_ID}
        className="inline-flex w-fit cursor-pointer items-center gap-1.5 rounded border border-border px-3 py-1.5 text-xs text-text transition hover:border-accent hover:text-accent focus-within:outline focus-within:outline-2 focus-within:outline-accent"
      >
        <Images aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
        {t("restore.batch.apply")}
        <input
          id={INPUT_ID}
          type="file"
          multiple
          accept={PHOTO_ACCEPT}
          onChange={handleChange}
          disabled={batch.pending > 0}
          className="sr-only"
        />
      </label>
      <p className="text-xs text-text-dim">{t("restore.batch.hint")}</p>
      {batchSkipsFaces(base) && <p className="text-xs text-text-dim">{t("restore.batch.noFaces")}</p>}
      <BatchStatus pending={batch.pending} sent={batch.sent} failed={batch.failed} />
    </div>
  );
}
