import { ArrowLeft } from "lucide-react";
import { useTranslation } from "../../i18n/LocaleProvider";

interface PeopleSummaryProps {
  personIds: number[];
  numbers: Record<number, number>;
  failed: number;
  onBack: () => void;
}

function BatchDetails({ personIds, numbers, onBack }: Omit<PeopleSummaryProps, "failed">) {
  const { t } = useTranslation();
  const sorted = [...personIds].sort((a, b) => (numbers[a] ?? 0) - (numbers[b] ?? 0));
  return (
    <>
      <h2 className="text-sm font-semibold text-text">{t("editor.people.batchTitle", { count: personIds.length })}</h2>
      <ul aria-label={t("editor.people.batchPeople")} className="mt-2 flex flex-wrap gap-1">
        {sorted.map((id) => (
          <li key={id} className="rounded-full border border-border bg-bg px-2 py-0.5 text-[11px] text-text">
            {t("editor.people.person", { number: numbers[id] ?? id })}
          </li>
        ))}
      </ul>
      <p className="mt-2 text-xs text-text-dim">{t("editor.people.polishHint")}</p>
      <button
        type="button"
        onClick={onBack}
        className="mt-2 flex min-h-9 items-center gap-1.5 rounded border border-border px-3 text-xs text-text hover:bg-bg"
      >
        <ArrowLeft aria-hidden="true" className="h-3.5 w-3.5" />
        {t("editor.people.backToChoosing")}
      </button>
    </>
  );
}

export function PeopleSummary({ personIds, numbers, failed, onBack }: PeopleSummaryProps) {
  const { t } = useTranslation();
  if (personIds.length === 0 && failed === 0) return null;
  return (
    <section className="mb-3 rounded-lg border border-border bg-surface-2 p-3">
      {personIds.length > 0 && <BatchDetails personIds={personIds} numbers={numbers} onBack={onBack} />}
      <p aria-live="polite" className={failed > 0 ? "mt-2 text-xs text-text-dim" : "sr-only"}>
        {failed > 0 ? t("editor.people.failed", { count: failed }) : ""}
      </p>
    </section>
  );
}
