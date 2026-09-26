import { useId } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import { coverageText, DamageMaskEditor, type DamageMaskEditorProps } from "./DamageMaskEditor";

export interface DamageMaskSectionProps extends DamageMaskEditorProps {
  open: boolean;
  onToggle: () => void;
}

function ReviewLine({ needsReview, reviewed }: { needsReview: boolean; reviewed: boolean }) {
  const { t } = useTranslation();
  if (!needsReview) return null;
  if (reviewed) {
    return <p className="text-xs text-ok">{t("restore.mask.reviewed")}</p>;
  }
  return (
    <p role="alert" className="text-xs text-warn">
      {t("restore.mask.reviewRequired")}
    </p>
  );
}

// Fuera del panel plegado: con cobertura alta "Restore" manda a revisarla aqui.
export function DamageMaskSection({ open, onToggle, ...editor }: DamageMaskSectionProps) {
  const { t } = useTranslation();
  const headingId = useId();
  const editorId = useId();
  const { damage } = editor;

  return (
    <section aria-labelledby={headingId} className="flex flex-col gap-2">
      <div className="flex flex-wrap items-baseline gap-3">
        <h3 id={headingId} className="text-sm font-medium text-text">
          {t("restore.mask.title")}
        </h3>
        <p className="font-mono-tabular text-xs text-text-dim">
          {t("restore.mask.coverage", { pct: coverageText(damage.coverage) })}
        </p>
      </div>
      <ReviewLine needsReview={damage.needsReview} reviewed={damage.reviewed} />
      <button
        type="button"
        aria-expanded={open}
        aria-controls={editorId}
        onClick={onToggle}
        className="w-fit text-sm text-accent underline-offset-2 hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        {t("restore.mask.open")}
      </button>
      <div id={editorId}>{open && <DamageMaskEditor {...editor} />}</div>
    </section>
  );
}
