import { Cctv } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";

const BUTTON_CLASS =
  "rounded-sm border px-3 py-1 text-xs transition-[background-color,border-color,color] duration-fast focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";

export function CctvSuggestBanner({ onAccept }: { onAccept: () => void }) {
  const { t } = useTranslation();
  const [dismissed, setDismissed] = useState(false);
  if (dismissed) {
    return null;
  }
  return (
    <div role="status" className="flex flex-wrap items-center gap-3 rounded border border-accent bg-surface-2 p-3">
      <Cctv aria-hidden="true" className="h-4 w-4 text-accent" strokeWidth={1.75} />
      <span className="flex-1 text-sm text-text">{t("cctv.suggestMode")}</span>
      <button type="button" onClick={onAccept} className={`${BUTTON_CLASS} border-accent bg-accent text-bg hover:bg-accent-hover`}>
        {t("cctv.suggest.accept")}
      </button>
      <button
        type="button"
        onClick={() => setDismissed(true)}
        className={`${BUTTON_CLASS} border-border bg-surface text-text-dim hover:text-text`}
      >
        {t("cctv.suggest.dismiss")}
      </button>
    </div>
  );
}
