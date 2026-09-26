import { TriangleAlert } from "lucide-react";
import { useTranslation } from "../../../i18n/LocaleProvider";

export function AiLaneBanner() {
  const { t } = useTranslation();
  return (
    <div role="note" className="flex items-start gap-2 rounded border border-warn bg-surface-2 p-2 text-xs text-text">
      <TriangleAlert aria-hidden="true" className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warn" strokeWidth={1.75} />
      <div className="flex flex-col gap-1">
        <p>{t("cctv.ai.banner")}</p>
        <p>{t("cctv.plates.noAi")}</p>
      </div>
    </div>
  );
}
