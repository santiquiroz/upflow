import { useTranslation } from "../../i18n/LocaleProvider";
import { TranscribePanel } from "./TranscribePanel";

export function TranscribePage() {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="font-heading text-2xl font-semibold text-text">
          {t("transcribe.page.title")}
        </h1>
        <p className="mt-1 text-sm text-text-dim">
          {t("transcribe.page.description")}
        </p>
      </div>
      <TranscribePanel />
    </div>
  );
}
