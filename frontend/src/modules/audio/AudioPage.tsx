import { useTranslation } from "../../i18n/LocaleProvider";
import { AudioPanel } from "./AudioPanel";

export function AudioPage() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="font-heading text-2xl font-semibold text-text">Audio</h1>
        <p className="mt-1 text-sm text-text-dim">{t("audio.page.subtitle")}</p>
      </div>
      <AudioPanel />
    </div>
  );
}
