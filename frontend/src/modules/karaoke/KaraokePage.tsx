import { useTranslation } from "../../i18n/LocaleProvider";
import { KaraokeStudioPanel } from "./KaraokeStudioPanel";

export function KaraokePage() {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="font-heading text-2xl font-semibold text-text">
          {t("karaoke.page.title")}
        </h1>
        <p className="mt-1 text-sm text-text-dim">{t("karaoke.page.description")}</p>
      </div>
      <KaraokeStudioPanel />
    </div>
  );
}
