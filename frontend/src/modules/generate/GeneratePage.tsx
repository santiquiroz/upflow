import { useTranslation } from "../../i18n/LocaleProvider";
import { GeneratePanel } from "./GeneratePanel";

export function GeneratePage() {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="font-heading text-2xl font-semibold text-text">
          {t("generation.page.title")}
        </h1>
        <p className="mt-1 text-sm text-text-dim">
          {t("generation.page.description")}
        </p>
      </div>
      <GeneratePanel />
    </div>
  );
}
