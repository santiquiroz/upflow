import { useTranslation } from "../../i18n/LocaleProvider";
import { EditorPanel } from "./EditorPanel";

export function EditorPage() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="font-heading text-2xl font-semibold text-text">{t("editor.title")}</h1>
        <p className="mt-1 text-sm text-text-dim">{t("editor.description")}</p>
      </div>
      <EditorPanel />
    </div>
  );
}
