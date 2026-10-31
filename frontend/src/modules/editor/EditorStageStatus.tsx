import { useTranslation } from "../../i18n/LocaleProvider";

interface EditorStageStatusProps {
  segmenting: boolean;
  applying: boolean;
  healing: boolean;
}

const STATUS_CLASS = "absolute bottom-2 left-2 rounded bg-surface px-2 py-1 text-xs text-text-dim";

export function EditorStageStatus({ segmenting, applying, healing }: EditorStageStatusProps) {
  const { t } = useTranslation();
  return (
    <>
      {segmenting && <p className={STATUS_CLASS}>{t("editor.segmenting")}</p>}
      {applying && <p className={STATUS_CLASS}>{t("editor.retouch.applying")}</p>}
      {healing && <p className={STATUS_CLASS}>{t("editor.heal.healing")}</p>}
    </>
  );
}
