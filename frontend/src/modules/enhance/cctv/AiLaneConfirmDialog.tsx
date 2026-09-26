import { Modal } from "../../../components/Modal";
import { useTranslation } from "../../../i18n/LocaleProvider";

const TITLE_ID = "cctv-ai-confirm-title";

export function AiLaneConfirmDialog({ onConfirm, onCancel }: { onConfirm: () => void; onCancel: () => void }) {
  const { t } = useTranslation();
  return (
    <Modal titleId={TITLE_ID} onClose={onCancel} widthClassName="max-w-md">
      <h2 id={TITLE_ID} className="font-heading text-base font-semibold text-text">
        {t("cctv.ai.confirm.title")}
      </h2>
      <p className="text-sm text-text">{t("cctv.ai.confirm")}</p>
      <div className="flex justify-end gap-2">
        <button
          type="button"
          onClick={onCancel}
          className="rounded border border-border bg-surface px-3 py-1.5 text-sm text-text hover:border-text-faint focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          {t("cctv.ai.confirm.cancel")}
        </button>
        <button
          type="button"
          onClick={onConfirm}
          className="rounded bg-warn px-3 py-1.5 text-sm font-medium text-bg focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          {t("cctv.ai.confirm.continue")}
        </button>
      </div>
    </Modal>
  );
}
