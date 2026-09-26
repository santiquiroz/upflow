import { useTranslation } from "../../../i18n/LocaleProvider";

export function CctvModeToggle({ checked, onChange }: { checked: boolean; onChange: (checked: boolean) => void }) {
  const { t } = useTranslation();
  return (
    <label className="flex w-fit items-center gap-2 text-sm font-medium text-text">
      <input
        type="checkbox"
        role="switch"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="h-3.5 w-3.5 accent-accent"
      />
      {t("cctv.toggle")}
    </label>
  );
}
