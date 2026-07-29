import { Languages } from "lucide-react";
import { LOCALES, type Locale } from "../../i18n";
import { useTranslation } from "../../i18n/LocaleProvider";

// Cada idioma se nombra en si mismo, no traducido: quien busca su idioma en una
// pantalla que no entiende reconoce "Español", no "Spanish".
const ENDONYMS: Record<Locale, string> = {
  es: "Español",
  en: "English",
};

export function LanguageSection() {
  const { locale, setLocale, t } = useTranslation();

  return (
    <div className="flex flex-col gap-3 rounded border border-border bg-surface p-4">
      <h2 className="flex items-center gap-1.5 font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
        <Languages aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
        {t("settings.language.title")}
      </h2>
      <div
        className="flex gap-1 rounded border border-border bg-surface-2 p-1"
        role="radiogroup"
        aria-label={t("settings.language.title")}
      >
        {LOCALES.map((option) => {
          const active = option === locale;
          return (
            <button
              key={option}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => setLocale(option)}
              className={`flex-1 rounded px-3 py-1.5 text-sm transition-colors ${
                active
                  ? "bg-accent font-medium text-surface"
                  : "text-text-dim hover:bg-surface hover:text-text"
              }`}
            >
              {ENDONYMS[option]}
            </button>
          );
        })}
      </div>
      <p className="text-xs text-text-faint">{t("settings.language.description")}</p>
    </div>
  );
}
