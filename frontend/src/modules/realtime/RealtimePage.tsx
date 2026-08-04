import { ArrowUpRight, Zap } from "lucide-react";
import { useTranslation } from "../../i18n/LocaleProvider";
import { RealtimePanel } from "./RealtimePanel";

const ROADMAP_URL = "https://github.com/santiquiroz/upflow/blob/master/docs/REALTIME_MODULE.md";

// Lo que sigue sin ser viable, dicho sin rodeos. No es un "próximamente": son
// límites reales del ecosistema en Windows, no cosas que falte programar.
const NOT_VIABLE_YET: readonly string[] = [
  "realtime.notViable.frameGen",
  "realtime.notViable.afmf",
  "realtime.notViable.fidelityfx",
];

export function RealtimePage() {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-start gap-3">
        <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded border border-border bg-surface-2 text-accent">
          <Zap aria-hidden="true" className="h-5 w-5" strokeWidth={1.75} />
        </span>
        <div>
          <h1 className="font-heading text-2xl font-semibold text-text">{t("realtime.title")}</h1>
          <p className="mt-1 text-sm text-text-dim">{t("realtime.subtitle")}</p>
        </div>
      </div>

      <RealtimePanel />

      <div className="flex flex-col gap-2 rounded border border-border bg-surface-2 p-4">
        <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
          {t("realtime.notViable.title")}
        </h2>
        <ul className="flex flex-col gap-1.5">
          {NOT_VIABLE_YET.map((key) => (
            <li key={key} className="text-xs text-text-faint">
              {t(key)}
            </li>
          ))}
        </ul>
        <a
          href={ROADMAP_URL}
          target="_blank"
          rel="noreferrer"
          className="inline-flex w-fit items-center gap-1.5 text-xs text-accent transition-colors duration-fast hover:text-accent-hover"
        >
          {t("realtime.techDetail")}
          <ArrowUpRight aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
        </a>
      </div>
    </div>
  );
}
