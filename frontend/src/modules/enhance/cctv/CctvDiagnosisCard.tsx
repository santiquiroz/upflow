import { useTranslation } from "../../../i18n/LocaleProvider";
import type { CctvAnalysis } from "../../../services/cctv";
import { diagnosisRows, diagnosisWarnings } from "./cctvDiagnosis";

function SourceHash({ sha256 }: { sha256: string }) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[11px] text-text-faint">{t("cctv.hash.recorded")}</span>
      <code className="font-mono-tabular break-all text-xs text-text">{sha256}</code>
    </div>
  );
}

function WarningList({ warnings }: { warnings: string[] }) {
  if (warnings.length === 0) {
    return null;
  }
  return (
    <ul role="status" className="flex flex-col gap-1 rounded border border-warn bg-surface-2 p-2">
      {warnings.map((text) => (
        <li key={text} className="text-xs text-text">
          {text}
        </li>
      ))}
    </ul>
  );
}

export function CctvDiagnosisCard({ analysis }: { analysis: CctvAnalysis }) {
  const { t } = useTranslation();
  return (
    <section aria-labelledby="cctv-diagnosis-title" className="flex flex-col gap-3 rounded border border-border bg-surface p-4">
      <SourceHash sha256={analysis.sourceSha256} />
      <h3 id="cctv-diagnosis-title" className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
        {t("cctv.diag.title")}
      </h3>
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
        {diagnosisRows(t, analysis).map((row) => (
          <div key={row.labelKey} className="contents">
            <dt className="text-text-dim">{t(row.labelKey)}</dt>
            <dd className="font-mono-tabular text-text">{row.value}</dd>
          </div>
        ))}
      </dl>
      <WarningList warnings={diagnosisWarnings(t, analysis)} />
    </section>
  );
}
