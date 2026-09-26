import { Wand2 } from "lucide-react";
import { useId, useState } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { RestoreAnalysis, RestoreCapabilities, StepEta } from "../../../lib/restoreApiTypes";
import { findingText } from "./diagnosisText";
import { RestoreChainPanel } from "./RestoreChainPanel";
import { durationText, summaryKey } from "./restoreSteps";
import type { RestoreSelection } from "./useRestoreSelection";

interface RestoreSummaryProps {
  analysis: RestoreAnalysis;
  capabilities: RestoreCapabilities;
  selection: RestoreSelection;
  canRestore: boolean;
  onRestore: () => void;
}

function DiagnosisFindings({ analysis }: { analysis: RestoreAnalysis }) {
  const { t } = useTranslation();
  const { findings } = analysis.diagnosis;
  if (findings.length === 0) {
    return null;
  }
  return (
    <ul aria-label={t("restore.diag.title")} className="flex flex-wrap gap-1.5">
      {findings.map((finding, index) => {
        const text = findingText(finding, t);
        return (
          <li
            key={`${finding.key}-${index}`}
            className="rounded-sm border border-border bg-surface px-2 py-1 text-xs text-text-dim"
          >
            {t(text.key, text.params)}
          </li>
        );
      })}
    </ul>
  );
}

function EtaLine({ eta }: { eta: StepEta }) {
  const { t } = useTranslation();
  if (eta.cpuSeconds <= 0 && eta.gpuSeconds <= 0) {
    return null;
  }
  const gpu = durationText(eta.gpuSeconds);
  const cpu = durationText(eta.cpuSeconds);
  return (
    <p className="text-xs text-text-faint" title={t("restore.eta.hint")}>
      {t("restore.eta", { gpu: t(gpu.key, { count: gpu.count }), cpu: t(cpu.key, { count: cpu.count }) })}
    </p>
  );
}

export function RestoreSummary({
  analysis,
  capabilities,
  selection,
  canRestore,
  onRestore,
}: RestoreSummaryProps) {
  const { t } = useTranslation();
  const [expanded, setExpanded] = useState(false);
  const panelId = useId();
  const count = selection.enabledIds.length;

  return (
    <section className="flex flex-col gap-3">
      <DiagnosisFindings analysis={analysis} />
      <div className="flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={onRestore}
          disabled={!canRestore || count === 0}
          className="inline-flex w-fit items-center gap-2 rounded bg-accent px-4 py-2 text-sm font-medium text-bg transition-[background-color,opacity] duration-fast hover:bg-accent-hover active:bg-accent-press disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          <Wand2 aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
          {t("restore.action.start")}
        </button>
        <p className="text-sm text-text">
          {t(summaryKey(count), { count })}
          <span aria-hidden="true" className="px-1.5 text-text-faint">
            ·
          </span>
          <button
            type="button"
            aria-expanded={expanded}
            aria-controls={panelId}
            onClick={() => setExpanded((open) => !open)}
            className="text-accent underline-offset-2 hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
          >
            {t("restore.summary.customize")}
          </button>
        </p>
      </div>
      <EtaLine eta={selection.eta} />
      <div id={panelId} hidden={!expanded}>
        <RestoreChainPanel
          capabilities={capabilities}
          suggestedPresets={analysis.diagnosis.suggestedPresets}
          selection={selection}
        />
      </div>
    </section>
  );
}
