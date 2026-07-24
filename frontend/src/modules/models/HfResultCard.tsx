import { CheckCircle2, Download, Heart } from "lucide-react";
import { DEFAULT_INSTALL_POLL_INTERVAL_MS, useModelInstall } from "../../hooks/useModels";
import type { HfModelSearchResultResponse } from "../../lib/apiTypes";
import { InstallError, InstallProgress, isInstallInFlight } from "./installUi";

interface HfResultCardProps {
  result: HfModelSearchResultResponse;
  pollIntervalMs?: number;
}

function formatCount(count: number): string {
  return count.toLocaleString("en-US");
}

function InstalledIndicator() {
  return (
    <div className="flex items-center gap-2 text-sm text-ok">
      <CheckCircle2 aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
      <span>Installed</span>
    </div>
  );
}

function InstallButton({ onInstall }: { onInstall: () => void }) {
  return (
    <button
      type="button"
      onClick={onInstall}
      className="inline-flex shrink-0 items-center gap-2 rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg transition-[background-color] duration-fast hover:bg-accent-hover active:bg-accent-press focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
    >
      <Download aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
      Install
    </button>
  );
}

function ResultMeta({ result }: { result: HfModelSearchResultResponse }) {
  return (
    <dl className="flex flex-wrap items-center gap-4 text-xs text-text-dim">
      <div className="flex items-center gap-1">
        <dt className="text-text-faint">Downloads</dt>
        <dd className="font-mono-tabular text-text">{formatCount(result.downloads)}</dd>
      </div>
      <div className="flex items-center gap-1">
        <Heart aria-hidden="true" className="h-3.5 w-3.5 text-text-faint" strokeWidth={1.75} />
        <dd className="font-mono-tabular text-text">{formatCount(result.likes)}</dd>
      </div>
      {result.pipelineTag && (
        <dd className="rounded-sm bg-surface-2 px-1.5 py-0.5 text-text-dim">{result.pipelineTag}</dd>
      )}
    </dl>
  );
}

export function HfResultCard({ result, pollIntervalMs = DEFAULT_INSTALL_POLL_INTERVAL_MS }: HfResultCardProps) {
  const { phase, progressPct, errorMessage, install, reset } = useModelInstall(pollIntervalMs);

  return (
    <div className="flex flex-col gap-3 rounded border border-border bg-surface p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-col gap-1">
          <span className="text-sm font-medium text-text">{result.id}</span>
          {result.author && <span className="text-xs text-text-faint">{result.author}</span>}
        </div>
        {phase === "idle" && <InstallButton onInstall={() => install(result.id)} />}
      </div>
      <ResultMeta result={result} />
      {isInstallInFlight(phase) && <InstallProgress phase={phase} progressPct={progressPct} />}
      {phase === "installed" && <InstalledIndicator />}
      {phase === "error" && errorMessage && <InstallError message={errorMessage} onRetry={reset} />}
    </div>
  );
}
