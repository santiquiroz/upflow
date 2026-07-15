import { AlertTriangle, CheckCircle2, Download, Heart, Loader2 } from "lucide-react";
import { DEFAULT_INSTALL_POLL_INTERVAL_MS, useModelInstall, type ModelInstallPhase } from "../../hooks/useModels";
import type { HfModelSearchResultResponse } from "../../lib/apiTypes";

interface HfResultCardProps {
  result: HfModelSearchResultResponse;
  pollIntervalMs?: number;
}

const IN_FLIGHT_PHASES: readonly ModelInstallPhase[] = ["starting", "downloading", "validating", "converting"];

function isInstallInFlight(phase: ModelInstallPhase): boolean {
  return IN_FLIGHT_PHASES.includes(phase);
}

function installPhaseLabel(phase: ModelInstallPhase): string {
  switch (phase) {
    case "starting":
      return "Starting install…";
    case "downloading":
      return "Downloading…";
    case "validating":
      return "Validating…";
    case "converting":
      return "Converting…";
    default:
      return "Working…";
  }
}

function formatCount(count: number): string {
  return count.toLocaleString("en-US");
}

function DeterminateProgressBar({ label, percent }: { label: string; percent: number }) {
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuenow={Math.round(percent)}
      aria-valuemin={0}
      aria-valuemax={100}
      className="h-1.5 w-full overflow-hidden rounded-sm bg-surface-2"
    >
      <div
        className="h-full rounded-sm bg-accent transition-[width] duration-normal"
        style={{ width: `${percent}%` }}
      />
    </div>
  );
}

function IndeterminateProgressBar({ label }: { label: string }) {
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-busy="true"
      className="h-1.5 w-full overflow-hidden rounded-sm bg-surface-2"
    >
      <div className="job-progress-bar h-full w-1/3 rounded-sm bg-accent" />
    </div>
  );
}

function InstallProgress({ phase, progressPct }: { phase: ModelInstallPhase; progressPct: number | null }) {
  const label = installPhaseLabel(phase);
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-2 text-sm text-text">
        <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin text-accent" strokeWidth={1.75} />
        <span>{label}</span>
        {progressPct !== null && (
          <span className="font-mono-tabular text-text-dim">{Math.round(progressPct)}%</span>
        )}
      </div>
      {progressPct !== null ? (
        <DeterminateProgressBar label={label} percent={progressPct} />
      ) : (
        <IndeterminateProgressBar label={label} />
      )}
    </div>
  );
}

function InstalledIndicator() {
  return (
    <div className="flex items-center gap-2 text-sm text-ok">
      <CheckCircle2 aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
      <span>Installed</span>
    </div>
  );
}

function InstallError({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="flex flex-col gap-2">
      <div
        role="alert"
        className="flex items-start gap-2 rounded border border-danger bg-surface-2 px-3 py-2 text-sm text-danger"
      >
        <AlertTriangle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" strokeWidth={1.75} />
        <span>{message}</span>
      </div>
      <button
        type="button"
        onClick={onRetry}
        className="w-fit rounded-sm border border-border bg-surface px-3 py-1.5 text-sm text-text-dim transition-[border-color,color] duration-fast hover:border-text-faint hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        Try again
      </button>
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
