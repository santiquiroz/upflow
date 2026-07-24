import { AlertTriangle, Loader2 } from "lucide-react";
import { DeterminateProgressBar } from "../../components/DeterminateProgressBar";
import { IndeterminateProgressBar } from "../../components/IndeterminateProgressBar";
import type { ModelInstallPhase } from "../../hooks/useModels";

const IN_FLIGHT_PHASES: readonly ModelInstallPhase[] = ["starting", "downloading", "validating", "converting"];

export function isInstallInFlight(phase: ModelInstallPhase): boolean {
  return IN_FLIGHT_PHASES.includes(phase);
}

export function installPhaseLabel(phase: ModelInstallPhase): string {
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

export function InstallProgress({ phase, progressPct }: { phase: ModelInstallPhase; progressPct: number | null }) {
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

export function InstallError({ message, onRetry }: { message: string; onRetry: () => void }) {
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
