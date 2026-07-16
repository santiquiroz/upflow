import { ArrowUpCircle, X } from "lucide-react";
import { useState } from "react";
import { useUpdateCheck } from "../hooks/useUpdateCheck";

// Per-version dismissal: storing the exact latestVersion means a dismissed
// version never shows again, but a newer latestVersion (a different string)
// stops matching and the banner returns.
const DISMISS_STORAGE_KEY = "upflow.dismissedUpdate";

function readDismissedVersion(): string | null {
  try {
    return localStorage.getItem(DISMISS_STORAGE_KEY);
  } catch {
    return null;
  }
}

function persistDismissedVersion(version: string): void {
  try {
    localStorage.setItem(DISMISS_STORAGE_KEY, version);
  } catch {
    // localStorage may be unavailable (private mode / quota); the session-local
    // state below still hides the banner even when it cannot be persisted.
  }
}

export function UpdateBanner() {
  const { data } = useUpdateCheck();
  const [dismissedVersion, setDismissedVersion] = useState<string | null>(readDismissedVersion);

  if (!data || !data.updateAvailable || data.error || !data.latestVersion) {
    return null;
  }

  const latestVersion = data.latestVersion;
  if (dismissedVersion === latestVersion) {
    return null;
  }

  function dismiss(): void {
    persistDismissedVersion(latestVersion);
    setDismissedVersion(latestVersion);
  }

  return (
    <div
      role="status"
      aria-label="Update available"
      className="flex items-center gap-3 border-b border-border bg-surface-2 px-4 py-2 text-sm font-body text-text"
    >
      <ArrowUpCircle aria-hidden="true" className="h-4 w-4 shrink-0 text-accent" strokeWidth={1.75} />
      <span className="flex-1 min-w-0 truncate">
        New version <span className="font-mono-tabular text-accent">{latestVersion}</span> available
      </span>
      {data.releaseUrl && (
        <a
          href={data.releaseUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="shrink-0 rounded-sm px-2 py-1 font-medium text-accent transition-colors duration-fast hover:text-accent-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          View release
        </a>
      )}
      <button
        type="button"
        aria-label="Dismiss update notification"
        onClick={dismiss}
        className="flex h-6 w-6 shrink-0 items-center justify-center rounded-sm text-text-faint transition-colors duration-fast hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        <X aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
      </button>
    </div>
  );
}
