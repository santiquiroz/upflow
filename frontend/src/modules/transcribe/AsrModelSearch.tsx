import { CheckCircle2, Download, Search } from "lucide-react";
import { useState } from "react";
import {
  CompatFilterChips,
  matchesCompatFilter,
  type CompatFilter,
} from "../models/compatFilter";
import { DeterminateProgressBar } from "../../components/DeterminateProgressBar";
import { IndeterminateProgressBar } from "../../components/IndeterminateProgressBar";
import {
  DEFAULT_ASR_INSTALL_POLL_INTERVAL_MS,
  useAsrModelInstall,
  useAsrModelSearchResults,
  type AsrInstallPhase,
} from "../../hooks/useTranscribeJob";
import { useDebouncedValue } from "../../hooks/useDebouncedValue";
import { useTranslation } from "../../i18n/LocaleProvider";
import type {
  CompatVerdict,
  HfModelSearchResultResponse,
} from "../../lib/apiTypes";

const DEFAULT_SEARCH_DEBOUNCE_MS = 300;

interface AsrModelSearchProps {
  debounceMs?: number;
  installPollIntervalMs?: number;
}

function compatLabelKey(verdict: CompatVerdict | null): string {
  switch (verdict) {
    case "ready_onnx":
      return "transcribe.models.compat.readyOnnx";
    case "needs_conversion":
      return "transcribe.models.compat.needsConversion";
    case "gated":
      return "transcribe.models.compat.gated";
    case "incompatible":
      return "transcribe.models.compat.incompatible";
    default:
      return "transcribe.models.compat.unknown";
  }
}

function compatTone(verdict: CompatVerdict | null): string {
  if (verdict === "ready_onnx") {
    return "border-ok text-ok";
  }
  if (verdict === "incompatible") {
    return "border-danger text-danger";
  }
  return "border-warn text-warn";
}

function installPhaseLabelKey(phase: AsrInstallPhase): string {
  switch (phase) {
    case "starting":
      return "transcribe.models.install.starting";
    case "queued":
      return "transcribe.models.install.queued";
    case "downloading":
      return "transcribe.models.install.downloading";
    default:
      return "transcribe.models.install.working";
  }
}

function isInstalling(phase: AsrInstallPhase): boolean {
  return phase === "starting" || phase === "queued" || phase === "downloading";
}

function AsrModelCard({
  result,
  pollIntervalMs,
}: {
  result: HfModelSearchResultResponse;
  pollIntervalMs: number;
}) {
  const { t } = useTranslation();
  const { phase, progressPct, errorMessage, install, reset } =
    useAsrModelInstall(pollIntervalMs);
  const phaseLabel = t(installPhaseLabelKey(phase));

  return (
    <article className="flex flex-col gap-3 rounded border border-border bg-surface p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="break-all text-sm font-medium text-text">{result.id}</h3>
          {result.author && (
            <p className="mt-1 text-xs text-text-faint">
              {t("transcribe.models.author", { author: result.author })}
            </p>
          )}
        </div>
        {phase === "idle" && (
          <button
            type="button"
            onClick={() => install(result.id)}
            className="inline-flex shrink-0 items-center gap-2 rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg transition-[background-color] duration-fast hover:bg-accent-hover active:bg-accent-press focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
          >
            <Download aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
            {t("transcribe.models.install")}
          </button>
        )}
      </div>

      <dl className="flex flex-wrap gap-4 text-xs text-text-dim">
        <div className="flex gap-1">
          <dt>{t("transcribe.models.downloads")}</dt>
          <dd className="font-mono-tabular text-text">
            {result.downloads.toLocaleString("en-US")}
          </dd>
        </div>
        <div className="flex gap-1">
          <dt>{t("transcribe.models.likes")}</dt>
          <dd className="font-mono-tabular text-text">
            {result.likes.toLocaleString("en-US")}
          </dd>
        </div>
      </dl>

      <span
        className={`w-fit rounded-sm border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${compatTone(result.compat)}`}
      >
        {t(compatLabelKey(result.compat))}
      </span>

      {result.compatReasonKey && (
        <p className="text-xs text-text-dim">
          {t(result.compatReasonKey, result.compatReasonParams)}
        </p>
      )}

      {isInstalling(phase) && (
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center justify-between gap-2 text-sm text-text">
            <span>{phaseLabel}</span>
            {progressPct !== null && (
              <span className="font-mono-tabular text-text-dim">
                {Math.round(progressPct)}%
              </span>
            )}
          </div>
          {progressPct === null ? (
            <IndeterminateProgressBar label={phaseLabel} />
          ) : (
            <DeterminateProgressBar
              label={phaseLabel}
              percent={progressPct}
            />
          )}
        </div>
      )}

      {phase === "installed" && (
        <div className="flex items-center gap-2 text-sm text-ok">
          <CheckCircle2
            aria-hidden="true"
            className="h-4 w-4"
            strokeWidth={1.75}
          />
          <span>{t("transcribe.models.installed")}</span>
        </div>
      )}

      {phase === "error" && (
        <div className="flex flex-col gap-2">
          <p
            role="alert"
            className="rounded border border-danger bg-surface-2 px-3 py-2 text-sm text-danger"
          >
            {errorMessage ?? t("transcribe.models.install.failedFallback")}
          </p>
          <button
            type="button"
            onClick={reset}
            className="w-fit rounded-sm border border-border bg-surface px-3 py-1.5 text-sm text-text-dim hover:border-text-faint hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
          >
            {t("transcribe.models.retry")}
          </button>
        </div>
      )}
    </article>
  );
}

function SearchResults({
  query,
  installPollIntervalMs,
  filter,
}: {
  query: string;
  installPollIntervalMs: number;
  filter: CompatFilter;
}) {
  const { t } = useTranslation();
  const searchQuery = useAsrModelSearchResults(query);

  if (searchQuery.isLoading) {
    return (
      <p role="status" className="text-sm text-text-dim">
        {t("transcribe.models.searching")}
      </p>
    );
  }

  if (searchQuery.isError) {
    return (
      <p
        role="alert"
        className="rounded border border-danger bg-surface px-3 py-2 text-sm text-danger"
      >
        {t("transcribe.models.searchFailed")}
      </p>
    );
  }

  const results = (searchQuery.data?.results ?? []).filter((result) =>
    matchesCompatFilter(result, filter),
  );
  if (results.length === 0) {
    return (
      <p className="text-sm text-text-faint">
        {t("transcribe.models.noResults")}
      </p>
    );
  }

  return (
    <ul className="grid grid-cols-2 gap-3 max-[900px]:grid-cols-1">
      {results.map((result) => (
        <li key={result.id}>
          <AsrModelCard
            result={result}
            pollIntervalMs={installPollIntervalMs}
          />
        </li>
      ))}
    </ul>
  );
}

export function AsrModelSearch({
  debounceMs = DEFAULT_SEARCH_DEBOUNCE_MS,
  installPollIntervalMs = DEFAULT_ASR_INSTALL_POLL_INTERVAL_MS,
}: AsrModelSearchProps) {
  const { t } = useTranslation();
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<CompatFilter>("all");
  const debouncedQuery = useDebouncedValue(query, debounceMs).trim();

  return (
    <section className="flex flex-col gap-4">
      <div>
        <h2 className="font-heading text-lg font-semibold text-text">
          {t("transcribe.models.title")}
        </h2>
        <p className="mt-1 text-sm text-text-dim">
          {t("transcribe.models.description")}
        </p>
      </div>
      <label className="relative block">
        <span className="sr-only">{t("transcribe.models.searchLabel")}</span>
        <Search
          aria-hidden="true"
          className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-faint"
          strokeWidth={1.75}
        />
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder={t("transcribe.models.searchPlaceholder")}
          className="w-full rounded border border-border bg-surface py-2 pl-9 pr-3 text-sm text-text placeholder:text-text-faint focus:border-accent focus:outline-none"
        />
      </label>
      {debouncedQuery.length > 0 && (
        <CompatFilterChips value={filter} onChange={setFilter} name="asr-compat-filter" />
      )}
      <SearchResults
        query={debouncedQuery}
        installPollIntervalMs={installPollIntervalMs}
        filter={filter}
      />
    </section>
  );
}
