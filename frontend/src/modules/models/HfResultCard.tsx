import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Download,
  Heart,
} from "lucide-react";
import type { ReactNode } from "react";
import { useState } from "react";
import {
  DEFAULT_INSTALL_POLL_INTERVAL_MS,
  useModelInstall,
  type ModelInstallPhase,
} from "../../hooks/useModels";
import { useUpscalerModelPreflight } from "../../hooks/useUpscalerModelPreflight";
import { useTranslation } from "../../i18n/LocaleProvider";
import type {
  CompatVerdict,
  HfModelSearchResultResponse,
} from "../../lib/apiTypes";
import { InstallError, InstallProgress, isInstallInFlight } from "./installUi";
import { buildUpscalerWarnings } from "./upscalerWarnings";

interface HfResultCardProps {
  result: HfModelSearchResultResponse;
  pollIntervalMs?: number;
}

function formatCount(count: number): string {
  return count.toLocaleString("en-US");
}

function formatGb(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

const UPSCALER_COMPAT_KEYS: Partial<Record<CompatVerdict, string>> = {
  ready_onnx: "upscaler.compat.readyOnnx",
  needs_conversion: "upscaler.compat.needsConversion",
  gated: "upscaler.compat.gated",
  incompatible: "upscaler.compat.incompatible",
};

function UpscalerCompatBadge({
  verdict,
}: {
  verdict: CompatVerdict | null;
}) {
  const { t } = useTranslation();
  const key =
    (verdict && UPSCALER_COMPAT_KEYS[verdict]) ??
    "upscaler.compat.unknown";
  const tone =
    verdict === "ready_onnx"
      ? "border-ok text-ok"
      : verdict === "incompatible"
        ? "border-danger text-danger"
        : "border-warn text-warn";

  return (
    <span
      className={`w-fit rounded-sm border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${tone}`}
    >
      {t(key)}
    </span>
  );
}

export function InstalledIndicator() {
  return (
    <div className="flex items-center gap-2 text-sm text-ok">
      <CheckCircle2 aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
      <span>Installed</span>
    </div>
  );
}

export function InstallButton({
  onInstall,
  disabled = false,
  title,
}: {
  onInstall: () => void;
  // Un repo de archivo único no se puede instalar hasta saber qué checkpoint usar.
  // Bloquear el botón dice eso ANTES de intentarlo; dejarlo activo producía un fallo
  // cuyo mensaje no explicaba qué hacer.
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      type="button"
      onClick={onInstall}
      disabled={disabled}
      title={title}
      className="inline-flex shrink-0 items-center gap-2 rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg transition-[background-color] duration-fast hover:bg-accent-hover active:bg-accent-press focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-50"
    >
      <Download aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
      Install
    </button>
  );
}

export function ResultMeta({ result }: { result: HfModelSearchResultResponse }) {
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

interface ResultCardLayoutProps {
  result: HfModelSearchResultResponse;
  phase: ModelInstallPhase;
  progressPct: number | null;
  stageLabel?: string | null;
  errorMessage: string | null;
  onInstall: () => void;
  onReset: () => void;
  children?: ReactNode;
}

export function ResultCardLayout({
  result,
  phase,
  progressPct,
  stageLabel,
  errorMessage,
  onInstall,
  onReset,
  children,
}: ResultCardLayoutProps) {
  return (
    <div className="flex flex-col gap-3 rounded border border-border bg-surface p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-col gap-1">
          <span className="text-sm font-medium text-text">{result.id}</span>
          {result.author && <span className="text-xs text-text-faint">{result.author}</span>}
        </div>
        {phase === "idle" && <InstallButton onInstall={onInstall} />}
      </div>
      <ResultMeta result={result} />
      {children}
      {isInstallInFlight(phase) && (
        <InstallProgress phase={phase} progressPct={progressPct} stageLabel={stageLabel} />
      )}
      {phase === "installed" && <InstalledIndicator />}
      {phase === "error" && errorMessage && <InstallError message={errorMessage} onRetry={onReset} />}
    </div>
  );
}

export function HfResultCard({ result, pollIntervalMs = DEFAULT_INSTALL_POLL_INTERVAL_MS }: HfResultCardProps) {
  const { t } = useTranslation();
  const [detailsExpanded, setDetailsExpanded] = useState(false);
  const preflightQuery = useUpscalerModelPreflight(
    result.id,
    detailsExpanded,
  );
  const { phase, progressPct, errorMessage, install, reset } = useModelInstall(pollIntervalMs);
  const warnings = preflightQuery.data
    ? buildUpscalerWarnings(preflightQuery.data)
    : [];

  return (
    <ResultCardLayout
      result={result}
      phase={phase}
      progressPct={progressPct}
      errorMessage={errorMessage}
      onInstall={() => install(result.id)}
      onReset={reset}
    >
      <UpscalerCompatBadge verdict={result.compat} />

      <button
        type="button"
        aria-expanded={detailsExpanded}
        onClick={() => setDetailsExpanded((expanded) => !expanded)}
        className="inline-flex w-fit items-center gap-1.5 rounded-sm border border-border bg-surface px-2.5 py-1.5 text-xs text-text-dim transition-[border-color,color] duration-fast hover:border-text-faint hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        {detailsExpanded ? (
          <ChevronUp
            aria-hidden="true"
            className="h-3.5 w-3.5"
            strokeWidth={1.75}
          />
        ) : (
          <ChevronDown
            aria-hidden="true"
            className="h-3.5 w-3.5"
            strokeWidth={1.75}
          />
        )}
        {t(
          detailsExpanded
            ? "upscaler.details.hide"
            : "upscaler.details.show",
        )}
      </button>

      {detailsExpanded && (
        <div className="flex flex-col gap-3 border-t border-border pt-3">
          {preflightQuery.isLoading && (
            <p role="status" className="text-sm text-text-dim">
              {t("upscaler.preflight.loading")}
            </p>
          )}

          {preflightQuery.isError && (
            <p
              role="alert"
              className="rounded border border-warn bg-surface-2 px-3 py-2 text-sm text-warn"
            >
              {t("upscaler.warning.degraded")}
            </p>
          )}

          {preflightQuery.data && (
            <>
              {preflightQuery.data.compatReasonKey &&
                preflightQuery.data.compat !== "gated" &&
                preflightQuery.data.compat !== "incompatible" && (
                  <p className="text-sm text-text-dim">
                    {t(
                      preflightQuery.data.compatReasonKey,
                      preflightQuery.data.compatReasonParams,
                    )}
                  </p>
                )}

              {(preflightQuery.data.devices.length > 0 ||
                preflightQuery.data.freeRamBytes !== null ||
                preflightQuery.data.disk !== null ||
                preflightQuery.data.downloadBytes !== null) && (
                <dl
                  className="flex flex-col gap-2"
                  aria-label={t("upscaler.capacity.ariaLabel")}
                >
                  {preflightQuery.data.devices.map((device) => (
                    <div
                      key={device.id}
                      className="flex items-center justify-between gap-3 rounded border border-border bg-surface-2 px-3 py-2 text-sm"
                    >
                      <dt className="text-text">{device.name}</dt>
                      <dd className="font-mono-tabular text-xs text-text-dim">
                        {device.freeVramBytes === null
                          ? t("upscaler.capacity.vramUnknown")
                          : t("upscaler.capacity.vramFree", {
                              free: formatGb(device.freeVramBytes),
                            })}
                      </dd>
                    </div>
                  ))}

                  {preflightQuery.data.freeRamBytes !== null && (
                    <div className="flex items-center justify-between gap-3 rounded border border-border bg-surface-2 px-3 py-2 text-sm">
                      <dt className="text-text">
                        {t("upscaler.capacity.ramLabel")}
                      </dt>
                      <dd className="font-mono-tabular text-xs text-text-dim">
                        {t("upscaler.capacity.ramFree", {
                          free: formatGb(preflightQuery.data.freeRamBytes),
                        })}
                      </dd>
                    </div>
                  )}

                  {preflightQuery.data.disk !== null && (
                    <div className="flex items-center justify-between gap-3 rounded border border-border bg-surface-2 px-3 py-2 text-sm">
                      <dt className="break-all text-text">
                        {t("upscaler.capacity.diskLabel", {
                          path: preflightQuery.data.disk.targetPath,
                        })}
                      </dt>
                      <dd className="shrink-0 font-mono-tabular text-xs text-text-dim">
                        {t("upscaler.capacity.diskFree", {
                          free: formatGb(
                            preflightQuery.data.disk.freeBytes,
                          ),
                        })}
                      </dd>
                    </div>
                  )}

                  {preflightQuery.data.downloadBytes !== null && (
                    <div className="flex items-center justify-between gap-3 rounded border border-border bg-surface-2 px-3 py-2 text-sm">
                      <dt className="text-text">
                        {t("upscaler.capacity.downloadLabel")}
                      </dt>
                      <dd className="font-mono-tabular text-xs text-text-dim">
                        {formatGb(preflightQuery.data.downloadBytes)}
                      </dd>
                    </div>
                  )}
                </dl>
              )}

              {warnings.length > 0 && (
                <ul
                  className="flex flex-col gap-2"
                  aria-label={t("upscaler.warnings.ariaLabel")}
                >
                  {warnings.map((warning, index) => (
                    <li
                      key={`${warning.code}-${warning.key}-${index}`}
                      className="flex items-start gap-2 rounded border border-warn bg-surface-2 px-3 py-2 text-sm text-warn"
                    >
                      <AlertTriangle
                        aria-hidden="true"
                        className="mt-0.5 h-4 w-4 shrink-0"
                        strokeWidth={1.75}
                      />
                      <span>{t(warning.key, warning.params)}</span>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
        </div>
      )}
    </ResultCardLayout>
  );
}
