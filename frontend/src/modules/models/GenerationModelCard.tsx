import { AlertTriangle, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";
import {
  useGenerationModelInstall,
  useGenerationModelPreflight,
} from "../../hooks/useGenerationJob";
import type {
  CheckpointCandidate,
  CompatVerdict,
  HfModelSearchResultResponse,
  Precision,
  PrecisionCost,
} from "../../lib/apiTypes";
import {
  InstalledIndicator,
  InstallButton,
  ResultMeta,
} from "./HfResultCard";
import { InstallError, InstallProgress, isInstallInFlight } from "./installUi";
import { buildWarnings } from "./generationWarnings";
import { useTranslation } from "../../i18n/LocaleProvider";
import type { TranslationParams } from "../../i18n";

interface GenerationModelCardProps {
  result: HfModelSearchResultResponse;
}

const COMPAT_KEYS: Record<CompatVerdict, string> = {
  ready_onnx: "generation.compat.readyOnnx",
  needs_conversion: "generation.compat.needsConversion",
  single_file: "generation.compat.singleFile",
  gated: "generation.compat.gated",
  incompatible: "generation.compat.incompatible",
};

function formatGb(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

function checkpointReason(
  candidate: CheckpointCandidate,
  t: (key: string, params?: TranslationParams) => string,
): string {
  // `missing` llega como lista de claves de componente separadas por coma: cada
  // rol se traduce antes de armar la oracion, si no el usuario lee
  // "component.vae" en pantalla.
  const params: TranslationParams = { ...candidate.reasonParams };
  const missing = candidate.reasonParams.missing;
  if (missing) {
    params.missing = missing
      .split(",")
      .map((role) => t(role))
      .join(", ");
  }
  return t(candidate.reasonKey, params);
}

function CompatBadge({ verdict }: { verdict: CompatVerdict | null }) {
  const { t } = useTranslation();
  const label = t(verdict ? COMPAT_KEYS[verdict] : "generation.compat.unknown");
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
      {label}
    </span>
  );
}

function PrecisionPicker({
  repoId,
  costs,
  selected,
  onChange,
}: {
  repoId: string;
  costs: PrecisionCost[];
  selected: Precision | undefined;
  onChange: (precision: Precision) => void;
}) {
  const { t } = useTranslation();

  if (costs.length === 0) {
    return null;
  }

  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
        {t("generation.precision.title")}
      </legend>
      <div className="grid gap-2 sm:grid-cols-2">
        {costs.map((cost) => (
          <label
            key={cost.precision}
            className={`flex cursor-pointer items-center justify-between gap-3 rounded border px-3 py-2 text-sm transition-[background-color,border-color] duration-fast ${
              selected === cost.precision
                ? "border-accent bg-surface-2"
                : "border-border bg-surface hover:border-text-faint"
            }`}
          >
            <span className="flex items-center gap-2">
              <input
                type="radio"
                name={`generation-precision-${repoId}`}
                value={cost.precision}
                checked={selected === cost.precision}
                onChange={() => onChange(cost.precision)}
                className="h-3.5 w-3.5 accent-accent"
              />
              <span className="font-medium text-text">{cost.precision}</span>
            </span>
            <span className="font-mono-tabular text-xs text-text-dim">
              {t("generation.precision.download", {
                size: formatGb(cost.downloadBytes),
              })}
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

function formatFileSize(bytes: number): string {
  if (bytes >= 1024 ** 3) {
    return formatGb(bytes);
  }
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function CheckpointPicker({
  repoId,
  candidates,
  selected,
  onChange,
}: {
  repoId: string;
  candidates: CheckpointCandidate[];
  selected: string | undefined;
  onChange: (path: string) => void;
}) {
  const { t } = useTranslation();

  if (candidates.length === 0) {
    return null;
  }

  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
        {t("generation.checkpoint.title")}
      </legend>
      <div className="flex flex-col gap-2">
        {candidates.map((candidate) => {
          const selectable = candidate.installable === true;
          const fileName = candidate.path.split("/").pop() ?? candidate.path;

          return (
            <label
              key={candidate.path}
              className={`flex items-start gap-3 rounded border px-3 py-2 text-sm ${
                selected === candidate.path
                  ? "border-accent bg-surface-2"
                  : "border-border bg-surface"
              } ${selectable ? "cursor-pointer" : "cursor-not-allowed"}`}
            >
              <input
                type="radio"
                name={`generation-checkpoint-${repoId}`}
                value={candidate.path}
                checked={selected === candidate.path}
                disabled={!selectable}
                onChange={() => onChange(candidate.path)}
                className="mt-0.5 h-3.5 w-3.5 accent-accent"
              />
              <span className="flex min-w-0 flex-1 flex-col gap-1">
                <span className="flex items-start justify-between gap-3">
                  <span className="break-all font-medium text-text">{fileName}</span>
                  <span className="shrink-0 font-mono-tabular text-xs text-text-dim">
                    {formatFileSize(candidate.sizeBytes)}
                  </span>
                </span>
                {!selectable && (
                  <span className="text-xs text-warn">
                    {t(
                      candidate.installable === null
                        ? "checkpoint.prefix.unknown"
                        : "checkpoint.prefix.notInstallable",
                    )}
                    {checkpointReason(candidate, t)}
                  </span>
                )}
              </span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}

export function GenerationModelCard({ result }: GenerationModelCardProps) {
  const { t } = useTranslation();
  const [detailsExpanded, setDetailsExpanded] = useState(false);
  const [selectedPrecision, setSelectedPrecision] = useState<Precision | undefined>(
    result.availablePrecisions[0],
  );
  const [selectedCheckpointPath, setSelectedCheckpointPath] = useState<string | undefined>();
  const preflightQuery = useGenerationModelPreflight(result.id, detailsExpanded);
  const { phase, progressPct, stageLabel, errorMessage, install, reset } =
    useGenerationModelInstall();

  const checkpoints = preflightQuery.data?.checkpoints ?? [];
  const selectedCheckpoint =
    checkpoints.find(
      (candidate) =>
        candidate.path === selectedCheckpointPath && candidate.installable === true,
    ) ?? checkpoints.find((candidate) => candidate.installable === true);
  const precisionCosts =
    preflightQuery.data?.precisions.filter((cost) =>
      result.availablePrecisions.includes(cost.precision),
    ) ?? [];
  const warnings = preflightQuery.data
    ? buildWarnings(preflightQuery.data, selectedPrecision ?? "fp16", selectedCheckpoint)
    : [];

  return (
    <div className="flex flex-col gap-3 rounded border border-border bg-surface p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-col gap-1.5">
          <span className="text-sm font-medium text-text">{result.id}</span>
          {result.author && <span className="text-xs text-text-faint">{result.author}</span>}
          <CompatBadge verdict={result.compat} />
        </div>
        {phase === "idle" && (
          <InstallButton
            onInstall={() =>
              install(result.id, selectedPrecision, selectedCheckpoint?.path)
            }
          />
        )}
      </div>

      <ResultMeta result={result} />

      <button
        type="button"
        aria-expanded={detailsExpanded}
        onClick={() => setDetailsExpanded((expanded) => !expanded)}
        className="inline-flex w-fit items-center gap-1.5 rounded-sm border border-border bg-surface px-2.5 py-1.5 text-xs text-text-dim transition-[border-color,color] duration-fast hover:border-text-faint hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        {detailsExpanded ? (
          <ChevronUp aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
        ) : (
          <ChevronDown aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
        )}
        {t(
          detailsExpanded
            ? "generation.details.hide"
            : "generation.details.show",
        )}
      </button>

      {detailsExpanded && (
        <div className="flex flex-col gap-3 border-t border-border pt-3">
          {preflightQuery.isLoading && (
            <p role="status" className="text-sm text-text-dim">
              {t("generation.preflight.loading")}
            </p>
          )}

          {preflightQuery.isError && (
            <p role="alert" className="rounded border border-warn bg-surface-2 px-3 py-2 text-sm text-warn">
              {t("generation.warning.degraded")}
            </p>
          )}

          {preflightQuery.data && (
            <>
              <CheckpointPicker
                repoId={result.id}
                candidates={checkpoints}
                selected={selectedCheckpoint?.path}
                onChange={setSelectedCheckpointPath}
              />

              <PrecisionPicker
                repoId={result.id}
                costs={precisionCosts}
                selected={selectedPrecision}
                onChange={setSelectedPrecision}
              />

              {(preflightQuery.data.devices.length > 0 ||
                preflightQuery.data.freeRamBytes !== null) && (
                <dl className="flex flex-col gap-2">
                  {preflightQuery.data.devices.map((device) => (
                    <div
                      key={device.id}
                      className="flex items-center justify-between gap-3 rounded border border-border bg-surface-2 px-3 py-2 text-sm"
                    >
                      <dt className="text-text">{device.name}</dt>
                      <dd className="font-mono-tabular text-xs text-text-dim">
                        {device.freeVramBytes === null
                          ? t("generation.capacity.vramUnknown")
                          : t("generation.capacity.vramFree", {
                              free: formatGb(device.freeVramBytes),
                            })}
                      </dd>
                    </div>
                  ))}
                  {preflightQuery.data.freeRamBytes !== null && (
                    <div className="flex items-center justify-between gap-3 rounded border border-border bg-surface-2 px-3 py-2 text-sm">
                      <dt className="text-text">RAM</dt>
                      <dd className="font-mono-tabular text-xs text-text-dim">
                        {t("generation.capacity.ramFree", {
                          free: formatGb(preflightQuery.data.freeRamBytes),
                        })}
                      </dd>
                    </div>
                  )}
                </dl>
              )}

              {warnings.length > 0 && (
                <ul
                  className="flex flex-col gap-2"
                  aria-label={t("generation.warnings.ariaLabel")}
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

      {isInstallInFlight(phase) && (
        <InstallProgress phase={phase} progressPct={progressPct} stageLabel={stageLabel} />
      )}
      {phase === "installed" && <InstalledIndicator />}
      {phase === "error" && errorMessage && (
        <InstallError message={errorMessage} onRetry={reset} />
      )}
    </div>
  );
}
