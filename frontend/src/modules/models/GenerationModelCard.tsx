import { AlertTriangle, ChevronDown, ChevronUp } from "lucide-react";
import { useState } from "react";
import {
  useGenerationModelInstall,
  useGenerationModelPreflight,
} from "../../hooks/useGenerationJob";
import type {
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

interface GenerationModelCardProps {
  result: HfModelSearchResultResponse;
}

const COMPAT_LABELS: Record<CompatVerdict, string> = {
  ready_onnx: "ONNX listo",
  needs_conversion: "Requiere conversión",
  gated: "Acceso restringido",
  incompatible: "Incompatible",
};

function formatGb(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
}

function CompatBadge({ verdict }: { verdict: CompatVerdict | null }) {
  const label = verdict ? COMPAT_LABELS[verdict] : "Compatibilidad desconocida";
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
  if (costs.length === 0) {
    return null;
  }

  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
        Precisión
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
              {formatGb(cost.downloadBytes)} download
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

export function GenerationModelCard({ result }: GenerationModelCardProps) {
  const [detailsExpanded, setDetailsExpanded] = useState(false);
  const [selectedPrecision, setSelectedPrecision] = useState<Precision | undefined>(
    result.availablePrecisions[0],
  );
  const preflightQuery = useGenerationModelPreflight(result.id, detailsExpanded);
  const { phase, progressPct, stageLabel, errorMessage, install, reset } =
    useGenerationModelInstall();

  const precisionCosts =
    preflightQuery.data?.precisions.filter((cost) =>
      result.availablePrecisions.includes(cost.precision),
    ) ?? [];
  const warnings = preflightQuery.data
    ? buildWarnings(preflightQuery.data, selectedPrecision ?? "fp16")
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
          <InstallButton onInstall={() => install(result.id, selectedPrecision)} />
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
        {detailsExpanded ? "Ocultar detalles" : "Ver detalles"}
      </button>

      {detailsExpanded && (
        <div className="flex flex-col gap-3 border-t border-border pt-3">
          {preflightQuery.isLoading && (
            <p role="status" className="text-sm text-text-dim">
              Evaluando descarga y capacidad…
            </p>
          )}

          {preflightQuery.isError && (
            <p role="alert" className="rounded border border-warn bg-surface-2 px-3 py-2 text-sm text-warn">
              No se pudo evaluar este modelo. Podés instalarlo igual.
            </p>
          )}

          {preflightQuery.data && (
            <>
              <PrecisionPicker
                repoId={result.id}
                costs={precisionCosts}
                selected={selectedPrecision}
                onChange={setSelectedPrecision}
              />

              {preflightQuery.data.devices.length > 0 && (
                <dl className="flex flex-col gap-2">
                  {preflightQuery.data.devices.map((device) => (
                    <div
                      key={device.id}
                      className="flex items-center justify-between gap-3 rounded border border-border bg-surface-2 px-3 py-2 text-sm"
                    >
                      <dt className="text-text">{device.name}</dt>
                      <dd className="font-mono-tabular text-xs text-text-dim">
                        {device.freeVramBytes === null
                          ? "VRAM libre desconocida"
                          : `${formatGb(device.freeVramBytes)} VRAM libre`}
                      </dd>
                    </div>
                  ))}
                </dl>
              )}

              {warnings.length > 0 && (
                <ul className="flex flex-col gap-2" aria-label="Avisos de instalación">
                  {warnings.map((warning) => (
                    <li
                      key={`${warning.code}-${warning.message}`}
                      className="flex items-start gap-2 rounded border border-warn bg-surface-2 px-3 py-2 text-sm text-warn"
                    >
                      <AlertTriangle
                        aria-hidden="true"
                        className="mt-0.5 h-4 w-4 shrink-0"
                        strokeWidth={1.75}
                      />
                      <span>{warning.message}</span>
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
