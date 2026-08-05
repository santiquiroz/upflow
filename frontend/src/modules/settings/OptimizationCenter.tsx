import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "../../i18n/LocaleProvider";
import { AlertTriangle, Ban, CheckCircle2, Loader2, Lock, type LucideIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useCapabilities } from "../../hooks/useCapabilities";
import { getOnnxDiagnostics, scanOnnxDiagnostic } from "../../lib/api";
import type { LeverResponse, LeverStatus, OnnxDiagnosticEntryResponse } from "../../lib/apiTypes";
import { fetchEditableSettings, patchSetting } from "../../services/settings";

const RESIZABLE_BAR_STORAGE_KEY = "upflow.resizableBarConfirmed";
const EDITABLE_SETTINGS_QUERY_KEY = ["editable-settings"] as const;
const ONNX_DIAGNOSTICS_QUERY_KEY = ["onnx-diagnostics"] as const;

const STATUS_ICON: Record<LeverStatus, LucideIcon> = {
  ok: CheckCircle2,
  unavailable: AlertTriangle,
  not_applicable: Ban,
  needs_admin: Lock,
};

// Claves y no texto: este mapa vive a nivel de modulo, donde no hay hook que
// llamar. La traduccion ocurre al pintar la insignia.
const STATUS_LABEL_KEY: Record<LeverStatus, string> = {
  ok: "settings.levers.ok",
  unavailable: "settings.levers.unavailable",
  not_applicable: "settings.levers.notApplicable",
  needs_admin: "settings.levers.needsAdmin",
};

const STATUS_TEXT_CLASS: Record<LeverStatus, string> = {
  ok: "text-ok",
  unavailable: "text-danger",
  not_applicable: "text-text-dim",
  needs_admin: "text-warn",
};

function LeverStatusBadge({ status }: { status: LeverStatus }) {
  const { t } = useTranslation();
  const Icon = STATUS_ICON[status];
  return (
    <span className={`flex items-center gap-1.5 text-xs ${STATUS_TEXT_CLASS[status]}`}>
      <Icon aria-hidden="true" className="h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
      {t(STATUS_LABEL_KEY[status])}
    </span>
  );
}

function FixButton({
  lever,
  onFix,
  isFixing,
}: {
  lever: LeverResponse;
  onFix: (id: string) => void;
  isFixing: boolean;
}) {
  if (!lever.fixable) {
    return null;
  }
  return (
    <button
      type="button"
      onClick={() => onFix(lever.id)}
      disabled={isFixing}
      className="inline-flex shrink-0 items-center gap-1.5 rounded-sm border border-accent px-2 py-1 text-xs font-medium text-accent transition-[background-color,color] duration-fast hover:bg-accent hover:text-bg disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
    >
      {isFixing && <Loader2 aria-hidden="true" className="h-3.5 w-3.5 animate-spin" strokeWidth={1.75} />}
      {isFixing ? "Fixing…" : "Fix"}
    </button>
  );
}

function LeverRow({
  lever,
  onFix,
  isFixing,
}: {
  lever: LeverResponse;
  onFix: (id: string) => void;
  isFixing: boolean;
}) {
  return (
    <li className="flex items-center justify-between gap-4 rounded border border-border bg-surface-2 px-3 py-2">
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className="text-sm text-text">{lever.label}</span>
        <span className="text-xs text-text-faint">{lever.detail}</span>
      </div>
      <div className="flex shrink-0 items-center gap-3">
        <LeverStatusBadge status={lever.status} />
        <FixButton lever={lever} onFix={onFix} isFixing={isFixing} />
      </div>
    </li>
  );
}

function RescanButton({ onRescan, isRescanning }: { onRescan: () => void; isRescanning: boolean }) {
  return (
    <button
      type="button"
      onClick={onRescan}
      disabled={isRescanning}
      className="shrink-0 rounded-sm border border-border bg-surface px-3 py-1.5 text-xs text-text-dim transition-[border-color,color] duration-fast hover:border-text-faint hover:text-text disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
    >
      {isRescanning ? "Scanning…" : "Re-scan"}
    </button>
  );
}

function LeversEmptyState() {
  const { t } = useTranslation();
  return <p className="text-sm text-text-faint">{t("settings.levers.none")}</p>;
}

function LeversSectionStatus({ isLoading, isError }: { isLoading: boolean; isError: boolean }) {
  const { t } = useTranslation();
  if (isLoading) {
    return <p className="text-sm text-text-dim">{t("settings.levers.loading")}</p>;
  }
  if (isError) {
    return <p className="text-sm text-danger">{t("settings.levers.loadError")}</p>;
  }
  return null;
}

function LeversList({
  levers,
  onFix,
  fixingLeverId,
}: {
  levers: LeverResponse[];
  onFix: (id: string) => void;
  fixingLeverId: string | null;
}) {
  if (levers.length === 0) {
    return <LeversEmptyState />;
  }
  return (
    <ul className="flex flex-col gap-2">
      {levers.map((lever) => (
        <LeverRow key={lever.id} lever={lever} onFix={onFix} isFixing={fixingLeverId === lever.id} />
      ))}
    </ul>
  );
}

function LeversSection() {
  const { levers, isLoading, isError, rescan, isRescanning, fix, fixingLeverId } = useCapabilities();

  return (
    <div className="flex flex-col gap-3 rounded border border-border bg-surface p-4">
      <div className="flex items-center justify-between gap-4">
        <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
          Optimization Center
        </h2>
        <RescanButton onRescan={() => rescan()} isRescanning={isRescanning} />
      </div>
      <LeversSectionStatus isLoading={isLoading} isError={isError} />
      {!isLoading && !isError && <LeversList levers={levers} onFix={fix} fixingLeverId={fixingLeverId} />}
    </div>
  );
}

// Recibe `t`: es una funcion pura, no un componente.
function diagnosticSummaryLabel(
  entry: OnnxDiagnosticEntryResponse,
  t: (key: string) => string,
): string | null {
  if (!entry.report) {
    return null;
  }
  return entry.report.clean
    ? t("settings.levers.noCpuFallback")
    : `${entry.report.hotOps.length} op(s) on CPU`;
}

function DiagnosticSummary({ entry }: { entry: OnnxDiagnosticEntryResponse }) {
  const { t } = useTranslation();
  const label = diagnosticSummaryLabel(entry, t);
  if (!label) {
    return <span className="text-xs text-text-faint">{t("settings.diagnostics.notScanned")}</span>;
  }
  const toneClassName = entry.report?.clean ? "text-ok" : "text-warn";
  return <span className={`text-xs ${toneClassName}`}>{label}</span>;
}

function DiagnosticEntryRow({ entry }: { entry: OnnxDiagnosticEntryResponse }) {
  const queryClient = useQueryClient();
  const scanMutation = useMutation({
    mutationFn: () => scanOnnxDiagnostic(entry.modelId, entry.deviceId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ONNX_DIAGNOSTICS_QUERY_KEY }),
  });

  return (
    <li className="flex items-center justify-between gap-4 rounded border border-border bg-surface-2 px-3 py-2">
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className="text-sm text-text">{entry.modelId}</span>
        <span className="text-xs text-text-faint">{entry.deviceId}</span>
      </div>
      <div className="flex shrink-0 items-center gap-3">
        <DiagnosticSummary entry={entry} />
        <button
          type="button"
          onClick={() => scanMutation.mutate()}
          disabled={scanMutation.isPending}
          className="rounded-sm border border-border bg-surface px-2 py-1 text-xs text-text-dim transition-[border-color,color] duration-fast hover:border-text-faint hover:text-text disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          {scanMutation.isPending ? "Scanning…" : "Scan"}
        </button>
      </div>
    </li>
  );
}

function DiagnosticsEmptyState() {
  const { t } = useTranslation();
  return <p className="text-sm text-text-faint">{t("settings.diagnostics.none")}</p>;
}

function DiagnosticsSectionStatus({ isLoading, isError }: { isLoading: boolean; isError: boolean }) {
  const { t } = useTranslation();
  if (isLoading) {
    return <p className="text-sm text-text-dim">{t("settings.diagnostics.loading")}</p>;
  }
  if (isError) {
    return <p className="text-sm text-danger">{t("settings.diagnostics.loadError")}</p>;
  }
  return null;
}

function DiagnosticsSection() {
  const diagnosticsQuery = useQuery({ queryKey: ONNX_DIAGNOSTICS_QUERY_KEY, queryFn: getOnnxDiagnostics });
  const entries = diagnosticsQuery.data?.entries ?? [];

  return (
    <div className="flex flex-col gap-3 rounded border border-border bg-surface p-4">
      <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">Diagnostics</h2>
      <p className="text-xs text-text-faint">
        Checks whether an ONNX model silently falls back to the CPU execution provider on your GPU. Run manually per
        model/device — this is not part of a real job.
      </p>
      <DiagnosticsSectionStatus isLoading={diagnosticsQuery.isLoading} isError={diagnosticsQuery.isError} />
      {!diagnosticsQuery.isLoading && !diagnosticsQuery.isError && entries.length === 0 && <DiagnosticsEmptyState />}
      {entries.length > 0 && (
        <ul className="flex flex-col gap-2">
          {entries.map((entry) => (
            <DiagnosticEntryRow key={`${entry.modelId}:${entry.deviceId}`} entry={entry} />
          ))}
        </ul>
      )}
    </div>
  );
}

function readLegacyResizableBarConfirmed(): boolean {
  try {
    return localStorage.getItem(RESIZABLE_BAR_STORAGE_KEY) === "true";
  } catch {
    return false;
  }
}

function clearLegacyResizableBarConfirmed(): void {
  try {
    localStorage.removeItem(RESIZABLE_BAR_STORAGE_KEY);
  } catch {
    // localStorage may be unavailable; the server remains the source of truth.
  }
}

function ResizableBarChecklist() {
  const [confirmed, setConfirmed] = useState(false);
  const initializedFromServer = useRef(false);
  const settingsQuery = useQuery({
    queryKey: EDITABLE_SETTINGS_QUERY_KEY,
    queryFn: fetchEditableSettings,
  });
  const patchMutation = useMutation({
    mutationFn: (checked: boolean) => patchSetting("rebar_confirmed", String(checked)),
  });

  useEffect(() => {
    if (!settingsQuery.data || initializedFromServer.current) {
      return;
    }
    initializedFromServer.current = true;
    const serverConfirmed =
      settingsQuery.data.settings.find((setting) => setting.key === "rebar_confirmed")?.configured ?? false;
    if (!serverConfirmed && readLegacyResizableBarConfirmed()) {
      setConfirmed(true);
      void patchSetting("rebar_confirmed", "true")
        .then(clearLegacyResizableBarConfirmed)
        .catch(() => {
          // Keep the legacy value so a later mount can retry the migration.
        });
      return;
    }
    setConfirmed(serverConfirmed);
  }, [settingsQuery.data]);

  function handleToggle(checked: boolean): void {
    setConfirmed(checked);
    patchMutation.mutate(checked);
  }

  return (
    <div className="flex flex-col gap-2 rounded border border-border bg-surface p-4">
      <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
        Resizable BAR / Above 4G Decoding
      </h2>
      <p className="text-xs text-text-faint">
        Not detectable from software — it lives in BIOS/UEFI firmware. Enable it in your motherboard&apos;s BIOS
        setup if supported, then confirm here so this panel remembers your setup.
      </p>
      <label className="flex items-center gap-2 text-sm text-text">
        <input
          type="checkbox"
          checked={confirmed}
          onChange={(event) => handleToggle(event.target.checked)}
          className="h-3.5 w-3.5 accent-accent"
        />
        I&apos;ve confirmed Resizable BAR / Above 4G Decoding is enabled in BIOS
      </label>
    </div>
  );
}

export function OptimizationCenter() {
  return (
    <div className="flex flex-col gap-4">
      <LeversSection />
      <DiagnosticsSection />
      <ResizableBarChecklist />
    </div>
  );
}
