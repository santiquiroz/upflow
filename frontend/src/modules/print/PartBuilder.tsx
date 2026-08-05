import { useQuery } from "@tanstack/react-query";
import { Download, Ruler } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "../../i18n/LocaleProvider";
import {
  fetchPartKinds,
  generatePart,
  type GeneratedPart,
  type PartKind,
} from "../../services/print";

function defaultsFor(kind: PartKind): Record<string, number> {
  return Object.fromEntries(kind.params.map((p) => [p.name, p.default]));
}

export function PartBuilder({ printer }: { printer: string }) {
  const { t } = useTranslation();
  const [kindId, setKindId] = useState<string | null>(null);
  const [values, setValues] = useState<Record<string, number>>({});
  const [part, setPart] = useState<GeneratedPart | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const kindsQuery = useQuery({ queryKey: ["partKinds"], queryFn: fetchPartKinds });
  const kinds = kindsQuery.data?.kinds ?? [];
  const kind = kinds.find((k) => k.id === kindId) ?? kinds[0] ?? null;

  function selectKind(id: string) {
    const elegida = kinds.find((k) => k.id === id);
    setKindId(id);
    setValues(elegida ? defaultsFor(elegida) : {});
    setPart(null);
    setError(null);
  }

  function currentValues(): Record<string, number> {
    // Los defaults valen hasta que el usuario toque el campo: sin esto, abrir la
    // pantalla y darle a construir mandaría un objeto vacío.
    return kind && Object.keys(values).length === 0 ? defaultsFor(kind) : values;
  }

  async function handleBuild() {
    if (!kind) {
      return;
    }
    setIsBusy(true);
    setError(null);
    setPart(null);
    try {
      setPart(await generatePart({ kind: kind.id, params: currentValues(), printer }));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setIsBusy(false);
    }
  }

  if (kinds.length === 0) {
    return null;
  }

  const medidas = currentValues();

  return (
    <div className="flex flex-col gap-4">
      <p className="text-sm text-text-dim">{t("part.subtitle")}</p>

      <label className="flex flex-col gap-2">
        <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
          {t("part.kind")}
        </span>
        <select
          value={kind?.id ?? ""}
          onChange={(event) => selectKind(event.target.value)}
          className="rounded border border-border bg-surface px-3 py-2 text-sm text-text focus:border-accent focus:outline-none"
        >
          {kinds.map((k) => (
            <option key={k.id} value={k.id}>
              {t(k.labelKey)}
            </option>
          ))}
        </select>
      </label>

      {kind && <p className="text-xs text-text-faint">{t(kind.descriptionKey)}</p>}

      <div className="grid grid-cols-3 gap-2 max-[600px]:grid-cols-1">
        {kind?.params.map((param) => (
          <label key={param.name} className="flex flex-col gap-1">
            <span className="text-xs text-text-dim">{t(param.labelKey)}</span>
            <input
              type="number"
              step="0.1"
              min={param.minimum}
              value={medidas[param.name] ?? param.default}
              aria-label={t(param.labelKey)}
              onChange={(event) =>
                setValues({ ...medidas, [param.name]: Number(event.target.value) })
              }
              className="rounded border border-border bg-surface px-3 py-2 font-mono-tabular text-sm text-text focus:border-accent focus:outline-none"
            />
          </label>
        ))}
      </div>

      <button
        type="button"
        onClick={() => void handleBuild()}
        disabled={isBusy || kind === null}
        className="inline-flex w-fit items-center gap-2 rounded bg-accent px-4 py-2 text-sm font-medium text-bg hover:bg-accent-hover disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        <Ruler aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
        {isBusy ? t("part.building") : t("part.build")}
      </button>

      {error && (
        <p role="alert" className="text-sm text-danger">
          {error}
        </p>
      )}

      {part && (
        <div className="flex flex-col gap-2 rounded border border-border bg-surface-2 p-3">
          <p role="status" className="text-sm text-text">
            {part.canPrint ? t("part.ready") : t("part.readyButDoesNotFit")}
          </p>
          <p className="font-mono-tabular text-xs text-text-dim">
            {part.sizeMm[0].toFixed(1)} × {part.sizeMm[1].toFixed(1)} ×{" "}
            {part.sizeMm[2].toFixed(1)} mm
            {part.volumeMm3 !== null && ` · ${(part.volumeMm3 / 1000).toFixed(1)} cm³`}
          </p>
          {part.blockers.map((b) => (
            <p key={b} className="text-xs text-danger">
              {b}
            </p>
          ))}
          {part.advice.map((a) => (
            <p key={a} className="text-xs text-text-dim">
              {a}
            </p>
          ))}
          <a
            href={part.downloadUrl}
            download
            className="inline-flex w-fit items-center gap-2 rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg hover:bg-accent-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
          >
            <Download aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
            {t("part.download")}
          </a>
        </div>
      )}
    </div>
  );
}
