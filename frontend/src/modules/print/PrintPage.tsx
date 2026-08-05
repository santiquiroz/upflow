import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Box, CheckCircle2, Download, Lightbulb, UploadCloud, Wrench } from "lucide-react";
import { PartBuilder } from "./PartBuilder";
import { useState, type ChangeEvent, type DragEvent } from "react";
import { useTranslation } from "../../i18n/LocaleProvider";
import {
  checkPrint,
  fetchPrinters,
  repairMesh,
  type MeshRepairResult,
  type PrintCheckResult,
} from "../../services/print";

const AXES = ["x", "y", "z"] as const;

function Dropzone({
  file,
  onFileSelected,
}: {
  file: File | null;
  onFileSelected: (file: File) => void;
}) {
  const { t } = useTranslation();

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    const dropped = event.dataTransfer.files[0];
    if (dropped) {
      onFileSelected(dropped);
    }
  }

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0];
    if (selected) {
      onFileSelected(selected);
    }
  }

  return (
    <label
      htmlFor="print-file-input"
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
      className="flex cursor-pointer flex-col items-center gap-2 rounded border border-dashed border-border bg-surface px-6 py-10 text-center transition-[border-color] duration-fast hover:border-accent"
    >
      <UploadCloud aria-hidden="true" className="h-6 w-6 text-text-faint" strokeWidth={1.5} />
      <span className="text-sm text-text">{file ? file.name : t("print.dropzone")}</span>
      <span className="text-xs text-text-faint">STL</span>
      <input
        id="print-file-input"
        type="file"
        accept=".stl"
        aria-label={t("print.dropzone")}
        className="sr-only"
        onChange={handleChange}
      />
    </label>
  );
}

function Verdict({ result }: { result: PrintCheckResult }) {
  const { t } = useTranslation();
  const Icon = result.canPrint ? CheckCircle2 : AlertTriangle;
  return (
    <div
      role="status"
      className={`flex items-center gap-2 rounded border p-3 ${
        result.canPrint ? "border-ok bg-surface-2" : "border-danger bg-surface-2"
      }`}
    >
      <Icon
        aria-hidden="true"
        className={`h-5 w-5 ${result.canPrint ? "text-ok" : "text-danger"}`}
        strokeWidth={1.75}
      />
      <span className="font-heading text-sm font-semibold text-text">
        {result.canPrint ? t("print.verdict.yes") : t("print.verdict.no")}
      </span>
    </div>
  );
}

function Measurements({ result }: { result: PrintCheckResult }) {
  const { t } = useTranslation();
  const [x, y, z] = result.sizeMm;
  const filas: [string, string][] = [
    [t("print.size"), `${x.toFixed(1)} × ${y.toFixed(1)} × ${z.toFixed(1)} mm`],
    [t("print.triangles"), result.triangleCount.toLocaleString()],
    [t("print.watertight"), result.watertight ? t("common.yes") : t("common.no")],
    [t("print.manifold"), result.manifold ? t("common.yes") : t("common.no")],
    [t("print.overhang"), `${Math.round(result.overhangRatio * 100)}%`],
  ];
  if (result.volumeMm3 !== null) {
    // En centímetros cúbicos porque es lo que se compara con el filamento.
    filas.push([t("print.volume"), `${(result.volumeMm3 / 1000).toFixed(1)} cm³`]);
  }

  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
      {filas.map(([etiqueta, valor]) => (
        <div key={etiqueta} className="contents">
          <dt className="text-text-dim">{etiqueta}</dt>
          <dd className="font-mono-tabular text-text">{valor}</dd>
        </div>
      ))}
    </dl>
  );
}

export function PrintPage() {
  const { t } = useTranslation();
  const [lane, setLane] = useState<"check" | "build">("check");
  const [file, setFile] = useState<File | null>(null);
  const [printer, setPrinter] = useState("ender-3");
  const [targetAxis, setTargetAxis] = useState("");
  const [targetMm, setTargetMm] = useState("");
  const [result, setResult] = useState<PrintCheckResult | null>(null);
  const [repair, setRepair] = useState<MeshRepairResult | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const printersQuery = useQuery({ queryKey: ["printers"], queryFn: fetchPrinters });
  const printers = printersQuery.data?.printers ?? [];

  async function handleCheck() {
    if (!file) {
      return;
    }
    setIsBusy(true);
    setError(null);
    setResult(null);
    try {
      const medida = Number(targetMm);
      setRepair(null);
      setResult(
        await checkPrint({
          file,
          printer,
          targetAxis: targetAxis || undefined,
          targetMm: targetAxis && medida > 0 ? medida : undefined,
        }),
      );
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setIsBusy(false);
    }
  }

  async function handleRepair() {
    if (!file) {
      return;
    }
    setIsBusy(true);
    setError(null);
    try {
      setRepair(await repairMesh(file));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setIsBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="flex items-center gap-2 font-heading text-xl font-semibold text-text">
          <Box aria-hidden="true" className="h-5 w-5 text-accent" strokeWidth={1.75} />
          {t("print.title")}
        </h1>
        <p className="text-sm text-text-dim">{t("print.subtitle")}</p>
      </header>

      {/* Los dos carriles con nombre: es el hallazgo de la investigación, no una
          decisión de diseño. Una malla generada no tiene cotas; una pieza que
          tiene que encajar las necesita. Esconder la diferencia sería mentir. */}
      <div role="tablist" className="flex gap-2 border-b border-border">
        {(["check", "build"] as const).map((id) => (
          <button
            key={id}
            role="tab"
            type="button"
            aria-selected={lane === id}
            onClick={() => setLane(id)}
            className={`px-3 py-2 text-sm ${
              lane === id
                ? "border-b-2 border-accent font-medium text-text"
                : "text-text-dim hover:text-text"
            }`}
          >
            {t(id === "check" ? "print.lane.check" : "print.lane.build")}
          </button>
        ))}
      </div>

      {lane === "build" && <PartBuilder printer={printer} />}

      <div
        className="grid grid-cols-[1fr_360px] gap-6 max-[900px]:grid-cols-1"
        hidden={lane !== "check"}
      >
        <div className="flex flex-col gap-4">
          <Dropzone file={file} onFileSelected={(f) => { setFile(f); setResult(null); }} />

          <label className="flex flex-col gap-2">
            <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
              {t("print.printer")}
            </span>
            <select
              value={printer}
              onChange={(event) => setPrinter(event.target.value)}
              className="rounded border border-border bg-surface px-3 py-2 text-sm text-text focus:border-accent focus:outline-none"
            >
              {printers.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.id} — {p.bedMm[0]}×{p.bedMm[1]}×{p.bedMm[2]} mm
                </option>
              ))}
            </select>
          </label>

          <fieldset className="flex flex-col gap-2">
            <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
              {t("print.targetSize")}
            </legend>
            <p className="text-xs text-text-faint">{t("print.targetSizeHint")}</p>
            <div className="flex gap-2">
              <select
                value={targetAxis}
                onChange={(event) => setTargetAxis(event.target.value)}
                aria-label={t("print.targetAxis")}
                className="rounded border border-border bg-surface px-3 py-2 text-sm text-text focus:border-accent focus:outline-none"
              >
                <option value="">{t("print.targetNone")}</option>
                {AXES.map((eje) => (
                  <option key={eje} value={eje}>
                    {eje.toUpperCase()}
                  </option>
                ))}
              </select>
              <input
                type="number"
                min="1"
                value={targetMm}
                onChange={(event) => setTargetMm(event.target.value)}
                aria-label={t("print.targetMm")}
                placeholder="mm"
                className="w-28 rounded border border-border bg-surface px-3 py-2 text-sm text-text focus:border-accent focus:outline-none"
              />
            </div>
          </fieldset>

          <button
            type="button"
            onClick={() => void handleCheck()}
            disabled={file === null || isBusy}
            className="inline-flex w-fit items-center gap-2 rounded bg-accent px-4 py-2 text-sm font-medium text-bg transition-[background-color,opacity] duration-fast hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
          >
            {isBusy ? t("print.checking") : t("print.check")}
          </button>

          {error && (
            <p role="alert" className="text-sm text-danger">
              {error}
            </p>
          )}
        </div>

        <aside className="flex min-h-64 flex-col gap-4 rounded border border-border bg-surface p-4">
          <h2 className="font-heading text-lg font-semibold text-text">{t("print.result")}</h2>

          {result === null && !isBusy && (
            <p className="text-sm text-text-faint">{t("print.waiting")}</p>
          )}

          {result !== null && (
            <>
              <Verdict result={result} />
              <Measurements result={result} />

              {!result.watertight && repair === null && (
                <button
                  type="button"
                  onClick={() => void handleRepair()}
                  disabled={isBusy}
                  className="inline-flex w-fit items-center gap-2 rounded border border-border bg-surface px-3 py-1.5 text-sm text-text hover:border-text-faint disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
                >
                  <Wrench aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
                  {t("print.repair")}
                </button>
              )}

              {repair !== null && (
                <div className="flex flex-col gap-2 rounded border border-border bg-surface-2 p-2">
                  <p role="status" className="text-sm text-text">
                    {repair.watertight ? t("print.repair.closed") : t("print.repair.stillOpen")}
                  </p>
                  {repair.blockers.map((b) => (
                    <p key={b} className="text-xs text-text-dim">
                      {b}
                    </p>
                  ))}
                  <a
                    href={repair.downloadUrl}
                    download
                    className="inline-flex w-fit items-center gap-2 rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg hover:bg-accent-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
                  >
                    <Download aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
                    {t("print.repair.download")}
                  </a>
                </div>
              )}

              {result.blockers.length > 0 && (
                <div className="flex flex-col gap-1">
                  <h3 className="font-heading text-xs font-semibold uppercase tracking-wide text-danger">
                    {t("print.blockers")}
                  </h3>
                  <ul className="flex flex-col gap-1">
                    {result.blockers.map((b) => (
                      <li key={b} className="text-xs text-text-dim">
                        {b}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {result.advice.length > 0 && (
                <div className="flex flex-col gap-1">
                  <h3 className="flex items-center gap-1 font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
                    <Lightbulb aria-hidden="true" className="h-3 w-3" strokeWidth={1.75} />
                    {t("print.advice")}
                  </h3>
                  <ul className="flex flex-col gap-1">
                    {result.advice.map((a) => (
                      <li key={a} className="text-xs text-text-dim">
                        {a}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </>
          )}
        </aside>
      </div>
    </div>
  );
}
