import { useQuery } from "@tanstack/react-query";
import { Download, Ruler } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "../../i18n/LocaleProvider";
import {
  cancelShape3dJob,
  createShape3dJob,
  getShape3dJob,
  type Shape3dJob,
} from "../../services/print";

const POLL_MS = 3000;
const MAX_CHARS = 400;

function isBusy(job: Shape3dJob | null): boolean {
  return job !== null && (job.status === "queued" || job.status === "running");
}

/** Las tres o ninguna: dos de tres no es una cota contra la que comparar. */
function parseSize(
  largo: string,
  ancho: string,
  alto: string,
): [number, number, number] | undefined {
  const medidas = [largo, ancho, alto].map(Number);
  if (medidas.some((m) => !Number.isFinite(m) || m <= 0)) {
    return undefined;
  }
  return [medidas[0], medidas[1], medidas[2]];
}

export function CadGenerator({ printer }: { printer: string }) {
  const { t } = useTranslation();
  const [prompt, setPrompt] = useState("");
  const [largo, setLargo] = useState("");
  const [ancho, setAncho] = useState("");
  const [alto, setAlto] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const jobQuery = useQuery({
    queryKey: ["shape3dJob", jobId],
    queryFn: () => getShape3dJob(jobId as string),
    enabled: jobId !== null,
    refetchInterval: (query) =>
      isBusy(query.state.data ?? null) ? POLL_MS : false,
  });
  const job = jobQuery.data ?? null;

  async function handleGenerate() {
    if (!prompt.trim()) {
      return;
    }
    setError(null);
    try {
      const creado = await createShape3dJob({
        prompt: prompt.trim(),
        printer,
        source: "cad",
        expectedSize: parseSize(largo, ancho, alto),
      });
      setJobId(creado.id);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    }
  }

  async function handleCancel() {
    if (jobId) {
      await cancelShape3dJob(jobId).catch(() => undefined);
      void jobQuery.refetch();
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <p className="text-sm text-text-dim">{t("cad.subtitle")}</p>
      {/* Lo contrario del aviso del otro carril: acá las cotas SON el producto. */}
      <p className="rounded border border-accent bg-surface-2 px-3 py-2 text-xs text-text-dim">
        {t("cad.hasDimensions")}
      </p>

      <label className="flex flex-col gap-2">
        <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
          {t("cad.prompt")}
        </span>
        <textarea
          value={prompt}
          maxLength={MAX_CHARS}
          rows={3}
          onChange={(event) => setPrompt(event.target.value)}
          placeholder={t("cad.promptPlaceholder")}
          className="rounded border border-border bg-surface px-3 py-2 text-sm text-text focus:border-accent focus:outline-none"
        />
        <span className="text-xs text-text-faint">
          {prompt.length}/{MAX_CHARS}
        </span>
      </label>

      <fieldset className="flex flex-col gap-2">
        <legend className="text-xs text-text-dim">{t("cad.expectedSize")}</legend>
        <div className="flex gap-2">
          {(
            [
              [t("cad.length"), largo, setLargo],
              [t("cad.width"), ancho, setAncho],
              [t("cad.height"), alto, setAlto],
            ] as const
          ).map(([etiqueta, valor, set]) => (
            <label key={etiqueta} className="flex flex-col gap-1">
              <span className="text-xs text-text-faint">{etiqueta}</span>
              <input
                type="number"
                min="1"
                value={valor}
                onChange={(event) => set(event.target.value)}
                aria-label={etiqueta}
                placeholder="mm"
                className="w-24 rounded border border-border bg-surface px-3 py-2 text-sm text-text focus:border-accent focus:outline-none"
              />
            </label>
          ))}
        </div>
        <span className="text-xs text-text-faint">{t("cad.sizeHint")}</span>
      </fieldset>

      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => void handleGenerate()}
          disabled={prompt.trim().length === 0 || isBusy(job)}
          className="inline-flex w-fit items-center gap-2 rounded bg-accent px-4 py-2 text-sm font-medium text-bg hover:bg-accent-hover disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          <Ruler aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
          {isBusy(job) ? t("cad.working") : t("cad.generate")}
        </button>
        {isBusy(job) && (
          <button
            type="button"
            onClick={() => void handleCancel()}
            className="inline-flex items-center gap-2 rounded border border-border bg-surface px-3 py-1.5 text-sm text-text hover:border-text-faint focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
          >
            {t("common.cancel")}
          </button>
        )}
      </div>

      {isBusy(job) && (
        <p role="status" className="text-sm text-text-dim">
          {t("cad.takesAWhile")}
        </p>
      )}

      {error && (
        <p role="alert" className="text-sm text-danger">
          {error}
        </p>
      )}

      {job?.status === "failed" && (
        <p role="alert" className="text-sm text-danger">
          {job.error}
        </p>
      )}

      {job?.status === "completed" && (
        <div className="flex flex-col gap-2 rounded border border-border bg-surface-2 p-3">
          <p role="status" className="text-sm text-text">
            {job.canPrint ? t("cad.ready") : t("cad.readyButNotPrintable")}
          </p>
          {job.sizeMm && (
            <p className="font-mono-tabular text-sm text-text">
              {job.sizeMm[0].toFixed(1)} × {job.sizeMm[1].toFixed(1)} ×{" "}
              {job.sizeMm[2].toFixed(1)} mm
            </p>
          )}
          {job.retries > 0 && (
            <p className="text-xs text-text-dim">
              {t("cad.correctedItself")} {job.retries}
            </p>
          )}
          {job.blockers.map((b) => (
            <p key={b} className="text-xs text-danger">
              {b}
            </p>
          ))}
          {job.advice.map((a) => (
            <p key={a} className="text-xs text-text-dim">
              {a}
            </p>
          ))}
          {job.code && (
            <details className="text-xs">
              <summary className="cursor-pointer text-text-dim hover:text-text">
                {t("cad.showCode")}
              </summary>
              {/* El .scad es lo unico ajustable: el STL ya salio cocido. */}
              <pre className="mt-2 max-h-64 overflow-auto rounded bg-surface p-2 font-mono-tabular text-xs text-text">
                {job.code}
              </pre>
            </details>
          )}
          {job.downloadUrl && (
            <a
              href={job.downloadUrl}
              download
              className="inline-flex w-fit items-center gap-2 rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg hover:bg-accent-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
            >
              <Download aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
              {t("cad.download")}
            </a>
          )}
        </div>
      )}
    </div>
  );
}
