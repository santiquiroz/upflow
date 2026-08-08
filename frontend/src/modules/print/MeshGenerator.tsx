import { useQuery } from "@tanstack/react-query";
import { Download, Sparkles, Wrench } from "lucide-react";
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

async function fetchStlAsFile(url: string, name: string): Promise<File> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const blob = await response.blob();
  return new File([blob], name, { type: "model/stl" });
}

export function MeshGenerator({
  printer,
  onRepairFile,
}: {
  printer: string;
  onRepairFile: (file: File) => void;
}) {
  const { t } = useTranslation();
  const [prompt, setPrompt] = useState("");
  const [targetMm, setTargetMm] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [isFetchingStl, setIsFetchingStl] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const jobQuery = useQuery({
    queryKey: ["shape3dJob", jobId],
    queryFn: () => getShape3dJob(jobId as string),
    enabled: jobId !== null,
    // Dos minutos de CPU: sondear cada tres segundos alcanza y no castiga la
    // maquina, que justo ahora esta ocupada generando.
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
      const medida = Number(targetMm);
      const creado = await createShape3dJob({
        prompt: prompt.trim(),
        printer,
        targetMm: medida > 0 ? medida : undefined,
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

  async function handleRepair() {
    if (!job?.downloadUrl) {
      return;
    }
    setError(null);
    setIsFetchingStl(true);
    try {
      onRepairFile(await fetchStlAsFile(job.downloadUrl, `mesh-${job.id}.stl`));
    } catch {
      setError(t("gen3d.repair.downloadFailed"));
    } finally {
      setIsFetchingStl(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <p className="text-sm text-text-dim">{t("gen3d.subtitle")}</p>
      {/* La advertencia va arriba y siempre, no escondida en un pie: una malla
          generada no tiene cotas, y eso decide para qué sirve la pieza. */}
      <p className="rounded border border-warn bg-surface-2 px-3 py-2 text-xs text-text-dim">
        {t("gen3d.noDimensions")}
      </p>

      <label className="flex flex-col gap-2">
        <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
          {t("gen3d.prompt")}
        </span>
        <textarea
          value={prompt}
          maxLength={MAX_CHARS}
          rows={3}
          onChange={(event) => setPrompt(event.target.value)}
          placeholder={t("gen3d.promptPlaceholder")}
          className="rounded border border-border bg-surface px-3 py-2 text-sm text-text focus:border-accent focus:outline-none"
        />
        <span className="text-xs text-text-faint">
          {prompt.length}/{MAX_CHARS}
        </span>
      </label>

      <label className="flex flex-col gap-1">
        <span className="text-xs text-text-dim">{t("gen3d.targetMm")}</span>
        <input
          type="number"
          min="1"
          value={targetMm}
          onChange={(event) => setTargetMm(event.target.value)}
          aria-label={t("gen3d.targetMm")}
          placeholder="mm"
          className="w-32 rounded border border-border bg-surface px-3 py-2 text-sm text-text focus:border-accent focus:outline-none"
        />
      </label>

      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => void handleGenerate()}
          disabled={prompt.trim().length === 0 || isBusy(job)}
          className="inline-flex w-fit items-center gap-2 rounded bg-accent px-4 py-2 text-sm font-medium text-bg hover:bg-accent-hover disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          <Sparkles aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
          {isBusy(job) ? t("gen3d.working") : t("gen3d.generate")}
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
          {t("gen3d.takesMinutes")}
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
            {job.canPrint ? t("gen3d.ready") : t("gen3d.readyButNotPrintable")}
          </p>
          {job.sizeMm && (
            <p className="font-mono-tabular text-xs text-text-dim">
              {job.sizeMm[0].toFixed(1)} × {job.sizeMm[1].toFixed(1)} ×{" "}
              {job.sizeMm[2].toFixed(1)} mm
              {job.triangleCount !== null && ` · ${job.triangleCount.toLocaleString()} △`}
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
          {job.downloadUrl && (
            <div className="flex flex-wrap gap-2">
              <a
                href={job.downloadUrl}
                download
                className="inline-flex w-fit items-center gap-2 rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg hover:bg-accent-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
              >
                <Download aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
                {t("gen3d.download")}
              </a>
              {/* Sin este botón el camino real es bajar el STL y volver a
                  subirlo a mano al carril de chequeo. Acá se hace ese viaje
                  por el usuario. */}
              {!job.canPrint && (
                <button
                  type="button"
                  onClick={() => void handleRepair()}
                  disabled={isFetchingStl}
                  className="inline-flex items-center gap-2 rounded border border-border bg-surface px-3 py-1.5 text-sm text-text hover:border-text-faint disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
                >
                  <Wrench aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
                  {t("gen3d.repair")}
                </button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
