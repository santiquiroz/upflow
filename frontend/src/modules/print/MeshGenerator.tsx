import { useQuery } from "@tanstack/react-query";
import { Camera, Download, Sparkles, Wrench } from "lucide-react";
import { useState, type ChangeEvent } from "react";
import { PackDownload } from "../../components/PackDownload";
import { useCapabilityTree } from "../../hooks/useCapabilityTree";
import { useTranslation } from "../../i18n/LocaleProvider";
import { uploadGenerationInitImage } from "../../services/generation";
import {
  cancelShape3dJob,
  createShape3dJob,
  getShape3dJob,
  type Shape3dJob,
} from "../../services/print";
import { SizeSuggestion, objectHintFromFileName } from "./SizeSuggestion";
import { trackShape3dJob } from "./trackShape3dJob";

const POLL_MS = 3000;
const MAX_CHARS = 400;

// De dónde salió la medida, dicho en la pantalla del resultado. Un mapa y no una
// clave armada al vuelo: un valor inesperado del servidor mostraría la clave
// cruda en pantalla.
const SIZE_SOURCE_KEYS: Record<string, string> = {
  user: "gen3d.size.source.user",
  estimate: "gen3d.size.source.estimate",
  default: "gen3d.size.source.default",
};

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

type GeneratorMode = "text" | "photo";

export function MeshGenerator({
  printer,
  onRepairFile,
}: {
  printer: string;
  onRepairFile: (file: File) => void;
}) {
  const { t } = useTranslation();
  const [mode, setMode] = useState<GeneratorMode>("text");
  const [prompt, setPrompt] = useState("");
  const [photo, setPhoto] = useState<File | null>(null);
  const [photoUrl, setPhotoUrl] = useState<string | null>(null);
  const [targetMm, setTargetMm] = useState("");
  // De dónde salió lo que hay en el campo. Arranca en "user" porque lo que se
  // tipea es del usuario; solo aceptar una sugerencia lo cambia.
  const [targetSource, setTargetSource] = useState<"user" | "estimate">("user");
  const [targetReference, setTargetReference] = useState("");
  const [jobId, setJobId] = useState<string | null>(null);
  const [isFetchingStl, setIsFetchingStl] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // El modo foto necesita su propio pack (openai/shap-e-img2img, otro repo que
  // el de texto): el estado sale del árbol de capacidades, que mira el disco.
  const capabilityTree = useCapabilityTree();
  const capabilities =
    capabilityTree.data?.domains.flatMap((domain) => [
      ...domain.capabilities,
      ...domain.roadmap,
    ]) ?? [];
  const statusOf = (id: string) =>
    capabilities.find((capability) => capability.id === id)?.status ?? null;
  const photoNeedsPack = statusOf("print.generatePhoto") === "needs_setup";
  // Sin servidor de modelo configurado no hay sugerencia y tampoco hay ruido:
  // la pantalla queda exactamente como estaba.
  const canEstimate = statusOf("print.estimateSize") === "available";

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

  const canGenerate =
    mode === "photo" ? photo !== null && !photoNeedsPack : prompt.trim().length > 0;
  // Qué objeto es, para preguntarle al modelo cuánto mide. Desde texto es la
  // descripción; desde una foto, lo único que hay es el nombre del archivo, y
  // solo si dice algo.
  const sizeHint =
    mode === "photo" ? objectHintFromFileName(photo?.name ?? "") : prompt.trim();

  function handlePickPhoto(event: ChangeEvent<HTMLInputElement>) {
    const picked = event.target.files?.[0];
    if (!picked) {
      return;
    }
    setPhoto(picked);
    // Se revoca la anterior: si no, cada cambio de foto deja un blob en memoria.
    setPhotoUrl((previous) => {
      if (previous) {
        URL.revokeObjectURL(previous);
      }
      return URL.createObjectURL(picked);
    });
  }

  // La medida y su procedencia viajan juntas, o no viaja ninguna: mandar el
  // número sin decir de dónde salió es lo que haría que el resultado mienta.
  function sizeFields() {
    const medida = Number(targetMm);
    if (!(medida > 0)) {
      return {};
    }
    return {
      targetMm: medida,
      targetMmSource: targetSource,
      targetMmReference: targetSource === "estimate" ? targetReference : undefined,
    };
  }

  function acceptSuggestion(longestMm: number, reference: string) {
    setTargetMm(String(longestMm));
    setTargetSource("estimate");
    setTargetReference(reference);
  }

  function handleTypeSize(value: string) {
    setTargetMm(value);
    // Tocar el campo lo vuelve una medida del usuario aunque haya salido de una
    // sugerencia: atribuirsela al modelo seria falsear la procedencia.
    setTargetSource("user");
    setTargetReference("");
  }

  async function createPhotoJob(): Promise<Shape3dJob> {
    // La foto viaja por el mismo staging que la generación de imágenes: se sube
    // aparte y el job JSON lleva solo el token.
    const subida = await uploadGenerationInitImage(photo as File);
    return createShape3dJob({
      printer,
      source: "photo",
      imageToken: subida.initImageToken,
      ...sizeFields(),
    });
  }

  async function handleGenerate() {
    if (!canGenerate) {
      return;
    }
    setError(null);
    try {
      const creado =
        mode === "photo"
          ? await createPhotoJob()
          : await createShape3dJob({
              prompt: prompt.trim(),
              printer,
              ...sizeFields(),
            });
      setJobId(creado.id);
      trackShape3dJob(creado, photo?.name ?? prompt.trim());
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

      <div role="tablist" className="flex gap-2 border-b border-border">
        {(["text", "photo"] as const).map((id) => (
          <button
            key={id}
            role="tab"
            type="button"
            aria-selected={mode === id}
            onClick={() => setMode(id)}
            className={`px-3 py-2 text-sm ${
              mode === id
                ? "border-b-2 border-accent font-medium text-text"
                : "text-text-dim hover:text-text"
            }`}
          >
            {t(`gen3d.mode.${id}`)}
          </button>
        ))}
      </div>

      {mode === "text" && (
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
      )}

      {mode === "photo" && (
        <div className="flex flex-col gap-3">
          {/* El copy honesto no es opcional: esto interpreta el objeto, no lo
              replica, y un dibujo plano sale como una extrusión plana. */}
          <p className="rounded border border-border bg-surface-2 px-3 py-2 text-xs text-text-dim">
            {t("gen3d.photo.disclaimer")}
          </p>

          {photoNeedsPack ? (
            <PackDownload
              pack="shap-e-img2img"
              reason={t("gen3d.photo.missingPack")}
              onDone={() => void capabilityTree.refetch()}
            />
          ) : (
            <label htmlFor="gen3d-photo-input" className="flex cursor-pointer flex-col gap-2">
              <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
                {t("gen3d.photo.choose")}
              </span>
              <span className="flex flex-col items-center gap-2 rounded border border-dashed border-border bg-surface px-4 py-6 text-center transition-[border-color] duration-fast hover:border-accent">
                {photoUrl ? (
                  <img
                    src={photoUrl}
                    alt={t("gen3d.photo.previewAlt")}
                    className="max-h-40 max-w-full rounded object-contain"
                  />
                ) : (
                  <Camera
                    aria-hidden="true"
                    className="h-6 w-6 text-text-faint"
                    strokeWidth={1.5}
                  />
                )}
                <span className="text-sm text-text">
                  {photo ? photo.name : t("gen3d.photo.drop")}
                </span>
                {photo && (
                  <span className="text-xs text-text-faint">{t("gen3d.photo.replace")}</span>
                )}
              </span>
              <input
                id="gen3d-photo-input"
                type="file"
                accept="image/*"
                aria-label={t("gen3d.photo.choose")}
                className="sr-only"
                onChange={handlePickPhoto}
              />
            </label>
          )}
        </div>
      )}

      <label className="flex flex-col gap-1">
        <span className="text-xs text-text-dim">{t("gen3d.targetMm")}</span>
        <input
          type="number"
          min="1"
          value={targetMm}
          onChange={(event) => handleTypeSize(event.target.value)}
          aria-label={t("gen3d.targetMm")}
          placeholder="mm"
          className="w-32 rounded border border-border bg-surface px-3 py-2 text-sm text-text focus:border-accent focus:outline-none"
        />
      </label>

      {/* La sugerencia va al lado del campo y NUNCA lo pisa: llena el campo solo
          cuando el usuario hace clic en aceptarla. */}
      {canEstimate && <SizeSuggestion hint={sizeHint} onAccept={acceptSuggestion} />}

      <div className="flex gap-2">
        <button
          type="button"
          onClick={() => void handleGenerate()}
          disabled={!canGenerate || isBusy(job)}
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
          {/* Un tamaño sin decir de dónde salió miente por omisión: los 100 mm
              por defecto no los eligió nadie. */}
          {job.targetMmSource && SIZE_SOURCE_KEYS[job.targetMmSource] && (
            <p className="text-xs text-text-faint">
              {t(SIZE_SOURCE_KEYS[job.targetMmSource])}
              {job.targetMmReference
                ? ` ${t("gen3d.estimate.reference", { reference: job.targetMmReference })}`
                : ""}
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
