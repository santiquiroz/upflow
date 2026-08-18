import {
  AudioLines,
  Check,
  Clipboard,
  Download,
  UploadCloud,
  X,
} from "lucide-react";
import {
  useEffect,
  useState,
  type ChangeEvent,
  type DragEvent,
} from "react";
import { DeterminateProgressBar } from "../../components/DeterminateProgressBar";
import { IndeterminateProgressBar } from "../../components/IndeterminateProgressBar";
import {
  DEFAULT_ASR_INSTALL_POLL_INTERVAL_MS,
  DEFAULT_TRANSCRIBE_POLL_INTERVAL_MS,
  useInstalledAsrModels,
  useTranscribeDevices,
  useTranscribeJob,
  type TranscribeJobPhase,
} from "../../hooks/useTranscribeJob";
import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "../../i18n/LocaleProvider";
import { PackDownload } from "../../components/PackDownload";
import { isEnglishOnly } from "../../lib/englishOnly";
import type { TranscribeJob } from "../../lib/apiTypes";
import { fetchTranslationPairs } from "../../services/transcribe";
import type {
  CreateTranscribeJobParams,
  TranscribeOutputMode,
} from "../../services/transcribe";
import { AsrModelSearch } from "./AsrModelSearch";

interface TranscribePanelProps {
  pollIntervalMs?: number;
  installPollIntervalMs?: number;
  searchDebounceMs?: number;
}

const LANGUAGE_OPTIONS = ["es", "en", "pt", "fr", "de", "it"] as const;

function isJobBusy(phase: TranscribeJobPhase): boolean {
  return phase === "uploading" || phase === "queued" || phase === "running";
}

function AudioDropzone({
  files,
  onFilesSelected,
}: {
  files: File[];
  onFilesSelected: (files: File[]) => void;
}) {
  const { t } = useTranslation();

  function handleDrop(event: DragEvent<HTMLLabelElement>): void {
    event.preventDefault();
    const dropped = Array.from(event.dataTransfer.files);
    if (dropped.length > 0) {
      onFilesSelected(dropped);
    }
  }

  function handleChange(event: ChangeEvent<HTMLInputElement>): void {
    const selected = Array.from(event.target.files ?? []);
    if (selected.length > 0) {
      onFilesSelected(selected);
    }
  }

  return (
    <label
      htmlFor="transcribe-file-input"
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
      className="flex cursor-pointer flex-col items-center gap-2 rounded border border-dashed border-border bg-surface px-6 py-10 text-center transition-[border-color] duration-fast hover:border-accent"
    >
      <UploadCloud
        aria-hidden="true"
        className="h-6 w-6 text-text-faint"
        strokeWidth={1.5}
      />
      <span className="text-sm text-text">
        {files.length === 0
          ? t("transcribe.file.drop")
          : files.length === 1
            ? files[0].name
            : t("enhance.batch.selected", { count: files.length })}
      </span>
      <span className="text-xs text-text-faint">
        {t("transcribe.file.formats")}
      </span>
      <input
        id="transcribe-file-input"
        type="file"
        accept="audio/*"
        multiple
        aria-label={t("transcribe.file.inputLabel")}
        className="sr-only"
        onChange={handleChange}
      />
    </label>
  );
}

// Un contenedor de video es lo unico a lo que se le pueden pegar subtitulos.
// La lista es de contenedores, no de codecs: es lo que se ve en el nombre.
const VIDEO_EXTENSIONS = [".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".ts"];

function hasPicture(file: File | null): boolean {
  if (file === null) {
    return false;
  }
  const nombre = file.name.toLowerCase();
  return VIDEO_EXTENSIONS.some((extension) => nombre.endsWith(extension));
}

function phaseLabelKey(phase: TranscribeJobPhase): string {
  switch (phase) {
    case "uploading":
      return "transcribe.job.uploading";
    case "queued":
      return "transcribe.job.queued";
    case "running":
      return "transcribe.job.running";
    default:
      return "transcribe.job.waiting";
  }
}

function subtitleHref(jobId: string, translateTo: string): string {
  const base = `/api/v1/transcribe/jobs/${jobId}/download?fmt=srt`;
  return translateTo ? `${base}&translate_to=${translateTo}` : base;
}

function TranscriptionResult({
  phase,
  job,
  errorMessage,
  onCancel,
  uploadPercent,
  translationTargets,
  installablePairs,
  onPairInstalled,
}: {
  phase: TranscribeJobPhase;
  job: TranscribeJob | undefined;
  errorMessage: string | null;
  onCancel: () => void;
  // `null` cuando el navegador no puede saber el total del envio.
  uploadPercent: number | null;
  translationTargets: string[];
  // Los que se pueden bajar. Con la lista vacia de instalados, esto es lo unico
  // que le da al usuario una salida.
  installablePairs: string[];
  onPairInstalled: () => void;
}) {
  const [translateTo, setTranslateTo] = useState("");
  const { t } = useTranslation();
  const [copyState, setCopyState] = useState<"idle" | "copied" | "error">(
    "idle",
  );

  useEffect(() => {
    setCopyState("idle");
  }, [job?.text]);

  async function copyText(): Promise<void> {
    if (!job?.text) {
      return;
    }
    try {
      await navigator.clipboard.writeText(job.text);
      setCopyState("copied");
    } catch {
      setCopyState("error");
    }
  }

  const phaseLabel = t(phaseLabelKey(phase));

  return (
    <aside className="flex min-h-64 flex-col gap-4 rounded border border-border bg-surface p-4">
      <h2 className="font-heading text-lg font-semibold text-text">
        {t("transcribe.result.title")}
      </h2>

      {job?.dubOverflowSegments ? (
        <p
          role="status"
          className="rounded border border-warn bg-surface-2 px-3 py-2 text-xs text-text-dim"
        >
          {t("transcribe.dub.overflow", { count: job.dubOverflowSegments })}
        </p>
      ) : null}

      {phase === "idle" && (
        <p className="text-sm text-text-faint">
          {t("transcribe.result.waiting")}
        </p>
      )}

      {phase === "uploading" && (
        <div className="flex flex-col gap-2">
          <div className="flex items-center justify-between gap-2">
            <p role="status" className="text-sm text-text">
              {phaseLabel}
            </p>
            {uploadPercent !== null && (
              <span className="font-mono-tabular text-sm text-text-dim">
                {uploadPercent}%
              </span>
            )}
          </div>
          {uploadPercent !== null ? (
            <DeterminateProgressBar label={phaseLabel} percent={uploadPercent} />
          ) : (
            <IndeterminateProgressBar label={phaseLabel} />
          )}
        </div>
      )}

      {(phase === "queued" || phase === "running") && (
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between gap-2">
            <p role="status" className="text-sm text-text">
              {phaseLabel}
            </p>
            {job?.progressPct !== null && job?.progressPct !== undefined && (
              <span className="font-mono-tabular text-sm text-text-dim">
                {Math.round(job.progressPct)}%
              </span>
            )}
          </div>
          {job?.progressPct === null || job?.progressPct === undefined ? (
            <IndeterminateProgressBar label={phaseLabel} />
          ) : (
            <DeterminateProgressBar
              label={phaseLabel}
              percent={job.progressPct}
            />
          )}
          <button
            type="button"
            onClick={onCancel}
            className="inline-flex w-fit items-center gap-2 rounded border border-danger px-3 py-1.5 text-sm text-danger hover:bg-surface-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-danger"
          >
            <X aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
            {t("transcribe.job.cancel")}
          </button>
        </div>
      )}

      {phase === "failed" && (
        <p
          role="alert"
          className="rounded border border-danger bg-surface-2 px-3 py-2 text-sm text-danger"
        >
          {errorMessage ?? t("transcribe.job.failedFallback")}
        </p>
      )}

      {phase === "cancelled" && (
        <p role="status" className="text-sm text-warn">
          {t("transcribe.job.cancelled")}
        </p>
      )}

      {phase === "completed" && job && (
        <div className="flex min-h-0 flex-1 flex-col gap-3">
          <div className="max-h-80 min-h-40 overflow-y-auto whitespace-pre-wrap rounded border border-border bg-surface-2 p-4 text-sm leading-6 text-text">
            {job.text || t("transcribe.result.empty")}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => void copyText()}
              className="inline-flex items-center gap-2 rounded border border-border bg-surface px-3 py-1.5 text-sm text-text hover:border-text-faint focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
            >
              {copyState === "copied" ? (
                <Check
                  aria-hidden="true"
                  className="h-4 w-4 text-ok"
                  strokeWidth={1.75}
                />
              ) : (
                <Clipboard
                  aria-hidden="true"
                  className="h-4 w-4"
                  strokeWidth={1.75}
                />
              )}
              {copyState === "copied"
                ? t("transcribe.result.copied")
                : t("transcribe.result.copy")}
            </button>
            {translationTargets.length === 0 && installablePairs.length > 0 && (
              <TranslationPairInstaller
                pairs={installablePairs}
                onDone={onPairInstalled}
              />
            )}
            {translationTargets.length > 0 && (
              <label className="flex items-center gap-2 text-sm text-text-dim">
                {t("transcribe.translate.label")}
                <select
                  value={translateTo}
                  onChange={(event) => setTranslateTo(event.target.value)}
                  className="rounded border border-border bg-surface px-2 py-1 text-sm text-text focus:border-accent focus:outline-none"
                >
                  <option value="">{t("transcribe.translate.none")}</option>
                  {translationTargets.map((code) => (
                    <option key={code} value={code}>
                      {code}
                    </option>
                  ))}
                </select>
              </label>
            )}
            {job.downloadUrl && (
              <a
                href={subtitleHref(job.id, translateTo)}
                download
                className="inline-flex items-center gap-2 rounded border border-border bg-surface px-3 py-1.5 text-sm text-text hover:border-text-faint focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
              >
                <Download aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
                {t("transcribe.result.downloadSubtitles")}
              </a>
            )}
            {job.videoUrl && (
              <a
                href={job.videoUrl}
                download
                className="inline-flex items-center gap-2 rounded border border-border bg-surface px-3 py-1.5 text-sm text-text hover:border-text-faint focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
              >
                <Download aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
                {t("transcribe.result.downloadVideo")}
              </a>
            )}
            {job.downloadUrl && (
              <a
                href={job.downloadUrl}
                download
                className="inline-flex items-center gap-2 rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg hover:bg-accent-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
              >
                <Download
                  aria-hidden="true"
                  className="h-4 w-4"
                  strokeWidth={1.75}
                />
                {t("transcribe.result.download")}
              </a>
            )}
            {copyState === "error" && (
              <span role="alert" className="text-xs text-danger">
                {t("transcribe.result.copyFailed")}
              </span>
            )}
          </div>
        </div>
      )}
    </aside>
  );
}

/**
 * Elegí un par de idiomas y bajalo.
 *
 * OPUS-MT publica un modelo POR PAR, así que "instalar la traducción" no es una
 * sola cosa. Antes, con ninguno instalado, la pantalla escondía la traducción
 * entera y no había forma de llegar a ella.
 */
function TranslationPairInstaller({
  pairs,
  onDone,
}: {
  pairs: string[];
  onDone: () => void;
}) {
  const { t } = useTranslation();
  const [par, setPar] = useState(pairs[0] ?? "");

  return (
    <div className="flex flex-col gap-2 rounded border border-border bg-surface-2 p-3">
      <p className="text-sm text-text-dim">{t("transcribe.translate.noPairs")}</p>
      <label className="flex items-center gap-2 text-sm text-text-dim">
        {t("transcribe.translate.pickPair")}
        <select
          value={par}
          onChange={(event) => setPar(event.target.value)}
          aria-label={t("transcribe.translate.pickPair")}
          className="rounded border border-border bg-surface px-2 py-1 text-sm text-text focus:border-accent focus:outline-none"
        >
          {pairs.map((codigo) => (
            <option key={codigo} value={codigo}>
              {codigo}
            </option>
          ))}
        </select>
      </label>
      {par && (
        <PackDownload key={par} pack="translation" variant={par} onDone={onDone} />
      )}
    </div>
  );
}

export function TranscribePanel({
  pollIntervalMs = DEFAULT_TRANSCRIBE_POLL_INTERVAL_MS,
  installPollIntervalMs = DEFAULT_ASR_INSTALL_POLL_INTERVAL_MS,
  searchDebounceMs,
}: TranscribePanelProps) {
  const { t } = useTranslation();
  const [files, setFiles] = useState<File[]>([]);
  // El primero manda para lo que depende del archivo (si tiene imagen, y por
  // lo tanto si se puede pedir video con subtitulos); el lote usa todos.
  const file = files[0] ?? null;
  const [modelId, setModelId] = useState("");
  const [language, setLanguage] = useState("");
  const [outputMode, setOutputMode] = useState<TranscribeOutputMode>("text");
  const [dubLanguage, setDubLanguage] = useState("");
  const [deviceId, setDeviceId] = useState("cpu");
  const modelsQuery = useInstalledAsrModels();
  const devicesQuery = useTranscribeDevices();
  const translationPairsQuery = useQuery({
    queryKey: ["translation-pairs"],
    queryFn: fetchTranslationPairs,
  });
  const translationTargets = [
    ...new Set((translationPairsQuery.data?.pairs ?? []).map((pair) => pair.target)),
  ];
  // Los pares que se pueden bajar. Con cero instalados la traduccion quedaba
  // escondida y sin forma de conseguirse: un callejon sin salida silencioso.
  const paresInstalables = translationPairsQuery.data?.installable ?? [];
  const {
    phase,
    job,
    errorMessage,
    submit,
    submitMany,
    pendingUploads,
    failedUploads,
    cancel,
    reset,
    uploadPercent,
  } =
    useTranscribeJob(pollIntervalMs);

  const models = modelsQuery.data ?? [];
  const soloIngles = isEnglishOnly(
    models.find((model) => model.id === modelId)?.name ?? modelId,
  );
  const selectedModelId = models.some((model) => model.id === modelId)
    ? modelId
    : (models[0]?.id ?? "");
  const devices = devicesQuery.data?.devices ?? [];
  const selectedDeviceId = devices.some((device) => device.id === deviceId)
    ? deviceId
    : devices.some(
          (device) => device.id === devicesQuery.data?.defaultDeviceId,
        )
      ? (devicesQuery.data?.defaultDeviceId ?? "")
      : (devices[0]?.id ?? "");

  function handleFilesSelected(elegidos: File[]): void {
    setFiles(elegidos);
    reset();
  }

  function handleSubmit(): void {
    if (!file || !selectedModelId || isJobBusy(phase)) {
      return;
    }
    const params: CreateTranscribeJobParams = {
      file,
      modelId: selectedModelId,
    };
    if (soloIngles) {
      params.language = "en";
    } else if (language) {
      params.language = language;
    }
    if (selectedDeviceId) {
      params.device = selectedDeviceId;
    }
    // Un audio no tiene imagen: pedir video ahi seria mandar al backend a
    // fallar por algo que la pantalla ya sabe.
    if (outputMode !== "text" && hasPicture(file)) {
      params.outputMode = outputMode;
    }
    if (params.outputMode === "dubbed_video") {
      params.targetLanguage = dubLanguage || translationTargets[0] || "";
    }
    if (files.length > 1) {
      // Un trabajo por archivo con los mismos ajustes. `outputMode` se resuelve
      // POR archivo: pedir video con subtitulos de un .wav no tiene sentido, y
      // en un lote mezclado el usuario elige una sola vez.
      submitMany(
        files.map((archivo) => ({
          ...params,
          file: archivo,
          ...(params.outputMode && !hasPicture(archivo) ? { outputMode: undefined } : {}),
        })),
      );
      return;
    }
    submit(params);
  }

  const canSubmit =
    file !== null && selectedModelId.length > 0 && !isJobBusy(phase);

  return (
    <div className="flex flex-col gap-8">
      <div className="grid grid-cols-[1fr_360px] gap-6 max-[900px]:grid-cols-1">
        <div className="flex flex-col gap-5">
          {modelsQuery.isLoading && (
            <p role="status" className="text-sm text-text-dim">
              {t("transcribe.models.loadingInstalled")}
            </p>
          )}

          {modelsQuery.isError && (
            <p
              role="alert"
              className="rounded border border-danger bg-surface px-3 py-2 text-sm text-danger"
            >
              {t("transcribe.models.loadInstalledFailed")}
            </p>
          )}

          {!modelsQuery.isLoading &&
            !modelsQuery.isError &&
            models.length === 0 && (
              <div className="rounded border border-warn bg-surface p-5">
                <h2 className="font-heading text-lg font-semibold text-text">
                  {t("transcribe.models.noneTitle")}
                </h2>
                <p className="mt-2 text-sm text-text-dim">
                  {t("transcribe.models.noneDescription")}
                </p>
              </div>
            )}

          {models.length > 0 && (
            <>
              <AudioDropzone
                files={files}
                onFilesSelected={handleFilesSelected}
              />

              <label className="flex flex-col gap-2">
                <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
                  {t("transcribe.model.label")}
                </span>
                <select
                  value={selectedModelId}
                  onChange={(event) => setModelId(event.target.value)}
                  className="rounded border border-border bg-surface px-3 py-2 text-sm text-text focus:border-accent focus:outline-none"
                >
                  {models.map((model) => (
                    <option key={model.id} value={model.id}>
                      {model.name}
                    </option>
                  ))}
                </select>
              </label>

              <label className="flex flex-col gap-2">
                <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
                  {t("transcribe.language.label")}
                </span>
                <select
                  value={soloIngles ? "en" : language}
                  disabled={soloIngles}
                  onChange={(event) => setLanguage(event.target.value)}
                  className="rounded border border-border bg-surface px-3 py-2 text-sm text-text focus:border-accent focus:outline-none disabled:opacity-50"
                >
                  <option value="">{t("transcribe.language.auto")}</option>
                  {LANGUAGE_OPTIONS.map((code) => (
                    <option key={code} value={code}>
                      {t(`transcribe.language.${code}`)}
                    </option>
                  ))}
                </select>
                {/* Ofrecer un idioma que el modelo no puede hacer no es una
                    opcion: la transcripcion falla, y antes de fallar el usuario
                    ya subio el archivo y espero. */}
                {soloIngles && (
                  <span className="text-xs text-warn">{t("transcribe.language.englishOnly")}</span>
                )}
              </label>

              {hasPicture(file) && (
                <label className="flex flex-col gap-2">
                  <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
                    {t("transcribe.output.label")}
                  </span>
                  <select
                    value={outputMode}
                    onChange={(event) =>
                      setOutputMode(event.target.value as TranscribeOutputMode)
                    }
                    className="rounded border border-border bg-surface px-3 py-2 text-sm text-text focus:border-accent focus:outline-none"
                  >
                    <option value="text">{t("transcribe.output.text")}</option>
                    <option value="video">{t("transcribe.output.video")}</option>
                    <option value="video_burned">
                      {t("transcribe.output.videoBurned")}
                    </option>
                    {translationTargets.length > 0 && (
                      <option value="dubbed_video">
                        {t("transcribe.output.dubbed")}
                      </option>
                    )}
                  </select>
                  {outputMode === "video_burned" && (
                    <span className="text-xs text-text-faint">
                      {t("transcribe.output.hint")}
                    </span>
                  )}
                  {outputMode === "dubbed_video" && (
                    <span className="text-xs text-text-faint">
                      {t("transcribe.dub.hint")}
                    </span>
                  )}
                </label>
              )}

              {hasPicture(file) && outputMode === "dubbed_video" && (
                <label className="flex flex-col gap-2">
                  <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
                    {t("transcribe.dub.language")}
                  </span>
                  <select
                    value={dubLanguage}
                    onChange={(event) => setDubLanguage(event.target.value)}
                    className="rounded border border-border bg-surface px-3 py-2 text-sm text-text focus:border-accent focus:outline-none"
                  >
                    {translationTargets.map((code) => (
                      <option key={code} value={code}>
                        {code}
                      </option>
                    ))}
                  </select>
                </label>
              )}

              <label className="flex flex-col gap-2">
                <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
                  {t("transcribe.device.label")}
                </span>
                {devicesQuery.isLoading ? (
                  <span role="status" className="text-sm text-text-dim">
                    {t("transcribe.device.loading")}
                  </span>
                ) : devicesQuery.isError ? (
                  <span role="alert" className="text-sm text-danger">
                    {t("transcribe.device.loadFailed")}
                  </span>
                ) : (
                  <select
                    value={selectedDeviceId}
                    onChange={(event) => setDeviceId(event.target.value)}
                    className="rounded border border-border bg-surface px-3 py-2 text-sm text-text focus:border-accent focus:outline-none"
                  >
                    {devices.map((device) => (
                      <option key={device.id} value={device.id}>
                        {device.name}
                      </option>
                    ))}
                  </select>
                )}
              </label>

              <button
                type="button"
                disabled={!canSubmit}
                onClick={handleSubmit}
                className="inline-flex w-fit items-center gap-2 rounded bg-accent px-4 py-2 text-sm font-medium text-bg transition-[background-color,opacity] duration-fast hover:bg-accent-hover active:bg-accent-press disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
              >
                <AudioLines
                  aria-hidden="true"
                  className="h-4 w-4"
                  strokeWidth={1.75}
                />
                {t("transcribe.submit")}
              </button>
            </>
          )}
        </div>

        <TranscriptionResult
          phase={phase}
          job={job}
          errorMessage={errorMessage}
          onCancel={cancel}
          uploadPercent={uploadPercent}
          translationTargets={translationTargets}
          installablePairs={paresInstalables}
          onPairInstalled={() => void translationPairsQuery.refetch()}
        />
        {pendingUploads > 0 && (
          <p role="status" className="text-xs text-text-dim">
            {t("enhance.batch.pending", { count: pendingUploads })}
          </p>
        )}
        {pendingUploads === 0 && failedUploads > 0 && (
          <p role="alert" className="text-xs text-danger">
            {t(failedUploads === 1 ? "enhance.batch.failedOne" : "enhance.batch.failed", {
              count: failedUploads,
            })}
          </p>
        )}
      </div>

      <AsrModelSearch
        debounceMs={searchDebounceMs}
        installPollIntervalMs={installPollIntervalMs}
      />
    </div>
  );
}
