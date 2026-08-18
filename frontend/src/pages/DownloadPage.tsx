import { isMultiUrl, parseUrlList } from "../modules/download/urlList";
import { uploadSequentially } from "../lib/sequentialUploads";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download as DownloadIcon, FolderOpen, Loader2, Music, Video } from "lucide-react";
import { useState } from "react";
import { useDebouncedValue } from "../hooks/useDebouncedValue";
import { useTranslation } from "../i18n/LocaleProvider";
import { DeterminateProgressBar } from "../components/DeterminateProgressBar";
import { IndeterminateProgressBar } from "../components/IndeterminateProgressBar";
import type { DownloadJob, MediaProbe } from "../lib/apiTypes";
import { jobQueueStore } from "../lib/jobQueueStore";
import {
  cancelDownloadJob,
  createDownloadJob,
  getDownloadJob,
  probeMedia,
} from "../services/download";
import {
  AUDIO_BITRATE_OPTIONS,
  AUDIO_FORMAT_OPTIONS,
  DEFAULT_HEIGHT,
  VIDEO_CONTAINER_OPTIONS,
  bitrateSelectable,
  clampPlaylistLimit,
  downloadFileHref,
  formatBytes,
  formatDuration,
  isProbablyUrl,
  offeredHeights,
  playlistNotice,
  type AudioFormat,
  type VideoContainer,
} from "../modules/download/downloadRequest";

const TERMINAL = new Set(["completed", "failed", "cancelled"]);

function ProbeSummary({ probe }: { probe: MediaProbe }) {
  const { t } = useTranslation();
  return (
    <div className="flex items-center gap-3 rounded border border-border bg-surface-2 px-3 py-2">
      {probe.thumbnailUrl && (
        <img
          src={probe.thumbnailUrl}
          alt={t("download.thumbnailAlt", { title: probe.title })}
          className="h-14 w-24 shrink-0 rounded object-cover"
        />
      )}
      <div className="flex min-w-0 flex-col gap-1">
        <span className="truncate text-sm text-text">{probe.title}</span>
        <span className="font-mono-tabular text-xs text-text-dim">
          {probe.uploader ?? "—"} · {formatDuration(probe.durationSeconds)} · {probe.extractor}
        </span>
      </div>
    </div>
  );
}

function JobProgress({ job, onCancel }: { job: DownloadJob; onCancel: () => void }) {
  const { t } = useTranslation();
  const done = TERMINAL.has(job.status);
  return (
    <div className="flex flex-col gap-2 rounded border border-border bg-surface px-3 py-2">
      <div className="flex items-center justify-between gap-2">
        <span className="truncate text-sm text-text">{job.mediaTitle ?? job.url}</span>
        <span className="font-mono-tabular text-xs text-text-dim">{job.status}</span>
      </div>

      {job.status === "running" && (
        <>
          {/* Sin total conocido la barra va indeterminada: algunos sitios no publican el
              tamaño, y dibujar un porcentaje inventado sería mentir sobre lo que falta. */}
          {job.progressPct === null ? (
            <IndeterminateProgressBar label={t("download.downloading")} />
          ) : (
            <DeterminateProgressBar label={t("download.downloading")} percent={job.progressPct} />
          )}
          <span className="font-mono-tabular text-xs text-text-dim">
            {formatBytes(job.downloadedBytes)} / {formatBytes(job.totalBytes)}
          </span>
        </>
      )}

      {job.status === "completed" && job.outputFiles.length > 0 && (
        <div className="flex flex-col gap-1">
          {/* Un link por archivo, servido por el backend: el nombre y la carpeta del
              server no alcanzan si no estás sentado en el server. */}
          <div className="flex flex-wrap gap-2">
            {job.outputFiles.map((name, index) => (
              <a
                key={name}
                href={downloadFileHref(job.id, index)}
                download
                className="flex items-center gap-1.5 rounded border border-border bg-surface px-3 py-1.5 text-sm text-text hover:border-accent"
              >
                <DownloadIcon className="h-3.5 w-3.5" /> {name}
              </a>
            ))}
          </div>
          <span className="text-xs text-text-dim">
            {t("download.readyForEnhance")}
          </span>
          {/* La carpeta, no solo el nombre: decir el nombre sin decir donde quedo
              obliga a salir a buscarlo. */}
          {job.outputDirectory && (
            <span className="flex items-center gap-1.5 font-mono-tabular text-xs text-text-faint">
              <FolderOpen className="h-3 w-3" /> {job.outputDirectory}
            </span>
          )}
        </div>
      )}

      {job.error && <span className="text-xs text-danger">{job.error}</span>}

      {!done && (
        <button type="button" onClick={onCancel} className="self-start rounded border border-border bg-surface px-3 py-1.5 text-sm text-text-dim hover:border-text-faint">{t("download.cancel")}</button>
      )}
    </div>
  );
}

export function DownloadPage() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [url, setUrl] = useState("");
  const [maxHeight, setMaxHeight] = useState(DEFAULT_HEIGHT);
  const [audioOnly, setAudioOnly] = useState(false);
  const [audioFormat, setAudioFormat] = useState<AudioFormat>("mp3");
  const [audioBitrate, setAudioBitrate] = useState<number | null>(null);
  const [videoContainer, setVideoContainer] = useState<VideoContainer>("mp4");
  const [includePlaylist, setIncludePlaylist] = useState(false);
  const [playlistLimit, setPlaylistLimit] = useState(10);
  const [jobId, setJobId] = useState<string | null>(null);
  const [pendientes, setPendientes] = useState(0);
  const [fallidas, setFallidas] = useState(0);

  // Automatico y no un boton: las calidades solo pueden reflejar la realidad si el
  // probe corrio, y con un boton manual la lista se ve "quemada" cuando nadie lo toca.
  // Con debounce para no pedirle al sitio una consulta por cada tecla -- ese exceso es
  // justo lo que hace que YouTube corte la sesion por una hora.
  const settledUrl = useDebouncedValue(url, 700);
  const probe = useQuery({
    queryKey: ["download-probe", settledUrl],
    queryFn: () => probeMedia({ url: settledUrl }),
    enabled: isProbablyUrl(settledUrl),
    retry: false,
    staleTime: 5 * 60 * 1000,
  });

  function opcionesComunes(destino: string) {
    return {
      url: destino,
      maxHeight,
      audioOnly,
      audioFormat,
      audioBitrateKbps: bitrateSelectable(audioFormat) ? audioBitrate : null,
      videoContainer,
      includePlaylist,
      playlistLimit: clampPlaylistLimit(playlistLimit),
    };
  }

  /** Varias URLs pegadas: una descarga por línea, de a una.
   *
   * De a una y no todas juntas por lo mismo que los archivos: en paralelo se
   * pelean el ancho de banda y la cola del servidor rechaza las últimas. La
   * primera se sigue en vivo en esta pantalla; el resto entra a la cola global.
   */
  async function encolarVarias(destinos: string[]) {
    const [primera, ...resto] = destinos;
    setPendientes(resto.length);
    setFallidas(0);
    await create.mutateAsync(primera).catch(() => undefined);
    const { failed } = await uploadSequentially(
      resto,
      async (destino) => {
        const job = await createDownloadJob(opcionesComunes(destino));
        jobQueueStore.addTrackedJob({
          id: job.id,
          kind: "download",
          fileName: job.mediaTitle ?? job.url,
          createdAt: Date.now(),
        });
      },
      setPendientes,
    );
    setFallidas(failed);
  }

  const create = useMutation({
    mutationFn: (destino: string = url) =>
      createDownloadJob({
        ...opcionesComunes(destino),
      }),
    onSuccess: (job) => {
      setJobId(job.id);
      // La descarga es una cola mas del programa: sin esto no aparecia en la
      // lista lateral ni tenia detalle, aunque el trabajo fuera igual de largo.
      jobQueueStore.addTrackedJob({
        id: job.id,
        kind: "download",
        fileName: job.mediaTitle ?? job.url,
        createdAt: Date.now(),
      });
    },
  });

  const jobQuery = useQuery({
    queryKey: ["download-job", jobId],
    queryFn: () => getDownloadJob(jobId as string),
    enabled: jobId !== null,
    refetchInterval: (query) =>
      query.state.data && TERMINAL.has(query.state.data.status) ? false : 1000,
  });

  const cancel = useMutation({
    mutationFn: () => cancelDownloadJob(jobId as string),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["download-job", jobId] }),
  });

  const probeData = probe.data ?? null;
  const notice = playlistNotice(probeData, includePlaylist, playlistLimit);
  const heights = offeredHeights(probeData);
  const urlLooksValid = isProbablyUrl(url);
  const varias = isMultiUrl(url);
  // Con una lista pegada, la vista previa no aplica (es por URL) y alcanza con
  // que la primera linea tenga forma de URL para habilitar el boton.
  const puedeDescargar = varias ? isProbablyUrl(parseUrlList(url)[0] ?? "") : urlLooksValid;

  return (
    <div className="flex flex-col gap-4">
      <section className="rounded border border-border bg-surface p-5">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <label htmlFor="download-url" className="text-xs font-medium text-text-dim">
              {t("download.url")}
            </label>
            <div className="flex gap-2">
              <input
                id="download-url"
                type="url"
                value={url}
                onChange={(event) => setUrl(event.target.value)}
                placeholder="https://…"
                className="flex-1 rounded border border-border bg-surface px-3 py-2 text-sm text-text"
              />
            </div>
            {probe.isFetching && (
              <span className="flex items-center gap-1.5 text-xs text-text-dim">
                <Loader2 className="h-3 w-3 animate-spin" /> {t("download.probing")}
              </span>
            )}
            {probe.isError && (
              <span className="text-xs text-danger">{(probe.error as Error).message}</span>
            )}
          </div>

          {probeData && <ProbeSummary probe={probeData} />}

          {notice && (
            <div className="rounded border border-warn bg-surface-2 px-3 py-2 text-xs text-warn">
              {t("download.playlistNotice", {
                count: notice.entryCount,
                willDownload: notice.willDownload,
              })}
              {notice.needsConfirmation && t("download.playlistOverLimit")}
            </div>
          )}

          <fieldset className="flex flex-col gap-2">
            <legend className="text-xs font-medium text-text-dim">{t("download.whatToFetch")}</legend>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setAudioOnly(false)}
                className={`flex items-center gap-1.5 ${audioOnly ? "rounded border px-3 py-1.5 text-sm transition-[background-color,border-color] duration-fast disabled:opacity-50 disabled:cursor-not-allowed border-border bg-surface text-text-dim hover:border-text-faint" : "rounded border px-3 py-1.5 text-sm transition-[background-color,border-color] duration-fast disabled:opacity-50 disabled:cursor-not-allowed border-accent bg-surface-2 text-text"}`}
              >
                <Video className="h-4 w-4" /> Video
              </button>
              <button
                type="button"
                onClick={() => setAudioOnly(true)}
                className={`flex items-center gap-1.5 ${audioOnly ? "rounded border px-3 py-1.5 text-sm transition-[background-color,border-color] duration-fast disabled:opacity-50 disabled:cursor-not-allowed border-accent bg-surface-2 text-text" : "rounded border px-3 py-1.5 text-sm transition-[background-color,border-color] duration-fast disabled:opacity-50 disabled:cursor-not-allowed border-border bg-surface text-text-dim hover:border-text-faint"}`}
              >
                <Music className="h-4 w-4" /> Solo audio
              </button>
            </div>
          </fieldset>

          {audioOnly && (
            <fieldset className="flex flex-col gap-2">
              <legend className="text-xs font-medium text-text-dim">Formato</legend>
              <div className="flex flex-wrap gap-2">
                {AUDIO_FORMAT_OPTIONS.map((format) => (
                  <label
                    key={format}
                    className={`cursor-pointer rounded border px-3 py-1.5 text-sm ${
                      audioFormat === format
                        ? "border-accent bg-surface-2 text-text"
                        : "border-border bg-surface text-text-dim"
                    }`}
                  >
                    <input
                      type="radio"
                      name="download-audio-format"
                      className="sr-only"
                      checked={audioFormat === format}
                      onChange={() => setAudioFormat(format)}
                    />
                    <span className="font-mono-tabular">{format.toUpperCase()}</span>
                  </label>
                ))}
              </div>
            </fieldset>
          )}

          {audioOnly && bitrateSelectable(audioFormat) && (
            <fieldset className="flex flex-col gap-2">
              <legend className="text-xs font-medium text-text-dim">{t("download.audioQuality")}</legend>
              <div className="flex flex-wrap gap-2">
                {AUDIO_BITRATE_OPTIONS.map((bitrate) => (
                  <label
                    key={bitrate ?? "vbr"}
                    className={`cursor-pointer rounded border px-3 py-1.5 text-sm ${
                      audioBitrate === bitrate
                        ? "border-accent bg-surface-2 text-text"
                        : "border-border bg-surface text-text-dim"
                    }`}
                  >
                    <input
                      type="radio"
                      name="download-audio-bitrate"
                      className="sr-only"
                      checked={audioBitrate === bitrate}
                      onChange={() => setAudioBitrate(bitrate)}
                    />
                    <span className="font-mono-tabular">
                      {bitrate === null ? t("download.audioQuality.best") : `${bitrate} kbps`}
                    </span>
                  </label>
                ))}
              </div>
            </fieldset>
          )}

          {!audioOnly && (
            <fieldset className="flex flex-col gap-2">
              <legend className="text-xs font-medium text-text-dim">{t("download.maxQuality")}</legend>
              <div className="flex flex-wrap gap-2">
                {heights.map((height) => (
                  <label
                    key={height}
                    className={`cursor-pointer rounded border px-3 py-1.5 text-sm ${
                      maxHeight === height
                        ? "border-accent bg-surface-2 text-text"
                        : "border-border bg-surface text-text-dim"
                    }`}
                  >
                    <input
                      type="radio"
                      name="download-height"
                      className="sr-only"
                      checked={maxHeight === height}
                      onChange={() => setMaxHeight(height)}
                    />
                    <span className="font-mono-tabular">{height}p</span>
                  </label>
                ))}
              </div>
            </fieldset>
          )}

          {!audioOnly && (
            <fieldset className="flex flex-col gap-2">
              <legend className="text-xs font-medium text-text-dim">Contenedor</legend>
              <div className="flex gap-2">
                {VIDEO_CONTAINER_OPTIONS.map((container) => (
                  <label
                    key={container}
                    className={`cursor-pointer rounded border px-3 py-1.5 text-sm ${
                      videoContainer === container
                        ? "border-accent bg-surface-2 text-text"
                        : "border-border bg-surface text-text-dim"
                    }`}
                  >
                    <input
                      type="radio"
                      name="download-container"
                      className="sr-only"
                      checked={videoContainer === container}
                      onChange={() => setVideoContainer(container)}
                    />
                    <span className="font-mono-tabular">{container.toUpperCase()}</span>
                  </label>
                ))}
              </div>
            </fieldset>
          )}

          {probeData?.isPlaylist && (
            <div className="flex items-center gap-3">
              <label className="flex items-center gap-2 text-sm text-text">
                <input
                  type="checkbox"
                  checked={includePlaylist}
                  onChange={(event) => setIncludePlaylist(event.target.checked)}
                  className="h-3.5 w-3.5 accent-accent"
                />
                {t("download.wholeList")}
              </label>
              {includePlaylist && (
                <label className="flex items-center gap-2 text-xs text-text-dim">
                  {t("download.listMax")}
                  <input
                    type="number"
                    min={1}
                    max={50}
                    value={playlistLimit}
                    onChange={(event) => setPlaylistLimit(Number(event.target.value))}
                    className="w-16 rounded border border-border bg-surface px-2 py-1 font-mono-tabular text-text"
                  />
                </label>
              )}
            </div>
          )}

          <button
            type="button"
            onClick={() => {
              const destinos = parseUrlList(url);
              if (destinos.length > 1) {
                void encolarVarias(destinos);
                return;
              }
              create.mutate(url);
            }}
            disabled={!puedeDescargar || create.isPending}
            className="self-start flex items-center gap-1.5 rounded border px-3 py-1.5 text-sm transition-[background-color,border-color] duration-fast disabled:opacity-50 disabled:cursor-not-allowed border-accent bg-surface-2 text-text"
          >
            <DownloadIcon className="h-4 w-4" />{" "}
            {varias ? t("download.submitMany", { count: parseUrlList(url).length }) : t("download.submit")}
          </button>
          {pendientes > 0 && (
            <p role="status" className="text-xs text-text-dim">
              {t("download.pending", { count: pendientes })}
            </p>
          )}
          {pendientes === 0 && fallidas > 0 && (
            <p role="alert" className="text-xs text-danger">
              {t("download.failedMany", { count: fallidas })}
            </p>
          )}
          {create.isError && (
            <span className="text-xs text-danger">{(create.error as Error).message}</span>
          )}
        </div>
      </section>

      {jobQuery.data && (
        <section className="rounded border border-border bg-surface p-5">
          <JobProgress job={jobQuery.data} onCancel={() => cancel.mutate()} />
        </section>
      )}
    </div>
  );
}
