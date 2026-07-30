import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Download as DownloadIcon, Loader2, Music, Video } from "lucide-react";
import { useState } from "react";
import { DeterminateProgressBar } from "../components/DeterminateProgressBar";
import { IndeterminateProgressBar } from "../components/IndeterminateProgressBar";
import type { DownloadJob, MediaProbe } from "../lib/apiTypes";
import {
  cancelDownloadJob,
  createDownloadJob,
  getDownloadJob,
  probeMedia,
} from "../services/download";
import {
  DEFAULT_HEIGHT,
  clampPlaylistLimit,
  formatBytes,
  formatDuration,
  isProbablyUrl,
  offeredHeights,
  playlistNotice,
} from "../modules/download/downloadRequest";

const TERMINAL = new Set(["completed", "failed", "cancelled"]);

function ProbeSummary({ probe }: { probe: MediaProbe }) {
  return (
    <div className="flex flex-col gap-1 rounded border border-border bg-surface-2 px-3 py-2">
      <span className="text-sm text-text">{probe.title}</span>
      <span className="font-mono-tabular text-xs text-text-dim">
        {probe.uploader ?? "—"} · {formatDuration(probe.durationSeconds)} · {probe.extractor}
      </span>
    </div>
  );
}

function JobProgress({ job, onCancel }: { job: DownloadJob; onCancel: () => void }) {
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
            <IndeterminateProgressBar label="Descargando" />
          ) : (
            <DeterminateProgressBar label="Descargando" percent={job.progressPct} />
          )}
          <span className="font-mono-tabular text-xs text-text-dim">
            {formatBytes(job.downloadedBytes)} / {formatBytes(job.totalBytes)}
          </span>
        </>
      )}

      {job.status === "completed" && job.outputFiles.length > 0 && (
        <span className="text-xs text-text-dim">
          Listo en tus archivos: {job.outputFiles.join(", ")} — ya podés escalarlo o limpiarle
          el audio desde Enhance.
        </span>
      )}

      {job.error && <span className="text-xs text-danger">{job.error}</span>}

      {!done && (
        <button type="button" onClick={onCancel} className="self-start rounded border border-border bg-surface px-3 py-1.5 text-sm text-text-dim hover:border-text-faint">Cancelar</button>
      )}
    </div>
  );
}

export function DownloadPage() {
  const queryClient = useQueryClient();
  const [url, setUrl] = useState("");
  const [maxHeight, setMaxHeight] = useState(DEFAULT_HEIGHT);
  const [audioOnly, setAudioOnly] = useState(false);
  const [includePlaylist, setIncludePlaylist] = useState(false);
  const [playlistLimit, setPlaylistLimit] = useState(10);
  const [jobId, setJobId] = useState<string | null>(null);

  const probe = useMutation({ mutationFn: () => probeMedia({ url }) });

  const create = useMutation({
    mutationFn: () =>
      createDownloadJob({
        url,
        maxHeight,
        audioOnly,
        includePlaylist,
        playlistLimit: clampPlaylistLimit(playlistLimit),
      }),
    onSuccess: (job) => setJobId(job.id),
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

  return (
    <div className="flex flex-col gap-4">
      <section className="rounded border border-border bg-surface p-5">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <label htmlFor="download-url" className="text-xs font-medium text-text-dim">
              Dirección del video
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
              <button
                type="button"
                onClick={() => probe.mutate()}
                disabled={!urlLooksValid || probe.isPending}
                className="rounded border px-3 py-1.5 text-sm transition-[background-color,border-color] duration-fast disabled:opacity-50 disabled:cursor-not-allowed border-border bg-surface text-text-dim hover:border-text-faint"
              >
                {probe.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : "Ver qué hay"}
              </button>
            </div>
            {probe.isError && (
              <span className="text-xs text-danger">{(probe.error as Error).message}</span>
            )}
          </div>

          {probeData && <ProbeSummary probe={probeData} />}

          {notice && (
            <div className="rounded border border-warn bg-surface-2 px-3 py-2 text-xs text-warn">
              Es una lista de {notice.entryCount} elementos. Se van a descargar{" "}
              <strong>{notice.willDownload}</strong>.
              {notice.needsConfirmation && " Revisá el límite antes de seguir."}
            </div>
          )}

          <fieldset className="flex flex-col gap-2">
            <legend className="text-xs font-medium text-text-dim">Qué traer</legend>
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

          {!audioOnly && (
            <fieldset className="flex flex-col gap-2">
              <legend className="text-xs font-medium text-text-dim">Calidad máxima</legend>
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

          {probeData?.isPlaylist && (
            <div className="flex items-center gap-3">
              <label className="flex items-center gap-2 text-sm text-text">
                <input
                  type="checkbox"
                  checked={includePlaylist}
                  onChange={(event) => setIncludePlaylist(event.target.checked)}
                  className="h-3.5 w-3.5 accent-accent"
                />
                Bajar la lista completa
              </label>
              {includePlaylist && (
                <label className="flex items-center gap-2 text-xs text-text-dim">
                  Máximo
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
            onClick={() => create.mutate()}
            disabled={!urlLooksValid || create.isPending}
            className="self-start flex items-center gap-1.5 rounded border px-3 py-1.5 text-sm transition-[background-color,border-color] duration-fast disabled:opacity-50 disabled:cursor-not-allowed border-accent bg-surface-2 text-text"
          >
            <DownloadIcon className="h-4 w-4" /> Descargar
          </button>
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
