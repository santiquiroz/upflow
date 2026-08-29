import { useState, type ChangeEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { UploadCloud } from "lucide-react";
import { useTranslation } from "../../i18n/LocaleProvider";
import { RecommendedAsrCard } from "../../components/RecommendedAsrCard";
import { isEnglishOnly } from "../../lib/englishOnly";
import { useAudioCapabilities } from "../../hooks/useAudioJob";
import {
  useInstalledAsrModels,
  useTranscribeDevices,
} from "../../hooks/useTranscribeJob";
import { fetchTranslationPairs } from "../../services/transcribe";
import {
  cancelKaraokeJob,
  createKaraokeJob,
  getKaraokeJob,
  renderKaraokeJob,
  updateKaraokeLyrics,
  type KaraokeBackgroundKind,
  type KaraokeJob,
  type KaraokeLyricEdit,
  type KaraokeLyricLine,
  type KaraokeSinger,
} from "../../services/karaoke";
import { useCleanupSelection } from "../audio/useCleanupSelection";

const POLL_INTERVAL_MS = 1500;
const LANGUAGE_OPTIONS = ["es", "en", "pt", "fr", "de", "it", "ja"] as const;
const SINGER_COUNT_OPTIONS = [2, 3, 4] as const;
// Paleta inicial por cantante (editable en el review): colores saturados y
// distintos entre si para que chip y subtitulo se reconozcan de una.
const DEFAULT_SINGER_COLORS = ["#FF5252", "#40C4FF", "#FFD740", "#69F0AE"];

const inputClassName =
  "rounded border border-border bg-surface px-3 py-2 text-sm text-text focus:border-accent focus:outline-none";
const labelClassName =
  "font-heading text-xs font-semibold uppercase tracking-wide text-text-dim";

function FileDropzone({
  file,
  onFileSelected,
}: {
  file: File | null;
  onFileSelected: (file: File) => void;
}) {
  const { t } = useTranslation();

  function handleChange(event: ChangeEvent<HTMLInputElement>): void {
    const selected = event.target.files?.[0];
    if (selected) {
      onFileSelected(selected);
    }
  }

  return (
    <label
      htmlFor="karaoke-file-input"
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        event.preventDefault();
        const dropped = event.dataTransfer.files?.[0];
        if (dropped) {
          onFileSelected(dropped);
        }
      }}
      className="flex cursor-pointer flex-col items-center gap-2 rounded border border-dashed border-border bg-surface px-6 py-10 text-center transition-[border-color] duration-fast hover:border-accent"
    >
      <UploadCloud aria-hidden="true" className="h-6 w-6 text-text-faint" strokeWidth={1.5} />
      <span className="text-sm text-text">
        {file === null ? t("karaoke.file.drop") : file.name}
      </span>
      <span className="text-xs text-text-faint">{t("karaoke.file.formats")}</span>
      <input
        id="karaoke-file-input"
        type="file"
        accept="audio/*,video/*"
        aria-label={t("karaoke.file.inputLabel")}
        className="sr-only"
        onChange={handleChange}
      />
    </label>
  );
}

function LyricsEditor({
  job,
  singerColors,
  onSaved,
}: {
  job: KaraokeJob;
  singerColors: Record<string, string>;
  onSaved: () => void;
}) {
  const { t } = useTranslation();
  const [edits, setEdits] = useState<Record<number, KaraokeLyricEdit>>({});
  const [names, setNames] = useState<Record<string, string>>({});
  const hasTranslation = job.translateTo !== null;
  const hasSingers = job.singers.length > 0;

  const save = useMutation({
    mutationFn: () =>
      updateKaraokeLyrics(job.id, Object.values(edits), renamedSingers()),
    onSuccess: () => {
      setEdits({});
      setNames({});
      onSaved();
    },
  });

  function renamedSingers(): KaraokeSinger[] | undefined {
    if (Object.keys(names).length === 0) {
      return undefined;
    }
    return job.singers.map((cantante) => ({
      id: cantante.id,
      label: names[cantante.id] ?? cantante.label,
    }));
  }

  function edit(
    index: number,
    campo: "text" | "translation" | "singer",
    value: string,
  ): void {
    setEdits((current) => ({
      ...current,
      [index]: { ...(current[index] ?? { index }), [campo]: value },
    }));
  }

  function lineSinger(line: KaraokeLyricLine): string | null {
    return edits[line.index]?.singer ?? line.singer;
  }

  function cycleSinger(line: KaraokeLyricLine): void {
    const ids = job.singers.map((cantante) => cantante.id);
    const next = ids[(ids.indexOf(lineSinger(line) ?? "") + 1) % ids.length];
    edit(line.index, "singer", next);
  }

  function singerLabel(singerId: string | null): string {
    if (singerId === null) {
      return "—";
    }
    return (
      names[singerId] ??
      job.singers.find((cantante) => cantante.id === singerId)?.label ??
      singerId
    );
  }

  const hasChanges = Object.keys(edits).length > 0 || Object.keys(names).length > 0;

  return (
    <div className="flex flex-col gap-2">
      <span className={labelClassName}>{t("karaoke.review.lyrics")}</span>
      {hasSingers && (
        <div className="flex flex-wrap gap-2">
          {job.singers.map((cantante) => (
            <input
              key={cantante.id}
              value={names[cantante.id] ?? cantante.label}
              aria-label={t("karaoke.singers.renameLabel", { label: cantante.label })}
              onChange={(event) =>
                setNames((current) => ({
                  ...current,
                  [cantante.id]: event.target.value,
                }))
              }
              className={`${inputClassName} w-36`}
            />
          ))}
        </div>
      )}
      <div className="flex max-h-80 flex-col gap-1 overflow-y-auto pr-1">
        {job.lines.map((line) => (
          <div key={line.index} className="flex items-center gap-2">
            <span className="w-14 shrink-0 text-right font-mono-tabular text-xs text-text-faint">
              {line.start.toFixed(1)}s
            </span>
            {hasSingers && (
              <button
                type="button"
                aria-label={t("karaoke.singers.badgeLabel", { index: line.index + 1 })}
                onClick={() => cycleSinger(line)}
                style={{ backgroundColor: singerColors[lineSinger(line) ?? ""] }}
                className="w-20 shrink-0 truncate rounded-full px-2 py-0.5 text-xs font-semibold text-bg"
              >
                {singerLabel(lineSinger(line))}
              </button>
            )}
            <input
              value={edits[line.index]?.text ?? line.text}
              aria-label={t("karaoke.review.lineLabel", { index: line.index + 1 })}
              onChange={(event) => edit(line.index, "text", event.target.value)}
              className={`${inputClassName} flex-1`}
            />
            {hasTranslation && (
              <input
                value={edits[line.index]?.translation ?? line.translation}
                aria-label={t("karaoke.review.translationLabel", {
                  index: line.index + 1,
                })}
                onChange={(event) => edit(line.index, "translation", event.target.value)}
                className={`${inputClassName} flex-1 text-text-dim`}
              />
            )}
          </div>
        ))}
      </div>
      {hasChanges && (
        <button
          type="button"
          onClick={() => save.mutate()}
          disabled={save.isPending}
          className="self-start rounded border border-border bg-surface px-3 py-1.5 text-sm text-text hover:border-accent disabled:opacity-50"
        >
          {t("karaoke.review.save")}
        </button>
      )}
      {save.isError && (
        <p role="alert" className="text-xs text-danger">
          {(save.error as Error).message}
        </p>
      )}
    </div>
  );
}

function ReviewStage({
  job,
  onRefetch,
}: {
  job: KaraokeJob;
  onRefetch: () => void;
}) {
  const { t } = useTranslation();
  const [backgroundKind, setBackgroundKind] = useState<KaraokeBackgroundKind>(
    job.sourceHasPicture ? "source" : "generated",
  );
  const [backgroundFile, setBackgroundFile] = useState<File | null>(null);
  const [size, setSize] = useState("medium");
  const [position, setPosition] = useState("bottom");
  const [baseColor, setBaseColor] = useState("#FFFF00");
  const [highlightColor, setHighlightColor] = useState("#FFFFFF");
  const [singerColors, setSingerColors] = useState<Record<string, string>>(() =>
    Object.fromEntries(
      job.singers.map((cantante, i) => [
        cantante.id,
        DEFAULT_SINGER_COLORS[i % DEFAULT_SINGER_COLORS.length],
      ]),
    ),
  );
  const [muteSinger, setMuteSinger] = useState("");
  const hasSingers = job.singers.length > 0;

  const render = useMutation({
    mutationFn: () =>
      renderKaraokeJob(job.id, {
        backgroundKind,
        background: backgroundKind === "image" || backgroundKind === "video" ? backgroundFile : null,
        subtitleSize: size,
        subtitlePosition: position,
        subtitleColor: baseColor,
        subtitleHighlightColor: highlightColor,
        singerColors: hasSingers ? singerColors : undefined,
        muteSinger: muteSinger || null,
      }),
    onSuccess: onRefetch,
  });

  const needsFile = backgroundKind === "image" || backgroundKind === "video";
  const canRender = !render.isPending && (!needsFile || backgroundFile !== null);

  const kinds: KaraokeBackgroundKind[] = ["source", "image", "video", "generated"];

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-col gap-2">
        <span className={labelClassName}>{t("karaoke.review.listen")}</span>
        {job.instrumentalUrl && (
          <audio controls src={job.instrumentalUrl} className="w-full" />
        )}
      </div>

      <LyricsEditor job={job} singerColors={singerColors} onSaved={onRefetch} />

      {hasSingers && (
        <fieldset className="flex flex-col gap-2">
          <legend className={labelClassName}>{t("karaoke.singers.legend")}</legend>
          <div className="flex flex-wrap gap-4">
            {job.singers.map((cantante) => (
              <label
                key={cantante.id}
                className="flex items-center gap-2 text-sm text-text"
              >
                <input
                  type="color"
                  value={singerColors[cantante.id] ?? "#FFFFFF"}
                  aria-label={t("karaoke.singers.colorLabel", { name: cantante.label })}
                  onChange={(event) =>
                    setSingerColors((current) => ({
                      ...current,
                      [cantante.id]: event.target.value,
                    }))
                  }
                />
                {cantante.label}
              </label>
            ))}
          </div>
          <label className="flex flex-col gap-2">
            <span className={labelClassName}>{t("karaoke.singers.practiceAs")}</span>
            <select
              value={muteSinger}
              aria-label={t("karaoke.singers.practiceAs")}
              onChange={(event) => setMuteSinger(event.target.value)}
              className={inputClassName}
            >
              <option value="">{t("karaoke.singers.practiceNone")}</option>
              {job.singers.map((cantante) => (
                <option key={cantante.id} value={cantante.id}>
                  {cantante.label}
                </option>
              ))}
            </select>
            <span className="text-xs text-text-faint">
              {t("karaoke.singers.practiceHint")}
            </span>
          </label>
          <p className="text-xs text-warn">{t("karaoke.singers.unisonWarning")}</p>
        </fieldset>
      )}

      <fieldset className="flex flex-col gap-2">
        <legend className={labelClassName}>{t("karaoke.background.label")}</legend>
        <div className="flex flex-wrap gap-2">
          {kinds.map((kind) => (
            <button
              key={kind}
              type="button"
              aria-pressed={backgroundKind === kind}
              disabled={kind === "source" && !job.sourceHasPicture}
              onClick={() => setBackgroundKind(kind)}
              className={`rounded-sm border px-3 py-1.5 text-sm disabled:opacity-40 ${
                backgroundKind === kind
                  ? "border-accent bg-accent text-bg"
                  : "border-border bg-surface text-text-dim hover:border-text-faint"
              }`}
            >
              {t(`karaoke.background.${kind}`)}
            </button>
          ))}
        </div>
        {backgroundKind === "source" && !job.sourceHasPicture && (
          <p className="text-xs text-warn">{t("karaoke.background.sourceUnavailable")}</p>
        )}
        {needsFile && (
          <input
            type="file"
            accept={backgroundKind === "image" ? "image/*" : "video/*"}
            aria-label={t("karaoke.background.filePick")}
            onChange={(event) => setBackgroundFile(event.target.files?.[0] ?? null)}
            className="text-sm text-text-dim"
          />
        )}
      </fieldset>

      <div className="grid grid-cols-2 gap-3 max-[600px]:grid-cols-1">
        <label className="flex flex-col gap-2">
          <span className={labelClassName}>{t("karaoke.style.size")}</span>
          <select value={size} onChange={(e) => setSize(e.target.value)} className={inputClassName}>
            <option value="small">{t("karaoke.style.size.small")}</option>
            <option value="medium">{t("karaoke.style.size.medium")}</option>
            <option value="large">{t("karaoke.style.size.large")}</option>
          </select>
        </label>
        <label className="flex flex-col gap-2">
          <span className={labelClassName}>{t("karaoke.style.position")}</span>
          <select
            value={position}
            onChange={(e) => setPosition(e.target.value)}
            className={inputClassName}
          >
            <option value="bottom">{t("karaoke.style.position.bottom")}</option>
            <option value="top">{t("karaoke.style.position.top")}</option>
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm text-text">
          <input
            type="color"
            value={baseColor}
            aria-label={t("karaoke.style.base")}
            onChange={(e) => setBaseColor(e.target.value)}
          />
          {t("karaoke.style.base")}
        </label>
        <label className="flex items-center gap-2 text-sm text-text">
          <input
            type="color"
            value={highlightColor}
            aria-label={t("karaoke.style.highlight")}
            onChange={(e) => setHighlightColor(e.target.value)}
          />
          {t("karaoke.style.highlight")}
        </label>
      </div>

      <button
        type="button"
        onClick={() => render.mutate()}
        disabled={!canRender}
        className="self-start rounded bg-accent px-4 py-2 text-sm font-semibold text-bg disabled:opacity-50"
      >
        {t("karaoke.render")}
      </button>
      {render.isError && (
        <p role="alert" className="text-sm text-danger">
          {(render.error as Error).message}
        </p>
      )}
    </div>
  );
}

export function KaraokeStudioPanel() {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [file, setFile] = useState<File | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [separationModel, setSeparationModel] = useState("");
  const [restoreMode, setRestoreMode] = useState("");
  const [asrModelId, setAsrModelId] = useState("");
  const [language, setLanguage] = useState("");
  const [romanize, setRomanize] = useState(false);
  const [translateTo, setTranslateTo] = useState("");
  const [deviceId, setDeviceId] = useState("cpu");
  const [detectSingers, setDetectSingers] = useState(false);
  const [singerCount, setSingerCount] = useState(2);

  const capacidades = useAudioCapabilities();
  const modelsQuery = useInstalledAsrModels();
  const devicesQuery = useTranscribeDevices();
  const translationPairsQuery = useQuery({
    queryKey: ["translation-pairs"],
    queryFn: fetchTranslationPairs,
  });

  const separationModels = (capacidades.data?.separationModels ?? []).filter(
    (model) => model.category === "karaoke" && model.installed,
  );
  const cleanup = useCleanupSelection(
    capacidades.data?.cleanupSteps,
    capacidades.data?.cleanupOverprocessingThreshold ?? 3,
  );
  const restoreModes = capacidades.data?.restoreModes ?? [];
  const asrModels = modelsQuery.data ?? [];
  const selectedAsrId = asrModels.some((m) => m.id === asrModelId)
    ? asrModelId
    : (asrModels[0]?.id ?? "");
  const soloIngles = isEnglishOnly(
    asrModels.find((m) => m.id === selectedAsrId)?.name ?? selectedAsrId,
  );
  const devices = devicesQuery.data?.devices ?? [];
  const selectedDeviceId = devices.some((d) => d.id === deviceId)
    ? deviceId
    : (devicesQuery.data?.defaultDeviceId ?? devices[0]?.id ?? "");
  // Solo los pares que salen del idioma elegido: traducir desde un idioma
  // distinto al del audio no tiene sentido.
  const translationTargets = (translationPairsQuery.data?.pairs ?? [])
    .filter((pair) => pair.source === language)
    .map((pair) => pair.target);

  const jobQuery = useQuery({
    queryKey: ["karaoke-job", jobId],
    queryFn: () => getKaraokeJob(jobId as string),
    enabled: jobId !== null,
    refetchInterval: (query) => {
      const phase = query.state.data?.phase;
      const status = query.state.data?.status;
      if (phase === "preparing" || phase === "rendering") {
        // Un render fallido queda con status failed y phase review: sin cortar
        // ahi, el poll seguiria para siempre.
        return status === "failed" ? false : POLL_INTERVAL_MS;
      }
      return false;
    },
  });
  const job = jobQuery.data ?? null;

  const create = useMutation({
    mutationFn: () =>
      createKaraokeJob({
        file: file as File,
        asrModelId: selectedAsrId,
        separationModelId: separationModel || null,
        cleanupSteps: cleanup.enabledIds,
        restoreMode: restoreMode || null,
        language: soloIngles ? "en" : language || undefined,
        romanize: language === "ja" && !soloIngles && romanize,
        translateTo: translateTo || null,
        device: selectedDeviceId || undefined,
        detectSingers,
        singerCount: detectSingers ? singerCount : undefined,
      }),
    onSuccess: (response) => setJobId(response.jobId),
  });

  const cancel = useMutation({
    mutationFn: () => cancelKaraokeJob(jobId as string),
    onSuccess: () => void jobQuery.refetch(),
  });

  function reset(): void {
    setJobId(null);
    setFile(null);
    create.reset();
    void queryClient.removeQueries({ queryKey: ["karaoke-job"] });
  }

  const busy = job !== null && (job.phase === "preparing" || job.phase === "rendering");
  const canSubmit = file !== null && selectedAsrId.length > 0 && !create.isPending && jobId === null;

  return (
    <div className="flex flex-col gap-6">
      {jobId === null && (
        <div className="flex flex-col gap-5">
          <RecommendedAsrCard />
          <FileDropzone file={file} onFileSelected={setFile} />

          <label className="flex flex-col gap-2">
            <span className={labelClassName}>{t("karaoke.config.separation")}</span>
            <select
              value={separationModel}
              onChange={(e) => setSeparationModel(e.target.value)}
              className={inputClassName}
            >
              <option value="">{t("karaoke.config.separationDefault")}</option>
              {separationModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
          </label>

          <fieldset className="flex flex-col gap-2">
            <legend className={labelClassName}>{t("karaoke.singers.legend")}</legend>
            <label className="flex items-center gap-2 text-sm text-text">
              <input
                type="checkbox"
                checked={detectSingers}
                onChange={(e) => setDetectSingers(e.target.checked)}
                className="accent-accent"
              />
              {t("karaoke.singers.toggle")}
            </label>
            <p className="text-xs text-text-faint">{t("karaoke.singers.hint")}</p>
            {detectSingers && (
              <label className="ml-6 flex items-center gap-2 text-sm text-text">
                {t("karaoke.singers.count")}
                <select
                  value={String(singerCount)}
                  aria-label={t("karaoke.singers.count")}
                  onChange={(e) => setSingerCount(Number(e.target.value))}
                  className={inputClassName}
                >
                  {SINGER_COUNT_OPTIONS.map((n) => (
                    <option key={n} value={String(n)}>
                      {n}
                    </option>
                  ))}
                </select>
              </label>
            )}
          </fieldset>

          <fieldset className="flex flex-col gap-2">
            <legend className={labelClassName}>{t("karaoke.config.cleanup")}</legend>
            <p className="text-xs text-text-faint">{t("karaoke.config.cleanupHint")}</p>
            <label className="flex items-center gap-2 text-sm text-text">
              <input
                type="checkbox"
                checked={cleanup.active}
                onChange={(e) => cleanup.setActive(e.target.checked)}
                className="accent-accent"
              />
              {t("karaoke.config.cleanupToggle")}
            </label>
            {cleanup.active &&
              (capacidades.data?.cleanupSteps ?? [])
                .filter((step) => step.installed)
                .map((step) => (
                  <label key={step.id} className="ml-6 flex items-center gap-2 text-sm text-text">
                    <input
                      type="checkbox"
                      checked={cleanup.isEnabled(step.id)}
                      onChange={(e) => cleanup.toggleStep(step.id, e.target.checked)}
                      className="accent-accent"
                    />
                    {step.name}
                  </label>
                ))}
            {cleanup.isOverprocessing && (
              <p className="text-xs text-warn">{t("karaoke.config.overprocessing")}</p>
            )}
          </fieldset>

          {restoreModes.length > 0 && (
            <fieldset className="flex flex-col gap-2">
              <legend className={labelClassName}>{t("karaoke.config.restore")}</legend>
              <label className="flex items-center gap-2 text-sm text-text">
                <input
                  type="radio"
                  name="karaoke-restore"
                  checked={restoreMode === ""}
                  onChange={() => setRestoreMode("")}
                  className="accent-accent"
                />
                {t("karaoke.config.restoreNone")}
              </label>
              {restoreModes.map((mode) => (
                <label key={mode} className="flex items-center gap-2 text-sm text-text">
                  <input
                    type="radio"
                    name="karaoke-restore"
                    checked={restoreMode === mode}
                    onChange={() => setRestoreMode(mode)}
                    className="accent-accent"
                  />
                  {t(`karaoke.config.restore.${mode}`)}
                </label>
              ))}
            </fieldset>
          )}

          <label className="flex flex-col gap-2">
            <span className={labelClassName}>{t("karaoke.config.asrModel")}</span>
            <select
              value={selectedAsrId}
              onChange={(e) => setAsrModelId(e.target.value)}
              className={inputClassName}
            >
              {asrModels.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
            </select>
            {soloIngles && (
              <span className="text-xs text-warn">{t("transcribe.language.englishOnly")}</span>
            )}
          </label>

          <label className="flex flex-col gap-2">
            <span className={labelClassName}>{t("transcribe.language.label")}</span>
            <select
              value={soloIngles ? "en" : language}
              disabled={soloIngles}
              onChange={(e) => {
                setLanguage(e.target.value);
                setTranslateTo("");
              }}
              className={inputClassName}
            >
              <option value="">{t("transcribe.language.auto")}</option>
              {LANGUAGE_OPTIONS.map((code) => (
                <option key={code} value={code}>
                  {t(`transcribe.language.${code}`)}
                </option>
              ))}
            </select>
          </label>

          {language === "ja" && !soloIngles && (
            <label className="flex items-center gap-2 text-sm text-text">
              <input
                type="checkbox"
                checked={romanize}
                onChange={(e) => setRomanize(e.target.checked)}
                className="accent-accent"
              />
              {t("transcribe.romanize.label")}
            </label>
          )}

          {language && !soloIngles && translationTargets.length > 0 && (
            <label className="flex flex-col gap-2">
              <span className={labelClassName}>{t("karaoke.config.translate")}</span>
              <select
                value={translateTo}
                onChange={(e) => setTranslateTo(e.target.value)}
                className={inputClassName}
              >
                <option value="">{t("karaoke.config.translateNone")}</option>
                {translationTargets.map((code) => (
                  <option key={code} value={code}>
                    {code}
                  </option>
                ))}
              </select>
              <span className="text-xs text-text-faint">{t("karaoke.config.translateHint")}</span>
            </label>
          )}

          <label className="flex flex-col gap-2">
            <span className={labelClassName}>{t("transcribe.device.label")}</span>
            <select
              value={selectedDeviceId}
              onChange={(e) => setDeviceId(e.target.value)}
              className={inputClassName}
            >
              {devices.map((device) => (
                <option key={device.id} value={device.id}>
                  {device.name}
                </option>
              ))}
            </select>
          </label>

          <button
            type="button"
            onClick={() => create.mutate()}
            disabled={!canSubmit}
            className="self-start rounded bg-accent px-4 py-2 text-sm font-semibold text-bg disabled:opacity-50"
          >
            {t("karaoke.submit")}
          </button>
          {create.isError && (
            <p role="alert" className="text-sm text-danger">
              {(create.error as Error).message}
            </p>
          )}
        </div>
      )}

      {job !== null && (
        <div className="flex flex-col gap-4">
          {busy && (
            <div role="status" className="flex flex-col gap-2">
              <p className="text-sm text-text">
                {job.phase === "preparing" ? t("karaoke.preparing") : t("karaoke.rendering")}
                {job.progressPct !== null && (
                  <span className="ml-2 font-mono-tabular text-text-dim">
                    {job.progressPct.toFixed(0)}%
                  </span>
                )}
              </p>
              <button
                type="button"
                onClick={() => cancel.mutate()}
                className="self-start rounded border border-border px-3 py-1.5 text-sm text-text-dim hover:border-danger hover:text-danger"
              >
                {t("karaoke.cancel")}
              </button>
            </div>
          )}

          {job.error && (
            <p role="alert" className="rounded border border-danger bg-surface px-3 py-2 text-sm text-danger">
              {job.error}
            </p>
          )}

          {job.phase === "review" && (
            <ReviewStage job={job} onRefetch={() => void jobQuery.refetch()} />
          )}

          {job.phase === "completed" && job.downloadUrl && (
            <div className="flex flex-col gap-3">
              <video controls src={job.downloadUrl} className="max-h-96 w-full rounded" />
              <div className="flex gap-3">
                <a
                  href={job.downloadUrl}
                  download
                  className="rounded bg-accent px-4 py-2 text-sm font-semibold text-bg"
                >
                  {t("karaoke.done.download")}
                </a>
                {job.practiceMixUrl && (
                  <a
                    href={job.practiceMixUrl}
                    download
                    className="rounded border border-border px-4 py-2 text-sm text-text-dim hover:border-accent"
                  >
                    {t("karaoke.singers.practiceDownload")}
                  </a>
                )}
                <button
                  type="button"
                  onClick={reset}
                  className="rounded border border-border px-4 py-2 text-sm text-text-dim hover:border-accent"
                >
                  {t("karaoke.reset")}
                </button>
              </div>
            </div>
          )}

          {(job.phase === "failed" || job.phase === "cancelled") && (
            <button
              type="button"
              onClick={reset}
              className="self-start rounded border border-border px-4 py-2 text-sm text-text-dim hover:border-accent"
            >
              {t("karaoke.reset")}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
