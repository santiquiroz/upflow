import { useQuery } from "@tanstack/react-query";
import { AudioWaveform, UploadCloud } from "lucide-react";
import { useState, type ChangeEvent, type DragEvent } from "react";
import { AccordionSection } from "../../components/AccordionSection";
import { DevicePicker } from "../../components/DevicePicker";
import { getDevices } from "../../lib/api";
import { FormatOptionFieldset, type FormatOption } from "../../components/FormatOptionFieldset";
import { JobCard } from "../../components/JobCard";
import { useAudioCapabilities, useAudioJob, type AudioJobPhase } from "../../hooks/useAudioJob";
import { useVoiceCatalog } from "../../hooks/useVoiceCatalog";
import { useTranslation } from "../../i18n/LocaleProvider";
import { denoiseLabel, restoreLabel } from "../../lib/audioLabels";
import type { DeviceInfoResponse, LossyQuality, MasteringPreset } from "../../lib/apiTypes";
import { formatDeviceSummary } from "../enhance/accordionSummaries";
import { CleanupChainPanel, cleanupSummaryKey } from "./CleanupChainPanel";
import { KaraokeSection } from "./KaraokeSection";
import { EnsembleSection } from "./EnsembleSection";
import { ModelComparison } from "./ModelComparison";
import { RehearsalSection } from "./RehearsalSection";
import { VoiceChainPanel, voiceSummaryKey } from "./VoiceChainPanel";
import {
  convertibleFileCount,
  isLossyFormat,
  joinAsChoices,
  losesQualityIrreversibly,
  selectableSectionKeys,
} from "./selectionHint";
import { useCleanupSelection } from "./useCleanupSelection";
import { useRehearsalSelection } from "./useRehearsalSelection";
import { useVoiceSelection } from "./useVoiceSelection";

type AudioOutputFormat = "flac" | "wav" | "mp3" | "m4a";

const OUTPUT_FORMAT_KEYS: readonly { value: AudioOutputFormat; labelKey: string; descriptionKey: string }[] = [
  { value: "flac", labelKey: "audio.format.flac", descriptionKey: "audio.format.flac.description" },
  { value: "wav", labelKey: "audio.format.wav", descriptionKey: "audio.format.wav.description" },
  { value: "m4a", labelKey: "audio.format.m4a", descriptionKey: "audio.format.m4a.description" },
  { value: "mp3", labelKey: "audio.format.mp3", descriptionKey: "audio.format.mp3.description" },
];

function outputFormatOptions(
  t: (key: string) => string,
): readonly FormatOption<AudioOutputFormat>[] {
  return OUTPUT_FORMAT_KEYS.map((option) => ({
    value: option.value,
    label: t(option.labelKey),
    description: t(option.descriptionKey),
  }));
}

// El nombre del escalon lo pone el frontend y el bitrate el backend: una sola
// fuente para los numeros, y la copia sigue siendo traducible.
function lossyQualityOptions(
  qualities: readonly LossyQuality[],
  outputFormat: string,
  t: (key: string, vars?: Record<string, string | number>) => string,
): readonly FormatOption<string>[] {
  return qualities.map((quality) => ({
    value: quality.id,
    label: t(`audio.quality.${quality.id}`),
    description: t("audio.quality.bitrate", {
      bitrate: quality.bitrates[outputFormat] ?? "",
    }),
  }));
}

// La clave del tooltip y no la frase: este texto dice para QUE material sirve la
// sección, que es justo lo que faltaba (ambos motores están entrenados con habla
// y en música apagan instrumentos), y esa aclaración tiene que existir en los
// dos idiomas.
const DENOISE_TOOLTIP = "audio.denoise.tooltip";
const CLEANUP_TOOLTIP = "audio.cleanup.tooltip";
const RESTORE_TOOLTIP =
  "Reconstruct high frequencies lost to lossy compression (MP3/AAC). Apollo is fast band-restore; AudioSR is diffusion super-resolution — much higher quality ceiling but far slower (minutes per minute of audio on GPU). Experimental — quality varies by source.";
const DEVICE_TOOLTIP = "Pick the compute device that runs the restoration model (CPU or a DirectML GPU).";
const VOICE_TOOLTIP = "audio.voice.tooltip";

interface ModeOption {
  value: string | null;
  label: string;
  experimental?: boolean;
}

function isJobBusy(phase: AudioJobPhase): boolean {
  return phase === "uploading" || phase === "queued" || phase === "running";
}

function segmentButtonClassName(isActive: boolean): string {
  const base =
    "inline-flex items-center gap-2 rounded-sm border px-3 py-1.5 text-sm transition-[background-color,border-color,color] duration-fast focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";
  if (isActive) {
    return `${base} border-accent bg-accent text-bg`;
  }
  return `${base} border-border bg-surface text-text-dim hover:border-text-faint hover:text-text`;
}

function ExperimentalBadge() {
  return (
    <span className="rounded-sm bg-surface-2 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-warn">
      Experimental
    </span>
  );
}

function ModeSegmentedControl({
  legend,
  options,
  value,
  onChange,
}: {
  legend: string;
  options: readonly ModeOption[];
  value: string | null;
  onChange: (value: string | null) => void;
}) {
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">{legend}</legend>
      <div className="flex flex-wrap gap-2">
        {options.map((option) => (
          <button
            key={option.label}
            type="button"
            aria-pressed={value === option.value}
            className={segmentButtonClassName(value === option.value)}
            onClick={() => onChange(option.value)}
          >
            {option.label}
            {option.experimental && <ExperimentalBadge />}
          </button>
        ))}
      </div>
    </fieldset>
  );
}

function Dropzone({
  files,
  onFilesSelected,
}: {
  files: File[];
  onFilesSelected: (files: File[]) => void;
}) {
  const { t } = useTranslation();
  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    const dropped = Array.from(event.dataTransfer.files);
    if (dropped.length > 0) {
      onFilesSelected(dropped);
    }
  }

  function handleInputChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files ?? []);
    if (selected.length > 0) {
      onFilesSelected(selected);
    }
  }

  const label =
    files.length === 0
      ? t("audio.dropzone")
      : files.length === 1
        ? files[0].name
        : t("enhance.batch.selected", { count: files.length });

  return (
    <label
      htmlFor="audio-file-input"
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
      className="flex cursor-pointer flex-col items-center gap-2 rounded border border-dashed border-border bg-surface px-6 py-10 text-center transition-[border-color] duration-fast hover:border-accent"
    >
      <UploadCloud aria-hidden="true" className="h-6 w-6 text-text-faint" strokeWidth={1.5} />
      <span className="text-sm text-text">{label}</span>
      <span className="text-xs text-text-faint">WAV, MP3, FLAC, M4A, OGG, OPUS</span>
      <input
        id="audio-file-input"
        type="file"
        accept="audio/*"
        multiple
        className="sr-only"
        onChange={handleInputChange}
      />
    </label>
  );
}

function buildDenoiseOptions(denoiseModes: string[], noneLabel: string): ModeOption[] {
  return [
    { value: null, label: noneLabel },
    ...denoiseModes.map((mode) => ({ value: mode, label: denoiseLabel(mode) })),
  ];
}

function buildRestoreOptions(restoreModes: string[], noneLabel: string): ModeOption[] {
  return [
    { value: null, label: noneLabel },
    ...restoreModes.map((mode) => ({ value: mode, label: restoreLabel(mode), experimental: true })),
  ];
}

function useMasteringCopy() {
  const { t } = useTranslation();
  return {
    masteringLabel: (preset: MasteringPreset | undefined) =>
      preset ? t(preset.labelKey) : null,
    masteringDescription: (preset: MasteringPreset | undefined) =>
      preset ? t(preset.descriptionKey) : null,
  };
}

export function AudioPanel() {
  const [files, setFiles] = useState<File[]>([]);
  const [denoise, setDenoise] = useState<string | null>(null);
  const [restore, setRestore] = useState<string | null>(null);
  const [outputFormat, setOutputFormat] = useState<AudioOutputFormat>("flac");
  const [lossyQuality, setLossyQuality] = useState<string>("maximum");
  // null = el dispositivo por defecto del backend (la GPU si hay una), igual
  // que los paneles de imagen y video. Antes arrancaba fijo en CPU y mandaba
  // device="cpu" pisando ese default: un restore quedaba ~10x mas lento sin
  // que nada lo dijera, con el badge "Default" colgando de la GPU.
  const [device, setDevice] = useState<DeviceInfoResponse | null>(null);
  // Misma clave que DevicePicker: sale del cache de react-query, sin pedido extra.
  const devicesQuery = useQuery({ queryKey: ["devices"], queryFn: getDevices });
  const backendDefault =
    devicesQuery.data?.devices.find((d) => d.id === devicesQuery.data?.defaultDeviceId) ?? null;
  const effectiveDevice = device ?? backendDefault;
  const hasGpu = (devicesQuery.data?.devices ?? []).some((d) => d.kind === "gpu");
  const runsOnCpuWithGpuAvailable = hasGpu && effectiveDevice?.kind === "cpu";
  const [master, setMaster] = useState<string | null>(null);
  const [separate, setSeparate] = useState(false);
  const [separationModel, setSeparationModel] = useState<string | null>(null);
  const [ensembleModels, setEnsembleModels] = useState<string[]>([]);

  const capabilitiesQuery = useAudioCapabilities();
  const voiceCatalogQuery = useVoiceCatalog();
  const voice = useVoiceSelection(voiceCatalogQuery.data);
  const cleanupCatalog = capabilitiesQuery.data?.cleanupSteps;
  const cleanup = useCleanupSelection(
    cleanupCatalog,
    capabilitiesQuery.data?.cleanupOverprocessingThreshold,
  );
  const {
    phase,
    job,
    errorMessage,
    submitMany,
    cancel,
    reset,
    uploadPercent,
    pendingUploads,
    failedUploads,
  } = useAudioJob();
  const { t, locale } = useTranslation();
  const { masteringLabel, masteringDescription } = useMasteringCopy();

  const denoiseModes = capabilitiesQuery.data?.denoiseModes ?? [];
  const restoreModes = capabilitiesQuery.data?.restoreModes ?? [];
  const restoreAvailable = restoreModes.length > 0;
  // Siempre disponible: lo hace ffmpeg, que ya viene con la app.
  const masteringPresets = capabilitiesQuery.data?.masteringPresets ?? [];
  const separationModels = capabilitiesQuery.data?.separationModels ?? [];
  const lossyQualities = capabilitiesQuery.data?.lossyQualities ?? [];
  // Default: el elegido, o el primer modelo instalado del catálogo.
  const effectiveSeparationModel =
    separationModel ?? separationModels.find((model) => model.installed)?.id ?? null;
  const selectedSeparationSpec = separationModels.find(
    (model) => model.id === effectiveSeparationModel,
  );
  const rehearsal = useRehearsalSelection(selectedSeparationSpec);
  // El resumen nombra los stems del modelo elegido: para karaoke dice
  // voz/instrumental y para la limpieza "sin reverb + reverb".
  const separationSummary = selectedSeparationSpec
    ? selectedSeparationSpec.stems.map((stem) => t(stem.labelKey)).join(" + ")
    : t("audio.karaoke.summary.on");

  function handleFilesSelected(selected: File[]) {
    setFiles(selected);
    reset();
  }

  function handleSubmit() {
    if (!canSubmit || files.length === 0) {
      return;
    }
    // Con karaoke activo se manda SOLO la separacion: el backend rechaza
    // combinarla, y las secciones deshabilitadas no deben viajar en el form.
    const comunes = separate
      ? {
          denoise: null,
          restore: null,
          outputFormat,
          lossyQuality,
          device: device?.id ?? null,
          voiceSteps: [],
          voiceDelivery: null,
          voicePresenceDb: null,
          master: null,
          cleanupSteps: [],
          separate: true,
          separationModel: effectiveSeparationModel,
          ensembleModels,
          practiceStems: rehearsal.enabledStems,
          practiceGuidePercent: rehearsal.guidePercent,
        }
      : {
          denoise,
          restore,
          outputFormat,
          lossyQuality,
          device: device?.id ?? null,
          voiceSteps: voice.enabledIds,
          voiceDelivery: voice.delivery,
          voicePresenceDb: voice.isEnabled("presence") ? voice.presenceDb : null,
          master,
          cleanupSteps: cleanup.enabledIds,
        };
    submitMany(files.map((file) => ({ file, ...comunes })));
  }

  // La mejora de voz cuenta como seleccion: alguien que solo quiere enfocar el
  // dialogo no tiene por que encender denoise ni restore para poder enviar.
  // Nivelar el volumen es una entrega valida por si sola: alguien puede querer
  // solo dejar el archivo al volumen del estandar, sin tocarle nada mas.
  const hasSelection =
    separate ||
    denoise !== null ||
    restore !== null ||
    master !== null ||
    cleanup.enabledIds.length > 0 ||
    voice.enabledIds.length > 0;
  // Cambiar de formato es una entrega valida POR SI SOLA: quien tiene un FLAC y
  // necesita un MP3 por compatibilidad no tiene ningun paso que pedir, y hasta
  // ahora el boton lo dejaba afuera pidiendole que eligiera una seccion.
  const convertibleFiles = convertibleFileCount(files, outputFormat);
  const isConversionOnly = !hasSelection && convertibleFiles > 0;
  // El unico caso que sigue sin tener nada que hacer: sin pasos y con todos los
  // archivos ya en el formato pedido. El backend lo rechaza igual; decirlo aca
  // evita el viaje.
  const hasNothingToDo = !hasSelection && files.length > 0 && convertibleFiles === 0;
  const separationReady = !separate || effectiveSeparationModel !== null;
  const canSubmit =
    files.length > 0 &&
    (hasSelection || isConversionOnly) &&
    separationReady &&
    (separate || !voice.needsDelivery) &&
    !isJobBusy(phase);

  return (
    <div className="grid grid-cols-[1fr_320px] gap-6 max-[900px]:grid-cols-1">
      <div className="flex flex-col gap-6">
        <Dropzone files={files} onFilesSelected={handleFilesSelected} />
        <AccordionSection
          title={t("audio.section.karaoke")}
          summary={separate ? separationSummary : t("audio.mode.none")}
          tooltip={t("audio.karaoke.tooltip")}
        >
          <KaraokeSection
            enabled={separate}
            onToggle={setSeparate}
            models={separationModels}
            selectedModel={effectiveSeparationModel}
            onSelectModel={setSeparationModel}
          />
          {separate && (
            <EnsembleSection
              models={separationModels}
              selectedModel={effectiveSeparationModel}
              chosen={ensembleModels}
              onChange={setEnsembleModels}
            />
          )}
          {separate && <ModelComparison file={files[0] ?? null} models={separationModels} />}
        </AccordionSection>
        {separate && (
          <p role="status" className="text-xs text-text-dim">
            {t("audio.karaoke.exclusiveNote")}
          </p>
        )}
        {separate && (
          <RehearsalSection model={selectedSeparationSpec} selection={rehearsal} />
        )}
        <div
          aria-disabled={separate}
          // inert saca el subarbol del tab order y del arbol de accesibilidad:
          // pointer-events-none solo bloquea el mouse, no Enter/Space ni Tab.
          // React 18 no tipa inert como prop; el spread esquiva el chequeo.
          {...(separate ? { inert: "" } : {})}
          className={
            separate
              ? "pointer-events-none flex flex-col gap-6 opacity-40"
              : "flex flex-col gap-6"
          }
        >
        {/* Igual que Acabado y Restauración: la sección existe cuando el
            backend reporta su catálogo, no antes. */}
        {cleanupCatalog !== undefined && cleanupCatalog.length > 0 && (
          <AccordionSection
            title={t("audio.section.cleanup")}
            summary={t(cleanupSummaryKey(cleanup.enabledIds.length), {
              count: cleanup.enabledIds.length,
            })}
            tooltip={t(CLEANUP_TOOLTIP)}
            defaultOpen
          >
            <CleanupChainPanel steps={cleanupCatalog} selection={cleanup} />
          </AccordionSection>
        )}
        <AccordionSection
          title={t("audio.section.denoise")}
          summary={denoiseLabel(denoise)}
          tooltip={t(DENOISE_TOOLTIP)}
          // Las DOS abiertas de entrada: la confusión que esto arregla es
          // elegir "reducción de ruido" para música, y solo se disuelve viendo
          // las dos secciones juntas con lo que dice cada una.
          defaultOpen
        >
          {/* La aclaración va visible y no solo en el tooltip: ambos motores
              están entrenados con habla, y en música apagan instrumentos. Sin
              decirlo, la sección se lee como "quitar ruido de cualquier cosa" y
              se elige para música, que es la confusión que esto arregla. */}
          <p className="mb-2 text-xs leading-relaxed text-text-dim">
            {t("audio.denoise.voiceOnlyHint")}
          </p>
          <ModeSegmentedControl
            legend={t("audio.section.denoise")}
            options={buildDenoiseOptions(denoiseModes, t("audio.mode.none"))}
            value={denoise}
            onChange={setDenoise}
          />
          {denoise !== null && (
            <p role="status" className="mt-2 text-xs text-text-dim">
              {t(`audio.denoise.mode.${denoise}.description`)}
            </p>
          )}
        </AccordionSection>
        {masteringPresets.length > 0 && (
          <AccordionSection
            title={t("audio.section.mastering")}
            summary={
              masteringLabel(masteringPresets.find((p) => p.id === master)) ?? t("audio.mastering.none")
            }
            tooltip={t("audio.mastering.tooltip")}
          >
            <ModeSegmentedControl
              legend={t("audio.section.mastering")}
              options={[
                { value: null, label: t("audio.mastering.none") },
                ...masteringPresets.map((preset) => ({
                  value: preset.id,
                  label: t(preset.labelKey),
                })),
              ]}
              value={master}
              onChange={setMaster}
            />
            {master !== null && (
              <p role="status" className="mt-2 text-xs text-text-dim">
                {masteringDescription(masteringPresets.find((p) => p.id === master))}{" "}
                {t("audio.mastering.target", {
                  lufs: masteringPresets.find((p) => p.id === master)?.targetLufs ?? "",
                })}
              </p>
            )}
          </AccordionSection>
        )}
        {restoreAvailable && (
          <AccordionSection
            title={t("audio.section.restore")}
            summary={restoreLabel(restore)}
            tooltip={RESTORE_TOOLTIP}
          >
            <ModeSegmentedControl
              legend={t("audio.section.restore")}
              options={buildRestoreOptions(restoreModes, t("audio.mode.none"))}
              value={restore}
              onChange={setRestore}
            />
            {restore === "audiosr" && (
              <p role="status" className="mt-2 text-xs text-warn">
                AudioSR runs a 258M-parameter diffusion model: expect roughly 2 minutes of processing
                per minute of audio on a GPU (much longer on CPU).
              </p>
            )}
          </AccordionSection>
        )}
        <AccordionSection
          title={t("voice.sectionTitle")}
          summary={t(voiceSummaryKey(voice.enabledIds.length), {
            count: voice.enabledIds.length,
          })}
          tooltip={t(VOICE_TOOLTIP)}
        >
          <VoiceChainPanel
            catalog={voiceCatalogQuery.data}
            isLoading={voiceCatalogQuery.isLoading}
            isError={voiceCatalogQuery.isError}
            selection={voice}
          />
        </AccordionSection>
        </div>
        <AccordionSection
          title={t("audio.section.device")}
          summary={formatDeviceSummary(effectiveDevice, t)}
          tooltip={DEVICE_TOOLTIP}
        >
          <DevicePicker value={device?.id ?? null} onChange={setDevice} requiresGpu={false} allowAuto={false} />
          {runsOnCpuWithGpuAvailable && (
            <p className="mt-2 text-xs text-warn">{t("audio.device.cpuSlowerHint")}</p>
          )}
        </AccordionSection>
        <FormatOptionFieldset
          legend={t("audio.section.outputFormat")}
          name="audio-output-format"
          options={outputFormatOptions(t)}
          value={outputFormat}
          onChange={setOutputFormat}
        />
        {/* Que se puede convertir SIN procesar nada tiene que estar escrito
            donde se elige el formato: es la unica pista de que el boton se
            enciende sin tocar ninguna seccion. */}
        <p className="text-xs text-text-dim">{t("audio.convert.standaloneHint")}</p>
        {lossyQualities.length > 0 && isLossyFormat(outputFormat) && (
          <FormatOptionFieldset
            legend={t("audio.section.lossyQuality")}
            name="audio-lossy-quality"
            options={lossyQualityOptions(lossyQualities, outputFormat, t)}
            value={lossyQuality}
            onChange={setLossyQuality}
          />
        )}
        {losesQualityIrreversibly(files, outputFormat) && (
          <p role="status" className="text-xs text-warn">
            {t("audio.convert.lossyWarning", { format: outputFormat.toUpperCase() })}
          </p>
        )}
        <div className="flex flex-col gap-2">
          {isConversionOnly && (
            <p role="status" className="text-xs text-text-dim">
              {t("audio.convert.onlyConversion", { format: outputFormat.toUpperCase() })}
            </p>
          )}
          {hasNothingToDo && (
            <p role="status" className="text-xs text-text-faint">
              {t("audio.hint.sameFormat", { format: outputFormat.toUpperCase() })}
            </p>
          )}
          {!hasSelection && !isConversionOnly && !hasNothingToDo && (
            <p role="status" className="text-xs text-text-faint">
              {t("audio.hint.pickOne", {
                options: joinAsChoices(
                  selectableSectionKeys({
                    masteringAvailable: masteringPresets.length > 0,
                    restoreAvailable,
                    cleanupAvailable: (cleanupCatalog ?? []).some((step) => step.installed),
                  }).map((key) => t(key)),
                  locale,
                ),
              })}
            </p>
          )}
          <button
            type="button"
            onClick={handleSubmit}
            disabled={!canSubmit}
            className="inline-flex w-fit items-center gap-2 rounded bg-accent px-4 py-2 text-sm font-medium text-bg transition-[background-color,opacity] duration-fast hover:bg-accent-hover active:bg-accent-press disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
          >
            <AudioWaveform aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
            {t(isConversionOnly ? "audio.cta.convert" : "audio.cta.enhance")}
          </button>
        </div>
      </div>
      <JobCard
        phase={phase}
        job={job}
        fileName={files[0]?.name}
        errorMessage={errorMessage}
        onCancel={cancel}
        uploadPercent={uploadPercent}
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
  );
}
