import { AudioWaveform, UploadCloud } from "lucide-react";
import { useState, type ChangeEvent, type DragEvent } from "react";
import { AccordionSection } from "../../components/AccordionSection";
import { CPU_DEVICE, DevicePicker } from "../../components/DevicePicker";
import { FormatOptionFieldset, type FormatOption } from "../../components/FormatOptionFieldset";
import { JobCard } from "../../components/JobCard";
import { useAudioCapabilities, useAudioJob, type AudioJobPhase } from "../../hooks/useAudioJob";
import { useVoiceCatalog } from "../../hooks/useVoiceCatalog";
import { useTranslation } from "../../i18n/LocaleProvider";
import { denoiseLabel, restoreLabel } from "../../lib/audioLabels";
import type { DeviceInfoResponse, MasteringPreset } from "../../lib/apiTypes";
import { formatDeviceSummary } from "../enhance/accordionSummaries";
import { VoiceChainPanel, voiceSummaryKey } from "./VoiceChainPanel";
import { joinAsChoices, selectableSectionKeys } from "./selectionHint";
import { useVoiceSelection } from "./useVoiceSelection";

type AudioOutputFormat = "flac" | "wav" | "mp3";

const OUTPUT_FORMAT_KEYS: readonly { value: AudioOutputFormat; labelKey: string; descriptionKey: string }[] = [
  { value: "flac", labelKey: "audio.format.flac", descriptionKey: "audio.format.flac.description" },
  { value: "wav", labelKey: "audio.format.wav", descriptionKey: "audio.format.wav.description" },
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

const DENOISE_TOOLTIP =
  "Remove background noise with an AI denoiser. DeepFilterNet is stronger; RNNoise is lighter. Runs before restoration.";
const RESTORE_TOOLTIP =
  "Reconstruct high frequencies lost to lossy compression (MP3/AAC). Apollo is fast band-restore; AudioSR is diffusion super-resolution — much higher quality ceiling but far slower (minutes per minute of audio on GPU). Experimental — quality varies by source.";
const DEVICE_TOOLTIP = "Pick the compute device that runs the restoration model (CPU or a DirectML GPU).";
const VOICE_TOOLTIP =
  "Shape the voice itself: remove noise, even out the volume, focus the dialogue, tame sibilance and match the loudness the destination platform expects. The steps run in a fixed order because each one works better on what the previous one left.";

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

function Dropzone({ file, onFileSelected }: { file: File | null; onFileSelected: (file: File) => void }) {
  const { t } = useTranslation();
  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    const dropped = event.dataTransfer.files[0];
    if (dropped) {
      onFileSelected(dropped);
    }
  }

  function handleInputChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0];
    if (selected) {
      onFileSelected(selected);
    }
  }

  return (
    <label
      htmlFor="audio-file-input"
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
      className="flex cursor-pointer flex-col items-center gap-2 rounded border border-dashed border-border bg-surface px-6 py-10 text-center transition-[border-color] duration-fast hover:border-accent"
    >
      <UploadCloud aria-hidden="true" className="h-6 w-6 text-text-faint" strokeWidth={1.5} />
      <span className="text-sm text-text">{file ? file.name : t("audio.dropzone")}</span>
      <span className="text-xs text-text-faint">WAV, MP3, FLAC, M4A, OGG, OPUS</span>
      <input id="audio-file-input" type="file" accept="audio/*" className="sr-only" onChange={handleInputChange} />
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
  const [file, setFile] = useState<File | null>(null);
  const [denoise, setDenoise] = useState<string | null>(null);
  const [restore, setRestore] = useState<string | null>(null);
  const [outputFormat, setOutputFormat] = useState<AudioOutputFormat>("flac");
  const [device, setDevice] = useState<DeviceInfoResponse | null>(CPU_DEVICE);
  const [master, setMaster] = useState<string | null>(null);

  const capabilitiesQuery = useAudioCapabilities();
  const voiceCatalogQuery = useVoiceCatalog();
  const voice = useVoiceSelection(voiceCatalogQuery.data);
  const { phase, job, errorMessage, submit, cancel, reset } = useAudioJob();
  const { t, locale } = useTranslation();
  const { masteringLabel, masteringDescription } = useMasteringCopy();

  const denoiseModes = capabilitiesQuery.data?.denoiseModes ?? [];
  const restoreModes = capabilitiesQuery.data?.restoreModes ?? [];
  const restoreAvailable = restoreModes.length > 0;
  // Siempre disponible: lo hace ffmpeg, que ya viene con la app.
  const masteringPresets = capabilitiesQuery.data?.masteringPresets ?? [];

  function handleFileSelected(selected: File) {
    setFile(selected);
    reset();
  }

  function handleSubmit() {
    if (!canSubmit || !file) {
      return;
    }
    submit({
      file,
      denoise,
      restore,
      outputFormat,
      device: device?.id ?? null,
      voiceSteps: voice.enabledIds,
      voiceDelivery: voice.delivery,
      voicePresenceDb: voice.isEnabled("presence") ? voice.presenceDb : null,
      master,
    });
  }

  // La mejora de voz cuenta como seleccion: alguien que solo quiere enfocar el
  // dialogo no tiene por que encender denoise ni restore para poder enviar.
  // Nivelar el volumen es una entrega valida por si sola: alguien puede querer
  // solo dejar el archivo al volumen del estandar, sin tocarle nada mas.
  const hasSelection =
    denoise !== null || restore !== null || master !== null || voice.enabledIds.length > 0;
  const canSubmit =
    file !== null && hasSelection && !voice.needsDelivery && !isJobBusy(phase);

  return (
    <div className="grid grid-cols-[1fr_320px] gap-6 max-[900px]:grid-cols-1">
      <div className="flex flex-col gap-6">
        <Dropzone file={file} onFileSelected={handleFileSelected} />
        <AccordionSection
          title={t("audio.section.denoise")}
          summary={denoiseLabel(denoise)}
          tooltip={DENOISE_TOOLTIP}
          defaultOpen
        >
          <ModeSegmentedControl
            legend={t("audio.section.denoise")}
            options={buildDenoiseOptions(denoiseModes, t("audio.mode.none"))}
            value={denoise}
            onChange={setDenoise}
          />
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
          tooltip={VOICE_TOOLTIP}
        >
          <VoiceChainPanel
            catalog={voiceCatalogQuery.data}
            isLoading={voiceCatalogQuery.isLoading}
            isError={voiceCatalogQuery.isError}
            selection={voice}
          />
        </AccordionSection>
        <AccordionSection
          title={t("audio.section.device")}
          summary={formatDeviceSummary(device)}
          tooltip={DEVICE_TOOLTIP}
        >
          <DevicePicker value={device?.id ?? null} onChange={setDevice} requiresGpu={false} allowAuto={false} />
        </AccordionSection>
        <FormatOptionFieldset
          legend={t("audio.section.outputFormat")}
          name="audio-output-format"
          options={outputFormatOptions(t)}
          value={outputFormat}
          onChange={setOutputFormat}
        />
        <div className="flex flex-col gap-2">
          {!hasSelection && (
            <p role="status" className="text-xs text-text-faint">
              {t("audio.hint.pickOne", {
                options: joinAsChoices(
                  selectableSectionKeys({
                    masteringAvailable: masteringPresets.length > 0,
                    restoreAvailable,
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
            Enhance audio
          </button>
        </div>
      </div>
      <JobCard phase={phase} job={job} fileName={file?.name} errorMessage={errorMessage} onCancel={cancel} />
    </div>
  );
}
