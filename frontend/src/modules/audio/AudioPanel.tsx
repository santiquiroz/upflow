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
import type { DeviceInfoResponse } from "../../lib/apiTypes";
import { formatDeviceSummary } from "../enhance/accordionSummaries";
import { VoiceChainPanel, voiceSummaryKey } from "./VoiceChainPanel";
import { useVoiceSelection } from "./useVoiceSelection";

type AudioOutputFormat = "flac" | "wav" | "mp3";

const OUTPUT_FORMAT_OPTIONS: readonly FormatOption<AudioOutputFormat>[] = [
  { value: "flac", label: "FLAC (recommended)", description: "Lossless quality, about 50% smaller than WAV." },
  { value: "wav", label: "WAV", description: "Lossless, uncompressed. Universal compatibility." },
  { value: "mp3", label: "MP3", description: "Lossy, smallest file — only if size matters more than quality." },
];

const DENOISE_TOOLTIP =
  "Remove background noise with an AI denoiser. DeepFilterNet is stronger; RNNoise is lighter. Runs before restoration.";
const MASTERING_TOOLTIP =
  "Deja el audio al volumen que piden las plataformas, medido segun el estandar EBU R128. Se mide primero y se corrige despues con esa medicion, que es como se hace un master de verdad: en una sola pasada el volumen bombea.";

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
      <span className="text-sm text-text">{file ? file.name : "Drop an audio file here or click to browse"}</span>
      <span className="text-xs text-text-faint">WAV, MP3, FLAC, M4A, OGG, OPUS</span>
      <input id="audio-file-input" type="file" accept="audio/*" className="sr-only" onChange={handleInputChange} />
    </label>
  );
}

function buildDenoiseOptions(denoiseModes: string[]): ModeOption[] {
  return [{ value: null, label: "None" }, ...denoiseModes.map((mode) => ({ value: mode, label: denoiseLabel(mode) }))];
}

function buildRestoreOptions(restoreModes: string[]): ModeOption[] {
  return [
    { value: null, label: "None" },
    ...restoreModes.map((mode) => ({ value: mode, label: restoreLabel(mode), experimental: true })),
  ];
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
  const { t } = useTranslation();

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
        <AccordionSection title="Denoise" summary={denoiseLabel(denoise)} tooltip={DENOISE_TOOLTIP} defaultOpen>
          <ModeSegmentedControl
            legend="Denoise"
            options={buildDenoiseOptions(denoiseModes)}
            value={denoise}
            onChange={setDenoise}
          />
        </AccordionSection>
        {masteringPresets.length > 0 && (
          <AccordionSection
            title="Acabado"
            summary={
              masteringPresets.find((p) => p.id === master)?.label ?? "Sin nivelar"
            }
            tooltip={MASTERING_TOOLTIP}
          >
            <ModeSegmentedControl
              legend="Acabado"
              options={[
                { value: null, label: "Sin nivelar" },
                ...masteringPresets.map((preset) => ({
                  value: preset.id,
                  label: preset.label,
                })),
              ]}
              value={master}
              onChange={setMaster}
            />
            {master !== null && (
              <p role="status" className="mt-2 text-xs text-text-dim">
                {masteringPresets.find((p) => p.id === master)?.description}{" "}
                Objetivo: {masteringPresets.find((p) => p.id === master)?.targetLufs} LUFS.
              </p>
            )}
          </AccordionSection>
        )}
        {restoreAvailable && (
          <AccordionSection title="Restore" summary={restoreLabel(restore)} tooltip={RESTORE_TOOLTIP}>
            <ModeSegmentedControl
              legend="Restore"
              options={buildRestoreOptions(restoreModes)}
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
        <AccordionSection title="Device" summary={formatDeviceSummary(device)} tooltip={DEVICE_TOOLTIP}>
          <DevicePicker value={device?.id ?? null} onChange={setDevice} requiresGpu={false} allowAuto={false} />
        </AccordionSection>
        <FormatOptionFieldset
          legend="Output format"
          name="audio-output-format"
          options={OUTPUT_FORMAT_OPTIONS}
          value={outputFormat}
          onChange={setOutputFormat}
        />
        <div className="flex flex-col gap-2">
          {!hasSelection && (
            <p role="status" className="text-xs text-text-faint">
              Pick at least one of Denoise, Restore or Voice.
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
