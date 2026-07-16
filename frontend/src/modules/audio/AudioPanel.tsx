import { AudioWaveform, UploadCloud } from "lucide-react";
import { useState, type ChangeEvent, type DragEvent } from "react";
import { AccordionSection } from "../../components/AccordionSection";
import { AUTO_DEVICE, DevicePicker } from "../../components/DevicePicker";
import { JobCard } from "../../components/JobCard";
import { useAudioCapabilities, useAudioJob, type AudioJobPhase } from "../../hooks/useAudioJob";
import { denoiseLabel, restoreLabel } from "../../lib/audioLabels";
import type { DeviceInfoResponse } from "../../lib/apiTypes";
import { formatDeviceSummary } from "../enhance/accordionSummaries";

const DENOISE_TOOLTIP =
  "Remove background noise with an AI denoiser. DeepFilterNet is stronger; RNNoise is lighter. Runs before restoration.";
const RESTORE_TOOLTIP =
  "Reconstruct high frequencies lost to lossy compression (MP3/AAC). Experimental — quality varies by source.";
const DEVICE_TOOLTIP = "Pick the compute device that runs the restoration model (CPU or a DirectML GPU).";

const RESTORE_APOLLO = "apollo";

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

const RESTORE_OPTIONS: readonly ModeOption[] = [
  { value: null, label: "None" },
  { value: RESTORE_APOLLO, label: "Apollo", experimental: true },
];

export function AudioPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [denoise, setDenoise] = useState<string | null>(null);
  const [restore, setRestore] = useState<string | null>(null);
  const [device, setDevice] = useState<DeviceInfoResponse | null>(AUTO_DEVICE);

  const capabilitiesQuery = useAudioCapabilities();
  const { phase, job, errorMessage, submit, reset } = useAudioJob();

  const denoiseModes = capabilitiesQuery.data?.denoiseModes ?? [];
  const restoreAvailable = capabilitiesQuery.data?.restoreAvailable ?? false;

  function handleFileSelected(selected: File) {
    setFile(selected);
    reset();
  }

  function handleSubmit() {
    if (!file || (denoise === null && restore === null)) {
      return;
    }
    submit({ file, denoise, restore, device: device?.id ?? null });
  }

  const hasSelection = denoise !== null || restore !== null;
  const canSubmit = file !== null && hasSelection && !isJobBusy(phase);

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
        {restoreAvailable && (
          <AccordionSection title="Restore" summary={restoreLabel(restore)} tooltip={RESTORE_TOOLTIP}>
            <ModeSegmentedControl
              legend="Restore"
              options={RESTORE_OPTIONS}
              value={restore}
              onChange={setRestore}
            />
          </AccordionSection>
        )}
        <AccordionSection title="Device" summary={formatDeviceSummary(device)} tooltip={DEVICE_TOOLTIP}>
          <DevicePicker value={device?.id ?? null} onChange={setDevice} requiresGpu={false} />
        </AccordionSection>
        <div className="flex flex-col gap-2">
          {!hasSelection && (
            <p role="status" className="text-xs text-text-faint">
              Pick at least one of Denoise or Restore.
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
      <JobCard phase={phase} job={job} fileName={file?.name} errorMessage={errorMessage} />
    </div>
  );
}
