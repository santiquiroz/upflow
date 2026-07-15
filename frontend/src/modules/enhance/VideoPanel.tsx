import { useQuery } from "@tanstack/react-query";
import { Film, UploadCloud } from "lucide-react";
import { useEffect, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { DevicePicker } from "../../components/DevicePicker";
import { JobCard } from "../../components/JobCard";
import { ModelPicker } from "../../components/ModelPicker";
import { useVideoJob, type VideoJobPhase } from "../../hooks/useVideoJob";
import { getDevices, getModels } from "../../lib/api";
import type { DeviceInfoResponse, DevicesResponse, ModelResponse, VideoProfileResponse } from "../../lib/apiTypes";
import { AudioEnhanceControls } from "./AudioEnhanceControls";
import { FpsBoostControls, type FpsBoostValue } from "./FpsBoostControls";
import { VideoProfileControls } from "./VideoProfileControls";

const OUTPUT_CONTAINERS = ["mp4", "mkv"] as const;
const VIDEO_CODECS = [
  { value: "libx264", label: "H.264" },
  { value: "libx265", label: "H.265" },
] as const;
const VIDEO_PRESETS = ["medium", "slow", "veryslow"] as const;

function resolveRequiresGpu(model: ModelResponse | null): boolean {
  return model?.kind === "builtin-ncnn";
}

function isJobBusy(phase: VideoJobPhase): boolean {
  return phase === "uploading" || phase === "queued" || phase === "running";
}

// A builtin-ncnn model needs a Vulkan GPU, so a cpu device can never run it.
// On a GPU-less machine resolvePreferredDevice falls back to a (disabled) cpu
// device, so guard here too -- otherwise an unrunnable job would be submittable.
function isDeviceUsable(device: DeviceInfoResponse | null, requiresGpu: boolean): boolean {
  if (device === null) {
    return false;
  }
  return !(requiresGpu && device.kind === "cpu");
}

function resolvePreferredDevice(
  devicesResponse: DevicesResponse,
  requiresGpu: boolean,
): DeviceInfoResponse | null {
  const { devices, defaultDeviceId } = devicesResponse;
  const defaultDevice = devices.find((candidate) => candidate.id === defaultDeviceId) ?? null;
  const firstNonCpuDevice = devices.find((candidate) => candidate.kind !== "cpu") ?? null;
  if (requiresGpu) {
    if (defaultDevice && defaultDevice.kind !== "cpu") {
      return defaultDevice;
    }
    return firstNonCpuDevice ?? devices[0] ?? null;
  }
  return defaultDevice ?? devices[0] ?? null;
}

function segmentButtonClassName(isActive: boolean): string {
  const base =
    "rounded-sm border px-3 py-1.5 text-sm transition-[background-color,border-color,color] duration-fast focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";
  if (isActive) {
    return `${base} border-accent bg-accent text-bg`;
  }
  return `${base} border-border bg-surface text-text-dim hover:border-text-faint hover:text-text`;
}

function SegmentedField<T extends string>({
  legend,
  options,
  value,
  onChange,
}: {
  legend: string;
  options: readonly { value: T; label: string }[];
  value: T;
  onChange: (value: T) => void;
}) {
  return (
    <div className="flex flex-col gap-2">
      <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">{legend}</span>
      <div role="group" aria-label={legend} className="flex flex-wrap gap-2">
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            aria-pressed={option.value === value}
            className={segmentButtonClassName(option.value === value)}
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function AdvancedVideoControls({
  outputContainer,
  onOutputContainerChange,
  videoCodec,
  onVideoCodecChange,
  videoPreset,
  onVideoPresetChange,
  crf,
  onCrfChange,
}: {
  outputContainer: string;
  onOutputContainerChange: (value: string) => void;
  videoCodec: string;
  onVideoCodecChange: (value: string) => void;
  videoPreset: string;
  onVideoPresetChange: (value: string) => void;
  crf: number;
  onCrfChange: (value: number) => void;
}) {
  return (
    <details className="rounded border border-border bg-surface">
      <summary className="cursor-pointer select-none px-3 py-2 text-sm text-text-dim">Advanced options</summary>
      <div className="flex flex-col gap-4 border-t border-border p-3">
        <SegmentedField
          legend="Container"
          options={OUTPUT_CONTAINERS.map((value) => ({ value, label: value.toUpperCase() }))}
          value={outputContainer}
          onChange={onOutputContainerChange}
        />
        <SegmentedField legend="Codec" options={VIDEO_CODECS} value={videoCodec} onChange={onVideoCodecChange} />
        <SegmentedField
          legend="Preset"
          options={VIDEO_PRESETS.map((value) => ({ value, label: value }))}
          value={videoPreset}
          onChange={onVideoPresetChange}
        />
        <label className="flex w-24 flex-col gap-1">
          <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">CRF</span>
          <input
            type="number"
            min={10}
            max={28}
            value={crf}
            onChange={(event) => onCrfChange(Number(event.target.value))}
            className="font-mono-tabular rounded-sm border border-border bg-surface px-2 py-1 text-sm text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
          />
        </label>
      </div>
    </details>
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
      htmlFor="video-file-input"
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
      className="flex cursor-pointer flex-col items-center gap-2 rounded border border-dashed border-border bg-surface px-6 py-10 text-center transition-[border-color] duration-fast hover:border-accent"
    >
      <UploadCloud aria-hidden="true" className="h-6 w-6 text-text-faint" strokeWidth={1.5} />
      <span className="text-sm text-text">{file ? file.name : "Drop a video here or click to browse"}</span>
      <span className="text-xs text-text-faint">MP4, MKV, MOV</span>
      <input id="video-file-input" type="file" accept="video/*" className="sr-only" onChange={handleInputChange} />
    </label>
  );
}

function resolveModelForProfile(
  profile: VideoProfileResponse | null,
  models: ModelResponse[],
): ModelResponse | null {
  if (!profile) {
    return null;
  }
  return models.find((candidate) => candidate.id === profile.modelKey) ?? null;
}

export function VideoPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [profile, setProfile] = useState<VideoProfileResponse | null>(null);
  const [model, setModel] = useState<ModelResponse | null>(null);
  const [device, setDevice] = useState<DeviceInfoResponse | null>(null);
  const [scale, setScale] = useState<number | null>(null);
  const [outputContainer, setOutputContainer] = useState("mp4");
  const [videoCodec, setVideoCodec] = useState("libx264");
  const [videoPreset, setVideoPreset] = useState("medium");
  const [crf, setCrf] = useState(18);
  const [keepAudio, setKeepAudio] = useState(true);
  const [fpsBoost, setFpsBoost] = useState<FpsBoostValue>({ fpsMultiplier: 1, targetFps: null });
  const [audioEnhance, setAudioEnhance] = useState<string | null>(null);

  const modelsQuery = useQuery({ queryKey: ["models"], queryFn: getModels });
  const devicesQuery = useQuery({ queryKey: ["devices"], queryFn: getDevices });
  const { phase, job, errorMessage, submit, reset } = useVideoJob();

  const requiresGpu = resolveRequiresGpu(model);

  // Only re-applies the profile's default model the first time a given profile
  // becomes selected (including the async case where modelsQuery resolves after
  // the click) -- excluding `model` from the deps means a manual ModelPicker
  // override afterward is never fought back to the profile default.
  const appliedProfileKeyRef = useRef<string | null>(null);
  useEffect(() => {
    if (!modelsQuery.data || !profile) {
      return;
    }
    if (appliedProfileKeyRef.current === profile.key) {
      return;
    }
    appliedProfileKeyRef.current = profile.key;
    setModel(resolveModelForProfile(profile, modelsQuery.data.models));
  }, [profile, modelsQuery.data]);

  useEffect(() => {
    if (!devicesQuery.data) {
      return;
    }
    const needsReassignment = device === null || (requiresGpu && device.kind === "cpu");
    if (!needsReassignment) {
      return;
    }
    const preferred = resolvePreferredDevice(devicesQuery.data, requiresGpu);
    if (preferred && preferred.id !== device?.id) {
      setDevice(preferred);
    }
  }, [devicesQuery.data, requiresGpu, device]);

  function handleFileSelected(selected: File) {
    setFile(selected);
    reset();
  }

  function handleProfileChange(nextProfile: VideoProfileResponse) {
    setProfile(nextProfile);
    setScale(nextProfile.scale);
    setOutputContainer("mp4");
    setVideoCodec(nextProfile.videoCodec);
    setVideoPreset(nextProfile.videoPreset);
    setCrf(nextProfile.crf);
    setKeepAudio(nextProfile.keepAudio);
    if (!nextProfile.keepAudio) {
      setAudioEnhance(null);
    }
    setFpsBoost({ fpsMultiplier: 1, targetFps: null });
  }

  function handleKeepAudioChange(checked: boolean) {
    setKeepAudio(checked);
    if (!checked) {
      setAudioEnhance(null);
    }
  }

  function handleSubmit() {
    if (!file || !profile || scale === null) {
      return;
    }
    submit({
      file,
      profileKey: profile.key,
      modelId: model?.id ?? null,
      device: device?.id ?? null,
      scale,
      outputContainer,
      videoCodec,
      videoPreset,
      crf,
      keepAudio,
      fpsMultiplier: fpsBoost.fpsMultiplier,
      targetFps: fpsBoost.targetFps,
      audioEnhance,
    });
  }

  const deviceUsable = isDeviceUsable(device, requiresGpu);
  const showNoGpuHint = model !== null && requiresGpu && !deviceUsable;
  const canSubmit = file !== null && profile !== null && scale !== null && deviceUsable && !isJobBusy(phase);

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="font-heading text-2xl font-semibold text-text">Enhance video</h1>
        <p className="mt-1 text-sm text-text-dim">Upscale, interpolate and clean up a video.</p>
      </div>
      <div className="grid grid-cols-[1fr_320px] gap-6 max-[900px]:grid-cols-1">
        <div className="flex flex-col gap-6">
          <Dropzone file={file} onFileSelected={handleFileSelected} />
          <VideoProfileControls value={profile?.key ?? null} onChange={handleProfileChange} />
          <ModelPicker value={model?.id ?? null} onChange={setModel} />
          <DevicePicker value={device?.id ?? null} onChange={setDevice} requiresGpu={requiresGpu} />
          <FpsBoostControls value={fpsBoost} onChange={setFpsBoost} />
          <label className="flex items-center gap-2 text-sm text-text">
            <input
              type="checkbox"
              checked={keepAudio}
              onChange={(event) => handleKeepAudioChange(event.target.checked)}
              className="h-3.5 w-3.5 accent-accent"
            />
            Keep original audio
          </label>
          <AudioEnhanceControls value={audioEnhance} onChange={setAudioEnhance} keepAudio={keepAudio} />
          <AdvancedVideoControls
            outputContainer={outputContainer}
            onOutputContainerChange={setOutputContainer}
            videoCodec={videoCodec}
            onVideoCodecChange={setVideoCodec}
            videoPreset={videoPreset}
            onVideoPresetChange={setVideoPreset}
            crf={crf}
            onCrfChange={setCrf}
          />
          <div className="flex flex-col gap-2">
            {showNoGpuHint && (
              <p role="status" className="text-xs text-warn">
                This profile's model requires a Vulkan GPU; no GPU device is available.
              </p>
            )}
            <button
              type="button"
              onClick={handleSubmit}
              disabled={!canSubmit}
              className="inline-flex w-fit items-center gap-2 rounded bg-accent px-4 py-2 text-sm font-medium text-bg transition-[background-color,opacity] duration-fast hover:bg-accent-hover active:bg-accent-press disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
            >
              <Film aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
              Upscale video
            </button>
          </div>
        </div>
        <JobCard phase={phase} job={job} fileName={file?.name} errorMessage={errorMessage} />
      </div>
    </div>
  );
}
