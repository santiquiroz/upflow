import { useQuery } from "@tanstack/react-query";
import { Film, UploadCloud } from "lucide-react";
import { useEffect, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { AccordionSection } from "../../components/AccordionSection";
import { DevicePicker } from "../../components/DevicePicker";
import { JobCard } from "../../components/JobCard";
import { ModelPicker } from "../../components/ModelPicker";
import { RuntimePicker, formatRuntimeSummary } from "../../components/RuntimePicker";
import { EncoderPicker, formatEncoderSummary } from "../../components/EncoderPicker";
import { SlowPresetCostHint } from "../../components/SlowPresetCostHint";
import { useAudioCapabilities } from "../../hooks/useAudioJob";
import { useVideoJob, type VideoJobPhase } from "../../hooks/useVideoJob";
import { getDevices, getModels } from "../../lib/api";
import type {
  DeviceInfoResponse,
  DevicesResponse,
  ModelResponse,
  UpscaleBackend,
  VideoEncoder,
  VideoProfileResponse,
} from "../../lib/apiTypes";
import { restoreLabel } from "../../lib/audioLabels";
import { formatDeviceSummary, formatModelSummary } from "./accordionSummaries";
import { AUDIO_ENHANCE_OPTIONS, AudioEnhanceControls } from "./AudioEnhanceControls";
import { FpsBoostControls, TARGET_FPS_OPTIONS, type FpsBoostValue } from "./FpsBoostControls";
import { VideoProfileControls } from "./VideoProfileControls";

const OUTPUT_CONTAINERS = ["mp4", "mkv"] as const;
const VIDEO_CODECS = [
  { value: "libx264", label: "H.264" },
  { value: "libx265", label: "H.265" },
] as const;
const VIDEO_PRESETS = ["medium", "slow", "veryslow"] as const;

const PROFILE_TOOLTIP =
  "A profile is a preset combining model, scale, codec and quality tuned for a content type. Picking one fills in the fields below; you can still override them.";
const MODEL_TOOLTIP =
  "Pick the AI model that upscales the video. Builtin models run on ncnn/Vulkan; ONNX models can run on CPU or GPU.";
const DEVICE_TOOLTIP =
  "Pick the compute device that runs the job. A CPU device can't run a builtin (ncnn) model — that needs a Vulkan GPU.";
const RUNTIME_TOOLTIP =
  "Choose which backend runs the model. Auto picks the fastest backend for your GPU (ONNX/DirectML is ~2x faster on modern GPUs for video); NCNN Vulkan is the portable fallback that runs on any GPU.";
const ENCODER_TOOLTIP =
  "How the final video is encoded. Software (x264/x265) is best quality per bit. Auto (GPU) uses your GPU's hardware encoder (NVENC/AMF/QSV) — far faster in 4K at a small quality/size cost, with automatic fallback to software.";
const FPS_BOOST_TOOLTIP =
  "Interpolate extra frames to raise the video's frame rate, either by a fixed multiplier or by targeting a specific frame rate. Only one mode can be active at a time.";
const AUDIO_TOOLTIP =
  "Keep the original audio track, optionally cleaned up with noise reduction. Enhancement requires audio to be kept.";
const ADVANCED_TOOLTIP =
  "Fine-tune the output container, video codec, encoder preset and quality (CRF). A lower CRF means higher quality and a larger file.";

function formatProfileSummary(profile: VideoProfileResponse | null) {
  return profile ? profile.label : "Select a profile…";
}

function formatFpsBoostSummary(value: FpsBoostValue) {
  if (value.fpsMultiplier > 1) {
    return <span className="font-mono-tabular">{value.fpsMultiplier}×</span>;
  }
  if (value.targetFps) {
    const target = TARGET_FPS_OPTIONS.find((option) => option.value === value.targetFps);
    return <span className="font-mono-tabular">{target?.label ?? value.targetFps}</span>;
  }
  return "Off";
}

function formatAudioSummary(keepAudio: boolean, audioEnhance: string | null) {
  if (!keepAudio) {
    return "Disabled";
  }
  const mode = AUDIO_ENHANCE_OPTIONS.find((option) => option.value === audioEnhance) ?? AUDIO_ENHANCE_OPTIONS[0];
  return `Kept · ${mode.label}`;
}

function formatAdvancedSummary(outputContainer: string, videoCodec: string, videoPreset: string, crf: number) {
  const codecLabel = VIDEO_CODECS.find((codec) => codec.value === videoCodec)?.label ?? videoCodec;
  return (
    <>
      {outputContainer.toUpperCase()} · {codecLabel} · {videoPreset} · CRF{" "}
      <span className="font-mono-tabular">{crf}</span>
    </>
  );
}

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
    <div className="flex flex-col gap-4">
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
  const [backend, setBackend] = useState<UpscaleBackend>("auto");
  const [videoEncoder, setVideoEncoder] = useState<VideoEncoder>("auto");
  const [scale, setScale] = useState<number | null>(null);
  const [outputContainer, setOutputContainer] = useState("mp4");
  const [videoCodec, setVideoCodec] = useState("libx264");
  const [videoPreset, setVideoPreset] = useState("medium");
  const [crf, setCrf] = useState(18);
  const [keepAudio, setKeepAudio] = useState(true);
  const [fpsBoost, setFpsBoost] = useState<FpsBoostValue>({ fpsMultiplier: 1, targetFps: null });
  const [audioEnhance, setAudioEnhance] = useState<string | null>(null);
  const [audioRestore, setAudioRestore] = useState<string | null>(null);

  const modelsQuery = useQuery({ queryKey: ["models"], queryFn: getModels });
  const devicesQuery = useQuery({ queryKey: ["devices"], queryFn: getDevices });
  const capabilitiesQuery = useAudioCapabilities();
  const { phase, job, errorMessage, submit, cancel, reset } = useVideoJob();

  const requiresGpu = resolveRequiresGpu(model);
  const restoreModes = capabilitiesQuery.data?.restoreModes ?? [];
  const restoreAvailable = restoreModes.length > 0;

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
      setAudioRestore(null);
    }
    setFpsBoost({ fpsMultiplier: 1, targetFps: null });
  }

  function handleKeepAudioChange(checked: boolean) {
    setKeepAudio(checked);
    if (!checked) {
      setAudioEnhance(null);
      setAudioRestore(null);
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
      backend,
      videoEncoder,
      scale,
      outputContainer,
      videoCodec,
      videoPreset,
      crf,
      keepAudio,
      fpsMultiplier: fpsBoost.fpsMultiplier,
      targetFps: fpsBoost.targetFps,
      audioEnhance,
      audioRestore: keepAudio && restoreAvailable ? audioRestore : null,
    });
  }

  const deviceUsable = isDeviceUsable(device, requiresGpu);
  const showNoGpuHint = model !== null && requiresGpu && !deviceUsable;
  const canSubmit = file !== null && profile !== null && scale !== null && deviceUsable && !isJobBusy(phase);

  return (
    <div className="grid grid-cols-[1fr_320px] gap-6 max-[900px]:grid-cols-1">
      <div className="flex flex-col gap-6">
        <Dropzone file={file} onFileSelected={handleFileSelected} />
        <AccordionSection
          title="Profile"
          summary={formatProfileSummary(profile)}
          tooltip={PROFILE_TOOLTIP}
          defaultOpen
        >
          <VideoProfileControls value={profile?.key ?? null} onChange={handleProfileChange} />
        </AccordionSection>
        <AccordionSection title="Model" summary={formatModelSummary(model)} tooltip={MODEL_TOOLTIP}>
          <ModelPicker value={model?.id ?? null} onChange={setModel} />
        </AccordionSection>
        <AccordionSection title="Device" summary={formatDeviceSummary(device)} tooltip={DEVICE_TOOLTIP}>
          <DevicePicker value={device?.id ?? null} onChange={setDevice} requiresGpu={requiresGpu} />
        </AccordionSection>
        <AccordionSection title="Runtime" summary={formatRuntimeSummary(backend)} tooltip={RUNTIME_TOOLTIP}>
          <RuntimePicker value={backend} onChange={setBackend} />
        </AccordionSection>
        <AccordionSection title="Encoder" summary={formatEncoderSummary(videoEncoder)} tooltip={ENCODER_TOOLTIP}>
          <EncoderPicker value={videoEncoder} onChange={setVideoEncoder} />
          <SlowPresetCostHint encoder={videoEncoder} preset={videoPreset} scale={scale} />
        </AccordionSection>
        <AccordionSection
          title="FPS boost"
          summary={formatFpsBoostSummary(fpsBoost)}
          tooltip={FPS_BOOST_TOOLTIP}
        >
          <FpsBoostControls value={fpsBoost} onChange={setFpsBoost} />
        </AccordionSection>
        <AccordionSection
          title="Audio"
          summary={formatAudioSummary(keepAudio, audioEnhance)}
          tooltip={AUDIO_TOOLTIP}
        >
          <div className="flex flex-col gap-3">
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
            {keepAudio && restoreAvailable && (
              <fieldset className="flex flex-col gap-1.5">
                <legend className="text-sm text-text">Restore compression (experimental)</legend>
                {[null, ...restoreModes].map((mode) => (
                  <label key={mode ?? "none"} className="flex items-center gap-2 text-sm text-text">
                    <input
                      type="radio"
                      name="video-audio-restore"
                      checked={audioRestore === mode}
                      onChange={() => setAudioRestore(mode)}
                      className="h-3.5 w-3.5 accent-accent"
                    />
                    {mode === null ? "Off" : restoreLabel(mode)}
                  </label>
                ))}
                {audioRestore === "audiosr" && (
                  <p role="status" className="text-xs text-warn">
                    AudioSR (diffusion) adds roughly 2 minutes of processing per minute of audio on GPU.
                  </p>
                )}
              </fieldset>
            )}
          </div>
        </AccordionSection>
        <AccordionSection
          title="Advanced"
          summary={formatAdvancedSummary(outputContainer, videoCodec, videoPreset, crf)}
          tooltip={ADVANCED_TOOLTIP}
        >
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
        </AccordionSection>
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
      <JobCard phase={phase} job={job} fileName={file?.name} errorMessage={errorMessage} onCancel={cancel} />
    </div>
  );
}
