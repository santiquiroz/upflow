import { useQuery } from "@tanstack/react-query";
import { UploadCloud, Wand2 } from "lucide-react";
import { useEffect, useState, type ChangeEvent, type DragEvent } from "react";
import { AccordionSection } from "../../components/AccordionSection";
import { DevicePicker } from "../../components/DevicePicker";
import { JobCard } from "../../components/JobCard";
import { ModelPicker } from "../../components/ModelPicker";
import { useImageJob, type ImageJobPhase } from "../../hooks/useImageJob";
import { getDevices, getEngineInfo } from "../../lib/api";
import type { DeviceInfoResponse, DevicesResponse, ModelResponse } from "../../lib/apiTypes";
import { formatDeviceSummary, formatModelSummary } from "./accordionSummaries";
import { ScaleFormatControls } from "./ScaleFormatControls";

const MODEL_TOOLTIP =
  "Pick the AI model that upscales the image. Builtin models run on ncnn/Vulkan; ONNX models can run on CPU or GPU.";
const DEVICE_TOOLTIP =
  "Pick the compute device that runs the job. A CPU device can't run a builtin (ncnn) model — that needs a Vulkan GPU.";
const SCALE_FORMAT_TOOLTIP = "Choose the output resolution multiplier and the image file format.";

function formatScaleFormatSummary(scale: number | null, format: string) {
  const scaleLabel = scale !== null ? `${scale}x` : "—";
  return (
    <>
      <span className="font-mono-tabular">{scaleLabel}</span> · {format.toUpperCase()}
    </>
  );
}

function resolveRequiresGpu(model: ModelResponse | null): boolean {
  return model?.kind === "builtin-ncnn";
}

function resolveDefaultScale(allowedScales: number[]): number | null {
  if (allowedScales.length === 0) {
    return null;
  }
  return allowedScales.includes(4) ? 4 : allowedScales[0];
}

function isJobBusy(phase: ImageJobPhase): boolean {
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
      htmlFor="image-file-input"
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
      className="flex cursor-pointer flex-col items-center gap-2 rounded border border-dashed border-border bg-surface px-6 py-10 text-center transition-[border-color] duration-fast hover:border-accent"
    >
      <UploadCloud aria-hidden="true" className="h-6 w-6 text-text-faint" strokeWidth={1.5} />
      <span className="text-sm text-text">{file ? file.name : "Drop an image here or click to browse"}</span>
      <span className="text-xs text-text-faint">PNG, JPG, WEBP</span>
      <input id="image-file-input" type="file" accept="image/*" className="sr-only" onChange={handleInputChange} />
    </label>
  );
}

export function ImagePanel() {
  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState<ModelResponse | null>(null);
  const [device, setDevice] = useState<DeviceInfoResponse | null>(null);
  const [scale, setScale] = useState<number | null>(null);
  const [format, setFormat] = useState("png");

  const engineQuery = useQuery({ queryKey: ["engine"], queryFn: getEngineInfo });
  const devicesQuery = useQuery({ queryKey: ["devices"], queryFn: getDevices });
  const { phase, job, errorMessage, submit, reset } = useImageJob();

  const allowedScales = engineQuery.data?.allowedScales ?? [];
  const requiresGpu = resolveRequiresGpu(model);

  useEffect(() => {
    if (scale !== null) {
      return;
    }
    const defaultScale = resolveDefaultScale(allowedScales);
    if (defaultScale !== null) {
      setScale(defaultScale);
    }
  }, [allowedScales, scale]);

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

  function handleSubmit() {
    if (!file || !model || scale === null) {
      return;
    }
    submit({ file, modelId: model.id, device: device?.id ?? null, scale, outputFormat: format });
  }

  const deviceUsable = isDeviceUsable(device, requiresGpu);
  const showNoGpuHint = model !== null && requiresGpu && !deviceUsable;
  const canSubmit =
    file !== null && model !== null && scale !== null && deviceUsable && !isJobBusy(phase);

  return (
    <div className="grid grid-cols-[1fr_320px] gap-6 max-[900px]:grid-cols-1">
      <div className="flex flex-col gap-6">
        <Dropzone file={file} onFileSelected={handleFileSelected} />
        <AccordionSection title="Model" summary={formatModelSummary(model)} tooltip={MODEL_TOOLTIP} defaultOpen>
          <ModelPicker value={model?.id ?? null} onChange={setModel} />
        </AccordionSection>
        <AccordionSection title="Device" summary={formatDeviceSummary(device)} tooltip={DEVICE_TOOLTIP}>
          <DevicePicker value={device?.id ?? null} onChange={setDevice} requiresGpu={requiresGpu} />
        </AccordionSection>
        <AccordionSection
          title="Scale & format"
          summary={formatScaleFormatSummary(scale, format)}
          tooltip={SCALE_FORMAT_TOOLTIP}
        >
          <ScaleFormatControls
            allowedScales={allowedScales}
            scale={scale ?? allowedScales[0] ?? 4}
            onScaleChange={setScale}
            format={format}
            onFormatChange={setFormat}
          />
        </AccordionSection>
        <div className="flex flex-col gap-2">
          {showNoGpuHint && (
            <p role="status" className="text-xs text-warn">
              This builtin model requires a Vulkan GPU; no GPU device is available.
            </p>
          )}
          <button
            type="button"
            onClick={handleSubmit}
            disabled={!canSubmit}
            className="inline-flex w-fit items-center gap-2 rounded bg-accent px-4 py-2 text-sm font-medium text-bg transition-[background-color,opacity] duration-fast hover:bg-accent-hover active:bg-accent-press disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
          >
            <Wand2 aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
            Upscale
          </button>
        </div>
      </div>
      <JobCard phase={phase} job={job} fileName={file?.name} errorMessage={errorMessage} />
    </div>
  );
}
