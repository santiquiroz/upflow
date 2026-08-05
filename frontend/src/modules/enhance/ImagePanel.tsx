import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "../../i18n/LocaleProvider";
import { UploadCloud, Wand2 } from "lucide-react";
import { useEffect, useMemo, useState, type ChangeEvent, type DragEvent } from "react";
import { AccordionSection } from "../../components/AccordionSection";
import { DevicePicker } from "../../components/DevicePicker";
import { JobCard } from "../../components/JobCard";
import { ModelPicker } from "../../components/ModelPicker";
import { useImageJob, type ImageJobPhase } from "../../hooks/useImageJob";
import { getDevices, getEngineInfo } from "../../lib/api";
import type { DeviceInfoResponse, DevicesResponse, ModelResponse } from "../../lib/apiTypes";
import { formatDeviceSummary, formatModelSummary } from "./accordionSummaries";
import { ScaleFormatControls } from "./ScaleFormatControls";
import { correctScaleFor, scalesForModel } from "./scalesForModel";
import { exceedsUploadLimit, formatMegabytes } from "./uploadLimit";

const MODEL_TOOLTIP = "enhance.model.tooltip.image";
const DEVICE_TOOLTIP = "enhance.device.tooltip";
const SCALE_FORMAT_TOOLTIP = "enhance.image.scaleAndFormat.tooltip";

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
      htmlFor="image-file-input"
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
      className="flex cursor-pointer flex-col items-center gap-2 rounded border border-dashed border-border bg-surface px-6 py-10 text-center transition-[border-color] duration-fast hover:border-accent"
    >
      <UploadCloud aria-hidden="true" className="h-6 w-6 text-text-faint" strokeWidth={1.5} />
      <span className="text-sm text-text">{file ? file.name : t("enhance.image.dropzone")}</span>
      <span className="text-xs text-text-faint">PNG, JPG, WEBP</span>
      <input id="image-file-input" type="file" accept="image/*" className="sr-only" onChange={handleInputChange} />
    </label>
  );
}

export function ImagePanel() {
  const { t } = useTranslation();
  const [file, setFile] = useState<File | null>(null);
  const [rejectedUpload, setRejectedUpload] = useState<string | null>(null);
  const [model, setModel] = useState<ModelResponse | null>(null);
  const [device, setDevice] = useState<DeviceInfoResponse | null>(null);
  const [scale, setScale] = useState<number | null>(null);
  const [format, setFormat] = useState("png");

  const engineQuery = useQuery({ queryKey: ["engine"], queryFn: getEngineInfo });
  const devicesQuery = useQuery({ queryKey: ["devices"], queryFn: getDevices });
  const { phase, job, errorMessage, submit, cancel, reset, uploadPercent } = useImageJob();

  // Las escalas que se ofrecen son las del MODELO elegido, no las del motor: el
  // backend rechaza el job si no coinciden, y hasta ahora eso se descubria
  // despues de subir el archivo.
  // Memoizado: devuelve un array nuevo en cada llamada y sin esto el efecto de
  // abajo se dispararia en cada render.
  const supportedModels = engineQuery.data?.supportedModels;
  const allowedScales = engineQuery.data?.allowedScales;
  const offeredScales = useMemo(
    () => scalesForModel(model, supportedModels ?? [], allowedScales ?? []),
    [model, supportedModels, allowedScales],
  );
  const requiresGpu = resolveRequiresGpu(model);

  useEffect(() => {
    if (scale === null) {
      const defaultScale = resolveDefaultScale(offeredScales);
      if (defaultScale !== null) {
        setScale(defaultScale);
      }
      return;
    }
    // Cambiar de modelo puede dejar seleccionada una escala que el nuevo no hace.
    const corrected = correctScaleFor(scale, offeredScales);
    if (corrected !== null && corrected !== scale) {
      setScale(corrected);
    }
  }, [offeredScales, scale]);

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
    // Antes de tocar la red: el servidor corta la subida mientras la recibe, y
    // enterarse al final de un archivo grande es la peor forma de enterarse.
    const limitMb = engineQuery.data?.maxUploadMb ?? null;
    if (exceedsUploadLimit(selected.size, limitMb)) {
      setRejectedUpload(
        t("upload.tooLarge", {
          size: formatMegabytes(selected.size),
          limit: `${limitMb} MB`,
        }),
      );
      setFile(null);
      return;
    }
    setRejectedUpload(null);
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
        {rejectedUpload && (
          <p role="alert" className="text-xs text-danger">
            {rejectedUpload}
          </p>
        )}
        <AccordionSection title={t("enhance.summary.model")} summary={formatModelSummary(model, t)} tooltip={t(MODEL_TOOLTIP)} defaultOpen>
          <ModelPicker value={model?.id ?? null} onChange={setModel} />
        </AccordionSection>
        <AccordionSection title={t("enhance.summary.device")} summary={formatDeviceSummary(device, t)} tooltip={t(DEVICE_TOOLTIP)}>
          <DevicePicker value={device?.id ?? null} onChange={setDevice} requiresGpu={requiresGpu} />
        </AccordionSection>
        <AccordionSection
          title={t("enhance.image.scaleAndFormat")}
          summary={formatScaleFormatSummary(scale, format)}
          tooltip={t(SCALE_FORMAT_TOOLTIP)}
        >
          <ScaleFormatControls
            allowedScales={offeredScales}
            scale={scale ?? offeredScales[0] ?? 4}
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
      <JobCard
        phase={phase}
        job={job}
        fileName={file?.name}
        errorMessage={errorMessage}
        onCancel={cancel}
        uploadPercent={uploadPercent}
      />
    </div>
  );
}
