import { useRef, useState, type MutableRefObject } from "react";
import type { GenerationModelSummary, InitImageResponse } from "../../lib/apiTypes";
import { uploadGenerationInitImage, type CreateGenerationJobParams } from "../../services/generation";
import {
  canvasHasPixels, exportMaskBlob, exportProtectMaskBlob, NO_EDGE, readMaskImageData, type EdgeAdjust,
} from "./editorMaskCanvas";
import { buildEditorJobParams, supportsEdgeAdjust } from "./eraseJobParams";
import type { EditMode } from "./InpaintModelSelect";
import { fitGenerationSize } from "./maskCanvas";

export interface EditorSubmitForm {
  mode: EditMode;
  modelId: string | null;
  prompt: string;
  erasePromptText: string;
  eraseNegative: string;
  replaceNegative: string;
  steps: number;
  guidance: number;
  seedText: string;
  strength: number;
  device: string | null;
  edge: EdgeAdjust;
}

interface EditorSubmitOptions {
  baseInfo: InitImageResponse | null;
  models: GenerationModelSummary[];
  form: EditorSubmitForm;
  editMaskRef: MutableRefObject<HTMLCanvasElement | null>;
  protectMaskRef: MutableRefObject<HTMLCanvasElement | null>;
  uploadSequence: MutableRefObject<number>;
  segmentPendingRef: MutableRefObject<boolean>;
  blocked: boolean;
  review: { start: (eraseMask: ImageData, protectMaskToken?: string, edge?: EdgeAdjust) => void; clear: () => void };
  submit: (params: CreateGenerationJobParams) => void;
  onError: (message: string | null) => void;
}

async function uploadPng(blob: Blob, name: string): Promise<string> {
  return (await uploadGenerationInitImage(new File([blob], name, { type: "image/png" }))).initImageToken;
}

function protectBlobFor(protection: HTMLCanvasElement | null): Promise<Blob | null> {
  return protection && canvasHasPixels(protection) ? exportProtectMaskBlob(protection) : Promise.resolve(null);
}

function parseSeed(seedText: string): number | null {
  const seed = seedText.trim() === "" ? null : Number.parseInt(seedText, 10);
  return seed !== null && Number.isFinite(seed) ? seed : null;
}

function buildParams(
  form: EditorSubmitForm,
  baseInfo: InitImageResponse,
  maskImageToken: string,
  protectMaskToken: string | undefined,
  edge: EdgeAdjust | null,
): CreateGenerationJobParams {
  const size = fitGenerationSize(baseInfo.width, baseInfo.height);
  const erasing = form.mode === "erase";
  const negative = erasing ? form.eraseNegative : form.replaceNegative;
  return {
    prompt: erasing ? form.erasePromptText : form.prompt,
    negativePrompt: negative.trim() === "" ? null : negative,
    modelId: form.modelId!,
    steps: form.steps,
    guidance: form.guidance,
    width: size.width,
    height: size.height,
    seed: parseSeed(form.seedText),
    device: form.device,
    autoUpscale: false,
    upscaleModelName: null,
    upscaleScale: null,
    upscaleModelId: null,
    initImageToken: baseInfo.initImageToken,
    maskImageToken,
    strength: form.strength,
    ...(protectMaskToken ? { protectMaskToken } : {}),
    ...(edge ? { maskGrowPx: edge.growPx, maskFeatherPx: edge.featherPx } : {}),
  };
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function useEditorSubmit(options: EditorSubmitOptions) {
  const [preparingJob, setPreparingJob] = useState(false);
  const preparingJobRef = useRef(false);

  function canStart(): boolean {
    const { editMaskRef, baseInfo, form, blocked, segmentPendingRef } = options;
    return Boolean(editMaskRef.current && baseInfo && form.modelId) && !blocked && !preparingJobRef.current
      && !segmentPendingRef.current;
  }

  function setPreparing(value: boolean): void {
    preparingJobRef.current = value;
    setPreparingJob(value);
  }

  async function prepareAndSubmit(sequence: number): Promise<void> {
    const { editMaskRef, protectMaskRef, uploadSequence, baseInfo, form, models, review, submit } = options;
    const canvas = editMaskRef.current!;
    const eraseMask = form.mode === "erase" ? readMaskImageData(canvas) : null;
    const [maskBlob, protectBlob] = await Promise.all([exportMaskBlob(canvas), protectBlobFor(protectMaskRef.current)]);
    const maskToken = await uploadPng(maskBlob, "mask.png");
    if (sequence !== uploadSequence.current) return;
    const protectMaskToken = protectBlob ? await uploadPng(protectBlob, "protect-mask.png") : undefined;
    if (sequence !== uploadSequence.current) return;
    const edge = supportsEdgeAdjust(form.mode, form.modelId, models) ? form.edge : null;
    const params = buildParams(form, baseInfo!, maskToken, protectMaskToken, edge);
    if (eraseMask) review.start(eraseMask, protectMaskToken, edge ?? NO_EDGE);
    else review.clear();
    submit(buildEditorJobParams(params, form.mode, models));
  }

  async function handleGenerate(): Promise<void> {
    if (!canStart()) return;
    const sequence = options.uploadSequence.current;
    setPreparing(true);
    options.onError(null);
    try {
      await prepareAndSubmit(sequence);
    } catch (error) {
      if (sequence === options.uploadSequence.current) options.onError(messageOf(error));
    } finally {
      setPreparing(false);
    }
  }

  return { preparingJob, handleGenerate };
}
