import { useEffect, useRef, useState } from "react";
import { useGenerationJob } from "../../hooks/useGenerationJob";
import { useTranslation } from "../../i18n/LocaleProvider";
import type { EraseZone, GenerationModelSummary, InitImageResponse } from "../../lib/apiTypes";
import { uploadGenerationInitImage, type CreateGenerationJobParams } from "../../services/generation";
import { exportImageDataMaskBlob } from "./editorMaskCanvas";
import { buildRedoJobParams } from "./eraseJobParams";
import { fitGenerationSize } from "./maskCanvas";
import type { ReviewSession } from "./useReviewSession";
import { clipMaskToBox, redoneZone, type Box, type RedoEngine } from "./zones";

export interface RedoDefaults {
  prompt: string;
  negativePrompt: string | null;
  steps: number;
  guidance: number;
  device: string | null;
}

interface ZoneRedoOptions {
  session: ReviewSession | null;
  baseInfo: InitImageResponse | null;
  models: GenerationModelSummary[];
  defaults: RedoDefaults;
  onDone: (zone: EraseZone, resultUrl: string | null) => void;
}

interface PendingRedo {
  zone: EraseZone;
  engine: RedoEngine;
}

async function uploadResultImage(url: string): Promise<string> {
  const blob = await (await fetch(url)).blob();
  return (await uploadGenerationInitImage(new File([blob], "redo-base.png", { type: "image/png" }))).initImageToken;
}

async function uploadZoneMask(eraseMask: ImageData, box: Box): Promise<string> {
  const blob = await exportImageDataMaskBlob(clipMaskToBox(eraseMask, box));
  return (await uploadGenerationInitImage(new File([blob], "redo-mask.png", { type: "image/png" }))).initImageToken;
}

function baseParams(
  session: ReviewSession,
  baseInfo: InitImageResponse,
  defaults: RedoDefaults,
  tokens: [string, string],
): CreateGenerationJobParams {
  const size = fitGenerationSize(baseInfo.width, baseInfo.height);
  return {
    ...defaults,
    negativePrompt: defaults.negativePrompt?.trim() ? defaults.negativePrompt : null,
    modelId: "eraser-auto",
    width: size.width,
    height: size.height,
    seed: null,
    autoUpscale: false,
    upscaleModelName: null,
    upscaleScale: null,
    upscaleModelId: null,
    initImageToken: tokens[0],
    maskImageToken: tokens[1],
    strength: 1,
    maskGrowPx: session.maskGrowPx,
    maskFeatherPx: session.maskFeatherPx,
    ...(session.protectMaskToken ? { protectMaskToken: session.protectMaskToken } : {}),
  };
}

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function useZoneRedo({ session, baseInfo, models, defaults, onDone }: ZoneRedoOptions) {
  const { t } = useTranslation();
  const { job, errorMessage, submit, reset } = useGenerationJob();
  const [pending, setPendingState] = useState<PendingRedo | null>(null);
  const [errors, setErrors] = useState<Record<number, string>>({});
  const maskRef = useRef<ImageData | null>(null);
  const pendingRef = useRef<PendingRedo | null>(null);
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;
  const eraseMask = session?.eraseMask ?? null;
  const cancelledMessage = t("job.failed");

  function setPending(next: PendingRedo | null): void {
    pendingRef.current = next;
    setPendingState(next);
  }

  useEffect(() => {
    maskRef.current = eraseMask;
    setPending(null);
    setErrors({});
  }, [eraseMask]);

  function fail(index: number, message: string): void {
    setErrors((current) => ({ ...current, [index]: message }));
    setPending(null);
  }

  useEffect(() => {
    const current = pendingRef.current;
    if (!current) return;
    if (job?.status === "completed") {
      setPending(null);
      onDoneRef.current(redoneZone(current.zone, current.engine, job), job.downloadUrl);
      return;
    }
    const failure = errorMessage ?? (job?.status === "cancelled" ? cancelledMessage : null);
    if (failure) fail(current.zone.index, failure);
  }, [job, errorMessage, cancelledMessage]);

  async function redo(zone: EraseZone, engine: RedoEngine, prompt?: string): Promise<void> {
    const box = zone.box;
    if (!session?.resultUrl || !box || !baseInfo || pendingRef.current) return;
    const mask = session.eraseMask;
    reset();
    setPending({ zone, engine });
    setErrors(({ [zone.index]: _previous, ...rest }) => rest);
    try {
      const tokens = await Promise.all([uploadResultImage(session.resultUrl), uploadZoneMask(mask, box)]);
      if (maskRef.current !== mask) return;
      const params = buildRedoJobParams({ engine, prompt, models, base: baseParams(session, baseInfo, defaults, tokens) });
      if (!params) throw new Error(t("editor.replace.needsSdxl"));
      submit(params);
    } catch (error) {
      if (maskRef.current === mask) fail(zone.index, messageOf(error));
    }
  }

  const progressPct = pending ? job?.progressPct ?? null : null;
  return { redo, redoingIndex: pending?.zone.index ?? null, progressPct, errors };
}
