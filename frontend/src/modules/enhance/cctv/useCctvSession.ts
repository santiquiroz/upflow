import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { DEFAULT_POLL_INTERVAL_MS } from "../../../hooks/useVideoJob";
import {
  analyzeCctv,
  getCctvAnalysis,
  type CctvAnalysis,
  type CctvAnalysisJob,
  type CctvAnalyzeReply,
} from "../../../services/cctv";
import { errorInfoOf, type CctvErrorInfo } from "./cctvText";

export type CctvSessionPhase = "idle" | "uploading" | "analyzing" | "ready" | "failed";

export interface CctvSession {
  phase: CctvSessionPhase;
  uploadPercent: number | null;
  analysis: CctvAnalysis | null;
  error: CctvErrorInfo | null;
  analyze: (file: File) => void;
  reset: () => void;
}

const UPLOAD_COMPLETE = 100;

export function resolveAnalysis(reply: CctvAnalyzeReply | undefined, status: CctvAnalysisJob | undefined): CctvAnalysis | null {
  if (reply?.kind === "done") {
    return reply.analysis;
  }
  return status?.status === "completed" ? status.result : null;
}

export function analysisJobError(status: CctvAnalysisJob | undefined): CctvErrorInfo | null {
  if (status?.status !== "failed") {
    return null;
  }
  return { key: status.errorKey, message: status.error ?? "" };
}

interface PhaseInputs {
  isUploading: boolean;
  uploadPercent: number | null;
  error: CctvErrorInfo | null;
  analysis: CctvAnalysis | null;
  waitingForAnalysis: boolean;
}

export function sessionPhase(inputs: PhaseInputs): CctvSessionPhase {
  if (inputs.isUploading) {
    // Con el archivo entero arriba, el backend sigue: hash, remux, indice y diagnostico.
    return inputs.uploadPercent === UPLOAD_COMPLETE ? "analyzing" : "uploading";
  }
  if (inputs.error) {
    return "failed";
  }
  if (inputs.analysis) {
    return "ready";
  }
  return inputs.waitingForAnalysis ? "analyzing" : "idle";
}

function isStillRunning(status: CctvAnalysisJob | undefined): boolean {
  return status === undefined || status.status === "running";
}

export function useCctvSession(pollIntervalMs: number = DEFAULT_POLL_INTERVAL_MS): CctvSession {
  const [uploadPercent, setUploadPercent] = useState<number | null>(null);
  const upload = useMutation({
    mutationFn: (file: File) => analyzeCctv(file, { onProgress: setUploadPercent }),
  });
  const pendingId = upload.data?.kind === "pending" ? upload.data.analysisJobId : null;
  const statusQuery = useQuery({
    queryKey: ["cctvAnalysis", pendingId],
    queryFn: () => getCctvAnalysis(pendingId as string),
    enabled: pendingId !== null,
    refetchInterval: (query) => (isStillRunning(query.state.data) ? pollIntervalMs : false),
  });

  const analysis = resolveAnalysis(upload.data, statusQuery.data);
  const error = errorInfoOf(upload.error) ?? errorInfoOf(statusQuery.error) ?? analysisJobError(statusQuery.data);

  function analyze(file: File): void {
    setUploadPercent(null);
    upload.mutate(file);
  }

  function reset(): void {
    setUploadPercent(null);
    upload.reset();
  }

  return {
    phase: sessionPhase({
      isUploading: upload.isPending,
      uploadPercent,
      error,
      analysis,
      waitingForAnalysis: pendingId !== null,
    }),
    uploadPercent,
    analysis,
    error,
    analyze,
    reset,
  };
}
