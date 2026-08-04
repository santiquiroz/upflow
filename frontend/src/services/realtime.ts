import { apiGet, apiPostJson } from "../lib/api";
import type { RealtimeCapabilities, RealtimeStarted } from "../lib/apiTypes";

export function fetchRealtimeCapabilities(): Promise<RealtimeCapabilities> {
  return apiGet<RealtimeCapabilities>("/realtime/capabilities");
}

export function startRealtime(preset: string, maxFrameRate: number | null): Promise<RealtimeStarted> {
  const body: Record<string, unknown> = { preset };
  if (maxFrameRate !== null) body.maxFrameRate = maxFrameRate;
  return apiPostJson<RealtimeStarted>("/realtime/start", body);
}
