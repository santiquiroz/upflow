import { useQuery } from "@tanstack/react-query";
import { Info } from "lucide-react";
import { getEngineInfo, getHealth } from "../../lib/api";
import type { EngineInfoResponse, HealthResponse } from "../../lib/apiTypes";
import { DeviceDefault } from "../models/DeviceDefault";

function AvailabilityRow({ label, available }: { label: string; available: boolean }) {
  const toneClassName = available ? "text-ok" : "text-danger";
  return (
    <div className="flex items-center justify-between gap-4">
      <dt className="text-text-dim">{label}</dt>
      <dd className={`flex items-center gap-1.5 text-xs ${toneClassName}`}>
        <span aria-hidden="true" className={`h-1.5 w-1.5 rounded-full ${available ? "bg-ok" : "bg-danger"}`} />
        {available ? "Available" : "Unavailable"}
      </dd>
    </div>
  );
}

function EngineSection({ engine }: { engine: EngineInfoResponse }) {
  return (
    <div className="flex flex-col gap-3 rounded border border-border bg-surface p-4">
      <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">Engine</h2>
      <dl className="flex flex-col gap-2 text-sm">
        <div className="flex items-center justify-between gap-4">
          <dt className="text-text-dim">Engine</dt>
          <dd className="text-text">{engine.engine}</dd>
        </div>
        <div className="flex items-center justify-between gap-4">
          <dt className="text-text-dim">Default model</dt>
          <dd className="text-text">{engine.defaultModel}</dd>
        </div>
        <div className="flex items-center justify-between gap-4">
          <dt className="text-text-dim">Allowed scales</dt>
          <dd className="font-mono-tabular text-text">{engine.allowedScales.map((scale) => `${scale}x`).join(", ")}</dd>
        </div>
        <AvailabilityRow label="Engine binary" available={engine.available} />
        <AvailabilityRow label="ffmpeg" available={engine.ffmpegAvailable} />
      </dl>
    </div>
  );
}

function CapacitySection({ health }: { health: HealthResponse }) {
  return (
    <div className="flex flex-col gap-3 rounded border border-border bg-surface p-4">
      <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">Capacity</h2>
      <dl className="flex flex-col gap-2 text-sm">
        <div className="flex items-center justify-between gap-4">
          <dt className="text-text-dim">GPU concurrency</dt>
          <dd className="font-mono-tabular text-text">{health.gpuConcurrency}</dd>
        </div>
        <div className="flex items-center justify-between gap-4">
          <dt className="text-text-dim">Image queue depth</dt>
          <dd className="font-mono-tabular text-text">{health.queueDepth}</dd>
        </div>
        <div className="flex items-center justify-between gap-4">
          <dt className="text-text-dim">Video queue depth</dt>
          <dd className="font-mono-tabular text-text">{health.videoQueueDepth}</dd>
        </div>
      </dl>
    </div>
  );
}

function EnvExplanationNote() {
  return (
    <div className="flex items-start gap-2 rounded border border-border bg-surface-2 px-3 py-2 text-xs text-text-faint">
      <Info aria-hidden="true" className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
      <span>
        These values come from the app&apos;s .env configuration, set at install time. There is no settings-write API
        yet, so nothing on this page is editable from the UI — including audio enhance, frame interpolation, output
        retention (TTL) and upload size limits, which are configured but not exposed by the API.
      </span>
    </div>
  );
}

function EngineSectionStatus({ query }: { query: ReturnType<typeof useQuery<EngineInfoResponse>> }) {
  if (query.isLoading) {
    return <p className="text-sm text-text-dim">Loading engine info…</p>;
  }
  if (query.isError) {
    return <p className="text-sm text-danger">Could not load engine info.</p>;
  }
  if (!query.data) {
    return null;
  }
  return <EngineSection engine={query.data} />;
}

function CapacitySectionStatus({ query }: { query: ReturnType<typeof useQuery<HealthResponse>> }) {
  if (query.isLoading) {
    return <p className="text-sm text-text-dim">Loading capacity info…</p>;
  }
  if (query.isError) {
    return <p className="text-sm text-danger">Could not load capacity info.</p>;
  }
  if (!query.data) {
    return null;
  }
  return <CapacitySection health={query.data} />;
}

export function SettingsPage() {
  const engineQuery = useQuery({ queryKey: ["engine"], queryFn: getEngineInfo });
  const healthQuery = useQuery({ queryKey: ["health"], queryFn: getHealth });

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="font-heading text-2xl font-semibold text-text">Settings</h1>
        <p className="mt-1 text-sm text-text-dim">Current engine, device and capacity configuration.</p>
      </div>
      <EnvExplanationNote />
      <div className="grid grid-cols-2 gap-4 max-[900px]:grid-cols-1">
        <EngineSectionStatus query={engineQuery} />
        <CapacitySectionStatus query={healthQuery} />
        <DeviceDefault />
      </div>
    </div>
  );
}
