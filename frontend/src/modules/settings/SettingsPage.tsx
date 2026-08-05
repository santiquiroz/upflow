import { useQuery } from "@tanstack/react-query";
import { useTranslation } from "../../i18n/LocaleProvider";
import { Info } from "lucide-react";
import { getEngineInfo, getHealth } from "../../lib/api";
import type { EngineInfoResponse, HealthResponse } from "../../lib/apiTypes";
import { DeviceDefault } from "../models/DeviceDefault";
import { DeviceAcceleration } from "./DeviceAcceleration";
import { EditableSettingsSection } from "./EditableSettingsSection";
import { VideoUploadLimitSection } from "./VideoUploadLimitSection";
import { LanguageSection } from "./LanguageSection";
import { OptimizationCenter } from "./OptimizationCenter";

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
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-3 rounded border border-border bg-surface p-4">
      <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">{t("settings.section.engine")}</h2>
      <dl className="flex flex-col gap-2 text-sm">
        <div className="flex items-center justify-between gap-4">
          <dt className="text-text-dim">{t("settings.engine.name")}</dt>
          <dd className="text-text">{engine.engine}</dd>
        </div>
        <div className="flex items-center justify-between gap-4">
          <dt className="text-text-dim">{t("settings.engine.defaultModel")}</dt>
          <dd className="text-text">{engine.defaultModel}</dd>
        </div>
        <div className="flex items-center justify-between gap-4">
          <dt className="text-text-dim">{t("settings.engine.allowedScales")}</dt>
          <dd className="font-mono-tabular text-text">{engine.allowedScales.map((scale) => `${scale}x`).join(", ")}</dd>
        </div>
        <AvailabilityRow label={t("settings.engine.binary")} available={engine.available} />
        <AvailabilityRow label="ffmpeg" available={engine.ffmpegAvailable} />
      </dl>
    </div>
  );
}

function CapacitySection({ health }: { health: HealthResponse }) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-3 rounded border border-border bg-surface p-4">
      <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">{t("settings.section.capacity")}</h2>
      <dl className="flex flex-col gap-2 text-sm">
        <div className="flex items-center justify-between gap-4">
          <dt className="text-text-dim">{t("settings.engine.gpuConcurrency")}</dt>
          <dd className="font-mono-tabular text-text">{health.gpuConcurrency}</dd>
        </div>
        <div className="flex items-center justify-between gap-4">
          <dt className="text-text-dim">{t("settings.engine.imageQueueDepth")}</dt>
          <dd className="font-mono-tabular text-text">{health.queueDepth}</dd>
        </div>
        <div className="flex items-center justify-between gap-4">
          <dt className="text-text-dim">{t("settings.engine.videoQueueDepth")}</dt>
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
        Most values come from the app&apos;s .env configuration, set at install time, and are read-only here; the
        Credentials section below is editable. Audio enhance, frame interpolation, output retention (TTL), upload
        size limits, and auto-routing queued jobs to a free GPU (ENABLE_AUTO_ROUTE) are configured but not exposed by
        the settings API. Pick &quot;Auto&quot; in a job&apos;s Device section to opt into auto-routing per job
        regardless of this setting.
      </span>
    </div>
  );
}

function EngineSectionStatus({ query }: { query: ReturnType<typeof useQuery<EngineInfoResponse>> }) {
  const { t } = useTranslation();
  if (query.isLoading) {
    return <p className="text-sm text-text-dim">{t("settings.engine.loading")}</p>;
  }
  if (query.isError) {
    return <p className="text-sm text-danger">{t("settings.engine.loadError")}</p>;
  }
  if (!query.data) {
    return null;
  }
  return <EngineSection engine={query.data} />;
}

function CapacitySectionStatus({ query }: { query: ReturnType<typeof useQuery<HealthResponse>> }) {
  const { t } = useTranslation();
  if (query.isLoading) {
    return <p className="text-sm text-text-dim">{t("settings.capacity.loading")}</p>;
  }
  if (query.isError) {
    return <p className="text-sm text-danger">{t("settings.capacity.loadError")}</p>;
  }
  if (!query.data) {
    return null;
  }
  return <CapacitySection health={query.data} />;
}

export function SettingsPage() {
  const { t } = useTranslation();
  const engineQuery = useQuery({ queryKey: ["engine"], queryFn: getEngineInfo });
  const healthQuery = useQuery({ queryKey: ["health"], queryFn: getHealth });

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="font-heading text-2xl font-semibold text-text">Settings</h1>
        <p className="mt-1 text-sm text-text-dim">{t("settings.page.subtitle")}</p>
      </div>
      <EnvExplanationNote />
      <div className="grid grid-cols-2 gap-4 max-[900px]:grid-cols-1">
        <EngineSectionStatus query={engineQuery} />
        <CapacitySectionStatus query={healthQuery} />
        <EditableSettingsSection />
        {engineQuery.data && (
          <VideoUploadLimitSection currentMb={engineQuery.data.maxVideoUploadMb} />
        )}
        <DeviceDefault />
        <DeviceAcceleration />
        <LanguageSection />
      </div>
      <OptimizationCenter />
    </div>
  );
}
