import {
  CheckCircle2,
  Download,
  Map,
  PackageSearch,
} from "lucide-react";
import { useState } from "react";
import { useCapabilityTree } from "../../hooks/useCapabilityTree";
import { useTranslation } from "../../i18n/LocaleProvider";
import type {
  CapabilityDomainId,
  CapabilityResponse,
} from "../../lib/apiTypes";

interface CapabilityTreeProps {
  onSelect: (capability: CapabilityResponse) => void;
  onProvision: (capability: CapabilityResponse) => void;
}

function AvailableCapability({
  capability,
  onSelect,
}: {
  capability: CapabilityResponse;
  onSelect: (capability: CapabilityResponse) => void;
}) {
  const { t } = useTranslation();

  return (
    <li data-status={capability.status}>
      <button
        type="button"
        onClick={() => onSelect(capability)}
        className="flex w-full items-center gap-3 rounded border border-ok bg-surface-2 px-3 py-3 text-left text-sm font-medium text-text transition-[background-color,border-color] duration-fast hover:bg-surface focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        <CheckCircle2
          aria-hidden="true"
          className="h-4 w-4 shrink-0 text-ok"
          strokeWidth={2}
        />
        <span>{t(capability.labelKey)}</span>
      </button>
    </li>
  );
}

function SetupCapability({
  capability,
  onProvision,
}: {
  capability: CapabilityResponse;
  onProvision: (capability: CapabilityResponse) => void;
}) {
  const { t } = useTranslation();
  const label = t(capability.labelKey);

  return (
    <li
      data-status={capability.status}
      className="flex flex-col gap-2 rounded border border-warn bg-surface-2 px-3 py-3"
    >
      <div className="flex items-center gap-3">
        <PackageSearch
          aria-hidden="true"
          className="h-4 w-4 shrink-0 text-warn"
          strokeWidth={2}
        />
        <h3 className="text-sm font-medium text-text">{label}</h3>
      </div>
      {capability.setupReasonKey && (
        <p className="pl-7 text-xs leading-relaxed text-text-dim">
          {t(capability.setupReasonKey)}
        </p>
      )}
      {capability.missingPacks.length > 0 && (
        <button
          type="button"
          aria-label={`${t("capability.tree.download")}: ${label}`}
          onClick={() => onProvision(capability)}
          className="ml-7 inline-flex w-fit items-center gap-1.5 rounded-sm border border-warn bg-surface px-2.5 py-1.5 text-xs font-medium text-text transition-[background-color,border-color] duration-fast hover:bg-surface-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          <Download aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={2} />
          {t("capability.tree.download")}
        </button>
      )}
    </li>
  );
}

function RoadmapCapability({ capability }: { capability: CapabilityResponse }) {
  const { t } = useTranslation();

  return (
    <li
      data-status={capability.status}
      className="flex flex-col gap-2 rounded border border-border bg-surface px-3 py-3"
    >
      <div className="flex items-center gap-3">
        <Map
          aria-hidden="true"
          className="h-4 w-4 shrink-0 text-text-faint"
          strokeWidth={2}
        />
        <h4 className="text-sm font-medium text-text-dim">
          {t(capability.labelKey)}
        </h4>
      </div>
      {capability.unavailableReasonKey && (
        <p className="pl-7 text-xs leading-relaxed text-text-faint">
          {t(capability.unavailableReasonKey)}
        </p>
      )}
    </li>
  );
}

export function CapabilityTree({
  onSelect,
  onProvision,
}: CapabilityTreeProps) {
  const { t } = useTranslation();
  const query = useCapabilityTree();
  const [selectedDomain, setSelectedDomain] =
    useState<CapabilityDomainId | null>(null);

  if (query.isLoading) {
    return (
      <p role="status" className="text-sm text-text-dim">
        {t("capability.tree.loading")}
      </p>
    );
  }

  if (query.isError || !query.data) {
    return (
      <p role="alert" className="text-sm text-danger">
        {t("capability.tree.loadFailed")}
      </p>
    );
  }

  const domain =
    query.data.domains.find((item) => item.domain === selectedDomain) ??
    query.data.domains[0];

  if (!domain) {
    return (
      <p role="alert" className="text-sm text-danger">
        {t("capability.tree.loadFailed")}
      </p>
    );
  }

  const domainHeadingId = `capability-domain-${domain.domain}`;
  const roadmapHeadingId = `${domainHeadingId}-roadmap`;

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {query.data.domains.map((item) => {
          const isSelected = item.domain === domain.domain;
          return (
            <button
              key={item.domain}
              type="button"
              aria-pressed={isSelected}
              onClick={() => setSelectedDomain(item.domain)}
              className={`rounded border px-3 py-2 text-sm font-medium transition-[background-color,border-color,color] duration-fast focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent ${
                isSelected
                  ? "border-accent bg-surface-2 text-accent"
                  : "border-border bg-surface text-text-dim hover:border-text-faint hover:text-text"
              }`}
            >
              {t(item.labelKey)}
            </button>
          );
        })}
      </div>

      <section aria-labelledby={domainHeadingId} className="flex flex-col gap-3">
        <h2
          id={domainHeadingId}
          className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim"
        >
          {t(domain.labelKey)}
        </h2>
        <ul data-testid="live-capabilities" className="flex flex-col gap-2">
          {domain.capabilities.map((capability) =>
            capability.status === "available" ? (
              <AvailableCapability
                key={capability.id}
                capability={capability}
                onSelect={onSelect}
              />
            ) : (
              <SetupCapability
                key={capability.id}
                capability={capability}
                onProvision={onProvision}
              />
            ),
          )}
        </ul>

        {domain.roadmap.length > 0 && (
          <section
            aria-labelledby={roadmapHeadingId}
            className="mt-2 flex flex-col gap-2 border-t border-border pt-4"
          >
            <h3
              id={roadmapHeadingId}
              className="font-heading text-xs font-semibold uppercase tracking-wide text-text-faint"
            >
              {t("capability.tree.roadmap")}
            </h3>
            <ul data-testid="roadmap-capabilities" className="flex flex-col gap-2">
              {domain.roadmap.map((capability) => (
                <RoadmapCapability
                  key={capability.id}
                  capability={capability}
                />
              ))}
            </ul>
          </section>
        )}
      </section>
    </div>
  );
}
