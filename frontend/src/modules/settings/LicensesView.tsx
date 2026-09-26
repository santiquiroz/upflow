import { useQuery } from "@tanstack/react-query";
import { Scale } from "lucide-react";
import type { ReactNode } from "react";
import { useTranslation } from "../../i18n/LocaleProvider";
import {
  fetchLicenses,
  type LicensedModel,
  type LicensedPack,
  type ThirdPartyNotice,
} from "../../services/licenses";

type Translate = (key: string, params?: Record<string, string>) => string;

const LINEAGE_KEY = /D1[abc]/g;
const COMMERCIAL_USE_VALUES = new Set(["yes", "no", "unclear"]);
const REVISION_LENGTH = 12;
const WEB_URL = /^https?:\/\//i;

// Claves de la tabla de linaje de datos (spec §3.7); un valor combinado ("D1b + D1c") explica cada parte.
function lineageText(dataLineage: string, t: Translate): string {
  const keys = [...new Set(dataLineage.match(LINEAGE_KEY) ?? [])];
  if (keys.length === 0) {
    return dataLineage;
  }
  const notes = keys.map((key) => t(`licenses.lineage.${key}`)).join("; ");
  return `${dataLineage} — ${notes}`;
}

function commercialUseText(value: string, t: Translate): string {
  return COMMERCIAL_USE_VALUES.has(value) ? t(`licenses.commercial.${value}`) : value;
}

function sourceText(url: string, revision: string): string {
  const host = url.replace(WEB_URL, "").replace(/\/$/, "");
  return revision ? `${host} @ ${revision.slice(0, REVISION_LENGTH)}` : host;
}

function groupBySection(notices: ThirdPartyNotice[]): Array<[string, ThirdPartyNotice[]]> {
  const sections = [...new Set(notices.map((notice) => notice.section))];
  return sections.map((section) => [section, notices.filter((notice) => notice.section === section)]);
}

// Las URLs vienen del catalogo del backend; igual solo se enlaza lo que es web.
function ExternalLink({ href, children }: { href: string; children: string }) {
  if (!WEB_URL.test(href)) {
    return <span>{children}</span>;
  }
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-accent underline-offset-2 hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
    >
      {children}
    </a>
  );
}

function FieldRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[9rem_1fr] gap-x-3 max-[600px]:grid-cols-1">
      <dt className="text-text-faint">{label}</dt>
      <dd className="min-w-0 break-words text-text-dim">{children}</dd>
    </div>
  );
}

function LicenseText({ summary, text }: { summary: string; text: string }) {
  return (
    <details className="text-xs">
      <summary className="cursor-pointer text-text-dim hover:text-text">{summary}</summary>
      <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap rounded border border-border bg-surface-2 p-2 font-mono-tabular text-[11px] text-text-dim">
        {text}
      </pre>
    </details>
  );
}

function ModifiedList({ modifications }: { modifications: string[] }) {
  return (
    <ul className="flex flex-col gap-0.5">
      {modifications.map((change) => (
        <li key={change}>{change}</li>
      ))}
    </ul>
  );
}

function ModelLicenseCard({ model }: { model: LicensedModel }) {
  const { t } = useTranslation();
  const titleId = `license-model-${model.id}`;
  return (
    <article aria-labelledby={titleId} className="flex flex-col gap-2 rounded border border-border bg-surface-2 p-3">
      <h4 id={titleId} className="text-sm font-medium text-text">
        {model.name}
      </h4>
      <dl className="flex flex-col gap-1 text-xs">
        <FieldRow label={t("licenses.field.license")}>
          <ExternalLink href={model.licenseUrl}>{model.licenseSpdx}</ExternalLink>
        </FieldRow>
        <FieldRow label={t("licenses.field.copyright")}>{model.copyright}</FieldRow>
        <FieldRow label={t("licenses.field.attribution")}>{model.attribution}</FieldRow>
        <FieldRow label={t("licenses.field.trainingData")}>{lineageText(model.dataLineage, t)}</FieldRow>
        <FieldRow label={t("licenses.field.commercialUse")}>{commercialUseText(model.commercialUse, t)}</FieldRow>
        <FieldRow label={t("licenses.field.source")}>
          <ExternalLink href={model.sourceUrl}>{sourceText(model.sourceUrl, model.sourceRevision)}</ExternalLink>
        </FieldRow>
        {model.modifications.length > 0 && (
          <FieldRow label={t("licenses.field.modifications")}>
            <ModifiedList modifications={model.modifications} />
          </FieldRow>
        )}
      </dl>
      {model.files.map((file) => (
        <LicenseText key={file.name} summary={file.name} text={file.text} />
      ))}
    </article>
  );
}

function PackLicenses({ pack }: { pack: LicensedPack }) {
  return (
    <section className="flex flex-col gap-2">
      <h3 className="font-mono-tabular text-xs text-text-dim">{pack.pack}</h3>
      {pack.models.map((model) => (
        <ModelLicenseCard key={model.id} model={model} />
      ))}
    </section>
  );
}

function PackList({ packs }: { packs: LicensedPack[] }) {
  const { t } = useTranslation();
  return (
    <section className="flex flex-col gap-3">
      <h3 className="text-xs font-medium text-text-faint">{t("licenses.packs.title")}</h3>
      {packs.length === 0 ? (
        <p className="text-sm text-text-dim">{t("licenses.packs.empty")}</p>
      ) : (
        packs.map((pack) => <PackLicenses key={pack.pack} pack={pack} />)
      )}
    </section>
  );
}

// Los campos y el texto vienen de THIRD_PARTY_NOTICES.md, que es un documento legal en ingles: se muestran tal cual.
function NoticeCard({ notice }: { notice: ThirdPartyNotice }) {
  const { t } = useTranslation();
  const titleId = `license-notice-${notice.title.replace(/\W+/g, "-")}`;
  return (
    <article aria-labelledby={titleId} className="flex flex-col gap-2 rounded border border-border bg-surface-2 p-3">
      <h4 id={titleId} className="text-sm font-medium text-text">
        {notice.title}
      </h4>
      <dl className="flex flex-col gap-1 text-xs">
        {Object.entries(notice.fields).map(([field, values]) => (
          <FieldRow key={field} label={field}>
            {values.join("; ")}
          </FieldRow>
        ))}
      </dl>
      {notice.licenseText && <LicenseText summary={t("licenses.licenseText")} text={notice.licenseText} />}
    </article>
  );
}

function NoticeList({ notices }: { notices: ThirdPartyNotice[] }) {
  const { t } = useTranslation();
  if (notices.length === 0) {
    return null;
  }
  return (
    <section className="flex flex-col gap-3">
      <h3 className="text-xs font-medium text-text-faint">{t("licenses.thirdParty.title")}</h3>
      {groupBySection(notices).map(([section, entries]) => (
        <section key={section} className="flex flex-col gap-2">
          <h4 className="text-xs text-text-dim">{section}</h4>
          {entries.map((notice) => (
            <NoticeCard key={notice.title} notice={notice} />
          ))}
        </section>
      ))}
    </section>
  );
}

function LicensesBody() {
  const { t } = useTranslation();
  const query = useQuery({ queryKey: ["licenses"], queryFn: fetchLicenses });
  if (query.isLoading) {
    return <p className="text-sm text-text-dim">{t("licenses.loading")}</p>;
  }
  if (query.isError || !query.data) {
    return <p className="text-sm text-danger">{t("licenses.loadError")}</p>;
  }
  return (
    <>
      <PackList packs={query.data.packs} />
      <NoticeList notices={query.data.thirdParty} />
    </>
  );
}

export function LicensesView() {
  const { t } = useTranslation();
  return (
    <section aria-labelledby="licenses-title" className="flex flex-col gap-4 rounded border border-border bg-surface p-4">
      <div className="flex flex-col gap-1">
        <h2
          id="licenses-title"
          className="flex items-center gap-1.5 font-heading text-xs font-semibold uppercase tracking-wide text-text-dim"
        >
          <Scale aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
          {t("licenses.title")}
        </h2>
        <p className="text-sm text-text-dim">{t("licenses.description")}</p>
      </div>
      <LicensesBody />
    </section>
  );
}
