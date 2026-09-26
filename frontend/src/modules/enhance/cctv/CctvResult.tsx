import { useMutation } from "@tanstack/react-query";
import { Clock, Download, ExternalLink, ListChecks } from "lucide-react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { CctvJobSummary } from "../../../lib/apiTypes";
import { verifyCctvFiles, type CctvVerifyResult } from "../../../services/cctv";
import { artifactUrl, PACKAGE_ARTIFACT, REPORT_ARTIFACT, resultFiles, verifyOutcome, type ResultFile } from "./cctvResultFiles";
import { errorInfoOf, errorText, translateOr } from "./cctvText";

interface CctvResultProps {
  jobId: string;
  summary: CctvJobSummary;
  retentionHours: number | null;
}

const ACTION_CLASS =
  "inline-flex w-fit items-center gap-2 rounded border border-border bg-surface px-3 py-1.5 text-sm text-text transition-[background-color,border-color] duration-fast hover:border-accent hover:bg-surface-2 active:bg-surface-2 disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";
const PRIMARY_ACTION_CLASS =
  "inline-flex w-fit items-center gap-2 rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg transition-[background-color] duration-fast hover:bg-accent-hover active:bg-accent-press focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";
const ICON_CLASS = "h-4 w-4 shrink-0";
const VERIFY_HINT_ID = "cctv-verify-hint";

function SourceHash({ sha256 }: { sha256: string | null }) {
  const { t } = useTranslation();
  if (!sha256) {
    return null;
  }
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs text-text-dim">{t("cctv.result.sourceHash")}</span>
      <code className="break-all font-mono-tabular text-xs text-text">{sha256}</code>
    </div>
  );
}

function ReportLink({ url }: { url: string | null }) {
  const { t } = useTranslation();
  if (!url) {
    return null;
  }
  return (
    <a href={url} target="_blank" rel="noopener noreferrer" className={ACTION_CLASS}>
      <ExternalLink aria-hidden="true" className={ICON_CLASS} strokeWidth={1.75} />
      {t("cctv.report.open")}
    </a>
  );
}

function PackageLink({ url }: { url: string | null }) {
  const { t } = useTranslation();
  if (!url) {
    return null;
  }
  return (
    <a href={url} download className={PRIMARY_ACTION_CLASS}>
      <Download aria-hidden="true" className={ICON_CLASS} strokeWidth={1.75} />
      {t("cctv.package.download")}
    </a>
  );
}

function ChangedFiles({ mismatches, missing }: { mismatches: string[]; missing: string[] }) {
  const { t } = useTranslation();
  return (
    <div role="alert" className="flex flex-col gap-1 text-xs text-danger">
      <p>{t("cctv.verify.changed")}</p>
      <ul className="flex flex-col gap-0.5 pl-4">
        {mismatches.map((path) => (
          <li key={`changed-${path}`} className="break-all font-mono-tabular">
            {t("cctv.verify.mismatch", { path })}
          </li>
        ))}
        {missing.map((path) => (
          <li key={`missing-${path}`} className="break-all font-mono-tabular">
            {t("cctv.verify.missing", { path })}
          </li>
        ))}
      </ul>
    </div>
  );
}

function VerifyResultText({ result }: { result: CctvVerifyResult }) {
  const { t } = useTranslation();
  const outcome = verifyOutcome(result);
  if (outcome.kind === "changed") {
    return <ChangedFiles mismatches={outcome.mismatches} missing={outcome.missing} />;
  }
  return (
    <p role="status" className="text-xs text-ok">
      {t("cctv.verify.ok")} {t("cctv.verify.checked", { count: outcome.checked })}
    </p>
  );
}

function VerifyFiles({ jobId }: { jobId: string }) {
  const { t } = useTranslation();
  const verify = useMutation({ mutationFn: () => verifyCctvFiles(jobId) });
  const error = errorInfoOf(verify.error);
  return (
    <div className="flex flex-col gap-1">
      <button
        type="button"
        onClick={() => verify.mutate()}
        disabled={verify.isPending}
        title={t("cctv.verify.tooltip")}
        aria-describedby={VERIFY_HINT_ID}
        className={ACTION_CLASS}
      >
        <ListChecks aria-hidden="true" className={ICON_CLASS} strokeWidth={1.75} />
        {t("cctv.verify.action")}
      </button>
      <p id={VERIFY_HINT_ID} className="text-xs text-text-faint">
        {t("cctv.verify.tooltip")}
      </p>
      {verify.isPending && <p role="status" className="text-xs text-text-dim">{t("cctv.verify.checking")}</p>}
      {error && <p role="alert" className="text-xs text-danger">{errorText(t, error)}</p>}
      {verify.data && <VerifyResultText result={verify.data} />}
    </div>
  );
}

function FileRow({ file }: { file: ResultFile }) {
  const { t } = useTranslation();
  const label = file.labelKey ? t(file.labelKey, file.params) : file.name;
  return (
    <li>
      <a href={file.url} download className="text-sm text-accent underline-offset-2 hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent">
        {label}
      </a>
    </li>
  );
}

function FileList({ files }: { files: ResultFile[] }) {
  const { t } = useTranslation();
  if (files.length === 0) {
    return null;
  }
  return (
    <div className="flex flex-col gap-1">
      <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">{t("cctv.result.files")}</span>
      <ul aria-label={t("cctv.result.files")} className="flex flex-col gap-1">
        {files.map((file) => (
          <FileRow key={file.name} file={file} />
        ))}
      </ul>
    </div>
  );
}

function ResultWarnings({ warnings }: { warnings: string[] }) {
  const { t } = useTranslation();
  if (warnings.length === 0) {
    return null;
  }
  return (
    <ul aria-label={t("cctv.result.warnings")} className="flex flex-col gap-1 text-xs text-warn">
      {warnings.map((key) => (
        <li key={key}>{translateOr(t, key, key)}</li>
      ))}
    </ul>
  );
}

function RetentionNotice({ hours }: { hours: number | null }) {
  const { t } = useTranslation();
  if (hours === null) {
    return null;
  }
  return (
    <p role="note" className="flex items-start gap-2 text-xs text-text-dim">
      <Clock aria-hidden="true" className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
      {t("cctv.retention", { hours })}
    </p>
  );
}

export function CctvResult({ jobId, summary, retentionHours }: CctvResultProps) {
  const { t } = useTranslation();
  const artifacts = summary.artifacts;
  return (
    <section aria-labelledby="cctv-result-title" className="flex flex-col gap-4 rounded border border-border bg-surface p-4">
      <h3 id="cctv-result-title" className="font-heading text-sm font-semibold text-text">
        {t("cctv.result.title")}
      </h3>
      <SourceHash sha256={summary.sourceSha256} />
      <div className="flex flex-wrap items-start gap-3">
        <PackageLink url={artifactUrl(artifacts, PACKAGE_ARTIFACT)} />
        <ReportLink url={artifactUrl(artifacts, REPORT_ARTIFACT)} />
      </div>
      {summary.verifyUrl && <VerifyFiles jobId={jobId} />}
      <FileList files={resultFiles(artifacts)} />
      <ResultWarnings warnings={summary.warnings} />
      <RetentionNotice hours={retentionHours} />
      <p className="text-xs text-text-faint">{t("cctv.disclaimer")}</p>
    </section>
  );
}
