import { Download, FileJson, Info, PencilLine } from "lucide-react";
import { useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { CompareSlider } from "../../../components/CompareSlider";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { JobResponse } from "../../../lib/apiTypes";
import { editorHandoffStore, type EditorHandoffStore } from "../../../lib/editorHandoffStore";
import { restoreArtifactUrl } from "../../../services/restore";
import {
  editorSource,
  hasUncolored,
  showsInfoCard,
  type RestoreResultSummary,
} from "./restoreResultModel";

export interface RestoreResultProps {
  job: JobResponse;
  summary: RestoreResultSummary;
  beforeUrl: string;
  handoffStore?: EditorHandoffStore;
}

const EDITOR_PATH = "/editor";
const DOWNLOAD_LINK_CLASS =
  "inline-flex items-center gap-1.5 rounded border border-border px-3 py-1.5 text-xs text-text transition hover:border-accent hover:text-accent focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";

function DownloadLink({ href, fileName, icon, children }: { href: string; fileName: string; icon: ReactNode; children: ReactNode }) {
  return (
    <a href={href} download={fileName} className={DOWNLOAD_LINK_CLASS}>
      {icon}
      {children}
    </a>
  );
}

const DOWNLOAD_ICON = <Download aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />;

function RestoreDownloads({ job, summary }: { job: JobResponse; summary: RestoreResultSummary }) {
  const { t } = useTranslation();
  const names = summary.downloadNames;
  return (
    <div className="flex flex-wrap gap-2">
      {job.downloadUrl && (
        <DownloadLink href={job.downloadUrl} fileName={names.restored} icon={DOWNLOAD_ICON}>
          {t("restore.result.download.restored")}
        </DownloadLink>
      )}
      {hasUncolored(summary) && (
        <DownloadLink href={restoreArtifactUrl(job.jobId, "uncolored")} fileName={names.uncolored} icon={DOWNLOAD_ICON}>
          {t("restore.result.download.uncolored")}
        </DownloadLink>
      )}
      <DownloadLink href={restoreArtifactUrl(job.jobId, "beforeafter")} fileName={names.beforeAfter} icon={DOWNLOAD_ICON}>
        {t("restore.result.download.beforeAfter")}
      </DownloadLink>
      <DownloadLink
        href={restoreArtifactUrl(job.jobId, "sidecar")}
        fileName={names.sidecar}
        icon={<FileJson aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />}
      >
        {t("restore.result.download.details")}
      </DownloadLink>
    </div>
  );
}

function OpenInEditorButton({ onOpen }: { onOpen: () => void }) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-wrap items-center gap-2">
      <button
        type="button"
        onClick={onOpen}
        className="inline-flex items-center gap-1.5 rounded border border-accent px-3 py-1.5 text-xs font-medium text-accent transition hover:bg-accent hover:text-bg focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        <PencilLine aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
        {t("restore.result.openInEditor")}
      </button>
      <span className="text-xs text-text-dim">{t("restore.result.openInEditor.hint")}</span>
    </div>
  );
}

function InfoCard() {
  const { t } = useTranslation();
  return (
    <p role="note" className="flex items-start gap-2 rounded border border-border bg-surface-2 p-3 text-xs text-text-dim">
      <Info aria-hidden="true" className="mt-0.5 h-3.5 w-3.5 shrink-0 text-accent" strokeWidth={1.75} />
      <span>{t("restore.result.infoCard")}</span>
    </p>
  );
}

function UncoloredToggle({ checked, onChange }: { checked: boolean; onChange: (checked: boolean) => void }) {
  const { t } = useTranslation();
  return (
    <label className="inline-flex items-center gap-2 text-xs text-text">
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="h-3.5 w-3.5 accent-accent"
      />
      {t("restore.result.showUncolored")}
    </label>
  );
}

export function RestoreResult({ job, summary, beforeUrl, handoffStore = editorHandoffStore }: RestoreResultProps) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [showUncolored, setShowUncolored] = useState(false);
  const uncoloredAvailable = hasUncolored(summary);
  const afterArtifact = uncoloredAvailable && showUncolored ? "uncolored" : "view";

  function openInEditor() {
    handoffStore.offer(editorSource(job, summary));
    navigate(EDITOR_PATH);
  }

  return (
    <section className="flex flex-col gap-3" aria-labelledby="restore-result-title">
      <h3 id="restore-result-title" className="font-heading text-sm font-semibold text-text">
        {t("restore.result.title")}
      </h3>
      <CompareSlider
        beforeSrc={beforeUrl}
        afterSrc={restoreArtifactUrl(job.jobId, afterArtifact)}
        beforeAlt={t("restore.result.beforeAlt", { name: job.originalFilename })}
        afterAlt={t("restore.result.afterAlt", { name: job.originalFilename })}
        fullResolution={summary.viewFullResolution}
      />
      {uncoloredAvailable && <UncoloredToggle checked={showUncolored} onChange={setShowUncolored} />}
      {showsInfoCard(summary) && <InfoCard />}
      <RestoreDownloads job={job} summary={summary} />
      <OpenInEditorButton onOpen={openInEditor} />
      <p className="text-xs text-text-faint">{t("restore.local")}</p>
    </section>
  );
}
