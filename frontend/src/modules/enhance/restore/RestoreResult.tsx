import { Download, FileJson, Info, PencilLine } from "lucide-react";
import { useState, type ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { CompareSlider } from "../../../components/CompareSlider";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { JobResponse } from "../../../lib/apiTypes";
import { editorHandoffStore, type EditorHandoffStore } from "../../../lib/editorHandoffStore";
import type { RecomposeFaceChoice, RecomposeResponse } from "../../../lib/restoreApiTypes";
import { recomposeFaces, restoreArtifactUrl, type CreateRestoreJobParams } from "../../../services/restore";
import { FaceResultGrid } from "./FaceResultGrid";
import { canBatch } from "./restoreBatch";
import { RestoreBatchSection } from "./RestoreBatchSection";
import {
  editorSource,
  hasUncolored,
  showsInfoCard,
  withRecomposedSidecar,
  type RestoreResultSummary,
} from "./restoreResultModel";
import { revisedUrl } from "./restoreUrls";
import type { RestoreBatchDeps } from "./useRestoreBatch";

export interface RestoreResultProps {
  job: JobResponse;
  summary: RestoreResultSummary;
  beforeUrl: string;
  handoffStore?: EditorHandoffStore;
  recompose?: (jobId: string, faces: Record<number, RecomposeFaceChoice>) => Promise<RecomposeResponse>;
  // Lo que se pidio para esta foto: el lote lo repite en las demas.
  batchBase?: CreateRestoreJobParams | null;
  batchDeps?: RestoreBatchDeps;
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

interface RestoreDownloadsProps {
  job: JobResponse;
  summary: RestoreResultSummary;
  revision: number;
}

function RestoreDownloads({ job, summary, revision }: RestoreDownloadsProps) {
  const { t } = useTranslation();
  const names = summary.downloadNames;
  const artifactUrl = (name: "uncolored" | "beforeafter" | "sidecar") =>
    revisedUrl(restoreArtifactUrl(job.jobId, name), revision);
  return (
    <div className="flex flex-wrap gap-2">
      {job.downloadUrl && (
        <DownloadLink href={revisedUrl(job.downloadUrl, revision)} fileName={names.restored} icon={DOWNLOAD_ICON}>
          {t("restore.result.download.restored")}
        </DownloadLink>
      )}
      {hasUncolored(summary) && (
        <DownloadLink href={artifactUrl("uncolored")} fileName={names.uncolored} icon={DOWNLOAD_ICON}>
          {t("restore.result.download.uncolored")}
        </DownloadLink>
      )}
      <DownloadLink href={artifactUrl("beforeafter")} fileName={names.beforeAfter} icon={DOWNLOAD_ICON}>
        {t("restore.result.download.beforeAfter")}
      </DownloadLink>
      <DownloadLink
        href={artifactUrl("sidecar")}
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

export function RestoreResult({
  job,
  summary: initialSummary,
  beforeUrl,
  handoffStore = editorHandoffStore,
  recompose = recomposeFaces,
  batchBase = null,
  batchDeps,
}: RestoreResultProps) {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const [showUncolored, setShowUncolored] = useState(false);
  const [summary, setSummary] = useState(initialSummary);
  const [revision, setRevision] = useState(0);
  const uncoloredAvailable = hasUncolored(summary);
  const afterArtifact = uncoloredAvailable && showUncolored ? "uncolored" : "view";

  function openInEditor() {
    const source = editorSource(job, summary);
    handoffStore.offer({ ...source, url: revisedUrl(source.url, revision) });
    navigate(EDITOR_PATH);
  }

  function handleRecomposed(sidecar: Record<string, unknown>) {
    setSummary((current) => withRecomposedSidecar(current, sidecar));
    setRevision((current) => current + 1);
  }

  return (
    <section className="flex flex-col gap-3" aria-labelledby="restore-result-title">
      <h3 id="restore-result-title" className="font-heading text-sm font-semibold text-text">
        {t("restore.result.title")}
      </h3>
      <CompareSlider
        beforeSrc={beforeUrl}
        afterSrc={revisedUrl(restoreArtifactUrl(job.jobId, afterArtifact), revision)}
        beforeAlt={t("restore.result.beforeAlt", { name: job.originalFilename })}
        afterAlt={t("restore.result.afterAlt", { name: job.originalFilename })}
        fullResolution={summary.viewFullResolution}
      />
      {uncoloredAvailable && <UncoloredToggle checked={showUncolored} onChange={setShowUncolored} />}
      {summary.faces.length > 0 && (
        <FaceResultGrid
          jobId={job.jobId}
          faces={summary.faces}
          recomposeAvailable={summary.recomposeAvailable}
          onRecomposed={handleRecomposed}
          recompose={recompose}
        />
      )}
      {showsInfoCard(summary) && <InfoCard />}
      <RestoreDownloads job={job} summary={summary} revision={revision} />
      <OpenInEditorButton onOpen={openInEditor} />
      {batchBase && canBatch(batchBase) && <RestoreBatchSection base={batchBase} deps={batchDeps} />}
      <p className="text-xs text-text-faint">{t("restore.local")}</p>
    </section>
  );
}
