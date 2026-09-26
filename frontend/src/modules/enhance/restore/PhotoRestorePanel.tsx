import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { FileDropzone } from "../../../components/FileDropzone";
import { JobCard } from "../../../components/JobCard";
import { useTranslation } from "../../../i18n/LocaleProvider";
import { getEngineInfo } from "../../../lib/api";
import type { RestoreAnalysis, RestoreCapabilities, RestoreFace } from "../../../lib/restoreApiTypes";
import { getRestoreCapabilities } from "../../../services/restore";
import { exceedsUploadLimit, formatMegabytes } from "../uploadLimit";
import { FaceGrid } from "./FaceGrid";
import { stepBlendOf, withFaceChoices } from "./faceSelection";
import { GeometryTools } from "./GeometryTools";
import { RestoreResult } from "./RestoreResult";
import { readRestoreSummary } from "./restoreResultModel";
import { RestoreSummary } from "./RestoreSummary";
import type { FacePatch, RestoreSessionState } from "./restoreSessionState";
import { versionedUrl } from "./restoreUrls";
import { useRestoreJob, type RestoreJobRequest, type UseRestoreJobResult } from "./useRestoreJob";
import { useRestoreSelection, type RestoreSelection } from "./useRestoreSelection";
import { useRestoreSession } from "./useRestoreSession";

const PHOTO_ACCEPT = ".png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff,image/png,image/jpeg,image/webp,image/bmp,image/tiff";
const PHOTO_FORMATS = "PNG, JPG, WEBP, BMP, TIFF";
const RESTORE_CAPABILITIES_KEY = ["restore", "capabilities"];
const RESTORE_OUTPUT_FORMAT = "png";
const FACES_STEP = "faces";

function useStatusText(session: RestoreSessionState): string | null {
  const { t } = useTranslation();
  if (session.phase === "updating") {
    return t("restore.updating");
  }
  if (session.phase !== "analyzing") {
    return null;
  }
  if (session.uploadPercent !== null && session.uploadPercent < 100) {
    return t("restore.uploading", { percent: Math.round(session.uploadPercent) });
  }
  return t("restore.analyzing");
}

function PhotoFacts({ analysis }: { analysis: RestoreAnalysis }) {
  const { t } = useTranslation();
  return (
    <p className="font-mono-tabular text-xs text-text-dim">
      {t("restore.photo.size", { width: analysis.width, height: analysis.height, bits: analysis.bitDepth })}
    </p>
  );
}

function isJobActive(phase: UseRestoreJobResult["phase"]): boolean {
  return phase === "queued" || phase === "running";
}

// Sin reescalado todavia (llega con el selector de escala): la cadena corre a 1x.
function restoreJobRequest(
  analysis: RestoreAnalysis,
  selection: RestoreSelection,
  faces: RestoreFace[],
): RestoreJobRequest {
  return {
    params: {
      source: { token: analysis.token },
      steps: selection.enabledIds,
      options: withFaceChoices(selection.requestOptions, faces, analysis.faces),
      scale: 1,
      modelId: null,
      device: null,
      outputFormat: RESTORE_OUTPUT_FORMAT,
    },
    fileName: analysis.originalName,
  };
}

interface RestoreControlsProps {
  analysis: RestoreAnalysis;
  faces: RestoreFace[];
  imageRevision: number;
  capabilities: RestoreCapabilities;
  canRestore: boolean;
  onFaceChange: (index: number, patch: FacePatch) => void;
  onRestore: (request: RestoreJobRequest) => void;
}

// La grilla queda fuera del panel plegado: no se restaura una cara que no se vea.
function RestoreControls({
  analysis,
  faces,
  imageRevision,
  capabilities,
  canRestore,
  onFaceChange,
  onRestore,
}: RestoreControlsProps) {
  const selection = useRestoreSelection(analysis, capabilities);
  const showFaces = selection.isEnabled(FACES_STEP) && faces.length > 0;
  return (
    <>
      <RestoreSummary
        analysis={analysis}
        capabilities={capabilities}
        selection={selection}
        canRestore={canRestore}
        onRestore={() => onRestore(restoreJobRequest(analysis, selection, faces))}
      />
      {showFaces && (
        <FaceGrid
          faces={faces}
          proposed={analysis.faces}
          stepBlend={stepBlendOf(selection.optionsOf(FACES_STEP))}
          imageRevision={imageRevision}
          onChange={onFaceChange}
        />
      )}
    </>
  );
}

function CapabilitiesStatus({ isError }: { isError: boolean }) {
  const { t } = useTranslation();
  if (isError) {
    return (
      <p role="alert" className="text-xs text-danger">
        {t("restore.capabilities.loadFailed")}
      </p>
    );
  }
  return <p className="text-sm text-text-dim">{t("restore.capabilities.loading")}</p>;
}

export function PhotoRestorePanel() {
  const { t } = useTranslation();
  const session = useRestoreSession();
  const [files, setFiles] = useState<File[]>([]);
  const [rejectedUpload, setRejectedUpload] = useState<string | null>(null);
  const engineQuery = useQuery({ queryKey: ["engine"], queryFn: getEngineInfo });
  const capabilitiesQuery = useQuery({ queryKey: RESTORE_CAPABILITIES_KEY, queryFn: getRestoreCapabilities });
  const restoreJob = useRestoreJob();
  const statusText = useStatusText(session);
  const { analysis } = session;
  const resultSummary = restoreJob.job ? readRestoreSummary(restoreJob.job) : null;

  function handleFilesSelected(selected: File[]) {
    const [photo] = selected;
    restoreJob.reset();
    // Antes de tocar la red: el servidor corta la subida mientras la recibe.
    const limitMb = engineQuery.data?.maxUploadMb ?? null;
    if (exceedsUploadLimit(photo.size, limitMb)) {
      setRejectedUpload(t("upload.tooLarge", { size: formatMegabytes(photo.size), limit: `${limitMb} MB` }));
      setFiles([]);
      session.reset();
      return;
    }
    setRejectedUpload(null);
    setFiles([photo]);
    void session.analyze(photo);
  }

  return (
    <div className="flex flex-col gap-6">
      <FileDropzone
        inputId="restore-file-input"
        accept={PHOTO_ACCEPT}
        multiple={false}
        files={files}
        emptyLabel={t("restore.dropzone")}
        formatsHint={PHOTO_FORMATS}
        onFilesSelected={handleFilesSelected}
      />
      {rejectedUpload && (
        <p role="alert" className="text-xs text-danger">
          {rejectedUpload}
        </p>
      )}
      {statusText && (
        <p role="status" className="text-sm text-text-dim">
          {statusText}
        </p>
      )}
      {session.errorMessage && (
        <p role="alert" className="text-xs text-danger">
          {session.errorMessage}
        </p>
      )}
      {analysis && (
        <div className="flex flex-col gap-3">
          <PhotoFacts analysis={analysis} />
          <GeometryTools
            key={analysis.token}
            previewUrl={versionedUrl(analysis.previewUrl, session.revision)}
            alt={t("restore.preview.alt", { name: analysis.originalName })}
            geometry={analysis.geometry}
            workingSize={{ width: analysis.width, height: analysis.height }}
            busy={session.phase === "updating"}
            onApply={(geometry) => void session.applyGeometry(geometry)}
          />
          {capabilitiesQuery.data ? (
            <RestoreControls
              analysis={analysis}
              faces={session.faces}
              imageRevision={session.revision}
              capabilities={capabilitiesQuery.data}
              canRestore={session.phase === "ready" && !isJobActive(restoreJob.phase)}
              onFaceChange={session.updateFace}
              onRestore={restoreJob.submit}
            />
          ) : (
            <CapabilitiesStatus isError={capabilitiesQuery.isError} />
          )}
        </div>
      )}
      {(restoreJob.phase !== "idle" || restoreJob.errorMessage) && (
        <JobCard
          phase={restoreJob.phase}
          job={restoreJob.job}
          fileName={analysis?.originalName}
          errorMessage={restoreJob.errorMessage}
          onCancel={restoreJob.cancel}
        />
      )}
      {analysis && restoreJob.job && resultSummary ? (
        <RestoreResult
          key={restoreJob.job.jobId}
          job={restoreJob.job}
          summary={resultSummary}
          beforeUrl={versionedUrl(analysis.previewUrl, session.revision)}
        />
      ) : (
        <p className="text-xs text-text-faint">{t("restore.local")}</p>
      )}
    </div>
  );
}
