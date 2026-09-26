import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { FileDropzone } from "../../../components/FileDropzone";
import { useTranslation } from "../../../i18n/LocaleProvider";
import { getEngineInfo } from "../../../lib/api";
import type { RestoreAnalysis } from "../../../lib/restoreApiTypes";
import { exceedsUploadLimit, formatMegabytes } from "../uploadLimit";
import { GeometryTools } from "./GeometryTools";
import type { RestoreSessionState } from "./restoreSessionState";
import { useRestoreSession } from "./useRestoreSession";

const PHOTO_ACCEPT = ".png,.jpg,.jpeg,.webp,.bmp,.tif,.tiff,image/png,image/jpeg,image/webp,image/bmp,image/tiff";
const PHOTO_FORMATS = "PNG, JPG, WEBP, BMP, TIFF";

export function versionedUrl(url: string, revision: number): string {
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}v=${revision}`;
}

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

export function PhotoRestorePanel() {
  const { t } = useTranslation();
  const session = useRestoreSession();
  const [files, setFiles] = useState<File[]>([]);
  const [rejectedUpload, setRejectedUpload] = useState<string | null>(null);
  const engineQuery = useQuery({ queryKey: ["engine"], queryFn: getEngineInfo });
  const statusText = useStatusText(session);
  const { analysis } = session;

  function handleFilesSelected(selected: File[]) {
    const [photo] = selected;
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
        </div>
      )}
      <p className="text-xs text-text-faint">{t("restore.local")}</p>
    </div>
  );
}
