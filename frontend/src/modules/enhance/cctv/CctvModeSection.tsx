import { useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { JobCard } from "../../../components/JobCard";
import { useVideoCapabilities, type VideoJobPhase } from "../../../hooks/useVideoJob";
import { useTranslation } from "../../../i18n/LocaleProvider";
import { getEngineInfo } from "../../../lib/api";
import type { VideoCapabilities, VideoJobResponse } from "../../../lib/apiTypes";
import { getCctvPresets } from "../../../services/cctv";
import { exceedsUploadLimit, formatMegabytes } from "../uploadLimit";
import { CctvDiagnosisCard } from "./CctvDiagnosisCard";
import { CctvDropzone } from "./CctvDropzone";
import { CctvJobSetup } from "./CctvJobSetup";
import { CctvModeToggle } from "./CctvModeToggle";
import { CctvResult } from "./CctvResult";
import { errorInfoOf, errorText } from "./cctvText";
import { useCctvJob } from "./useCctvJob";
import { useCctvSession, type CctvSession } from "./useCctvSession";

function isJobBusy(phase: VideoJobPhase): boolean {
  return phase === "uploading" || phase === "queued" || phase === "running";
}

function isSessionBusy(session: CctvSession): boolean {
  return session.phase === "uploading" || session.phase === "analyzing";
}

function SessionStatus({ session }: { session: CctvSession }) {
  const { t } = useTranslation();
  if (session.phase === "uploading") {
    const text =
      session.uploadPercent === null ? t("cctv.uploadingUnknown") : t("cctv.uploading", { percent: session.uploadPercent });
    return <p role="status" className="text-xs text-text-dim">{text}</p>;
  }
  if (session.phase === "analyzing") {
    return <p role="status" className="text-xs text-text-dim">{t("cctv.analyzing")}</p>;
  }
  if (session.phase === "failed" && session.error) {
    return <p role="alert" className="text-xs text-danger">{errorText(t, session.error)}</p>;
  }
  return null;
}

function ModeUnavailable({ capabilities }: { capabilities: VideoCapabilities | undefined }) {
  const { t } = useTranslation();
  if (!capabilities || capabilities.cctvAvailable || !capabilities.cctvReasonKey) {
    return null;
  }
  return <p role="status" className="text-xs text-warn">{t(capabilities.cctvReasonKey)}</p>;
}

function CompletedResult({ job, retentionHours }: { job: VideoJobResponse | undefined; retentionHours: number | null }) {
  if (job?.status !== "completed" || !job.cctv) {
    return null;
  }
  return <CctvResult key={job.jobId} jobId={job.jobId} summary={job.cctv} retentionHours={retentionHours} />;
}

function useAnalyzeOnMount(initialFile: File | null, analyze: (file: File) => void): void {
  // El ref evita subir dos veces el mismo archivo cuando StrictMode repite el efecto.
  const startedRef = useRef(false);
  useEffect(() => {
    if (initialFile && !startedRef.current) {
      startedRef.current = true;
      analyze(initialFile);
    }
  }, [initialFile, analyze]);
}

export function CctvModeSection({ initialFile, onExit }: { initialFile: File | null; onExit: () => void }) {
  const { t } = useTranslation();
  const [fileName, setFileName] = useState<string | null>(initialFile?.name ?? null);
  const [rejectedUpload, setRejectedUpload] = useState<string | null>(null);
  const session = useCctvSession();
  const cctvJob = useCctvJob();
  const presetsQuery = useQuery({ queryKey: ["cctvPresets"], queryFn: getCctvPresets });
  const engineQuery = useQuery({ queryKey: ["engine"], queryFn: getEngineInfo });
  const capabilitiesQuery = useVideoCapabilities();
  useAnalyzeOnMount(initialFile, session.analyze);

  function handleFileSelected(file: File): void {
    // Igual que el panel de video: rechazar antes de subir un archivo que el backend no aceptaria.
    const limitMb = engineQuery.data?.maxVideoUploadMb ?? null;
    if (exceedsUploadLimit(file.size, limitMb)) {
      const tooLarge = t("upload.tooLarge", { size: formatMegabytes(file.size), limit: `${limitMb} MB` });
      setRejectedUpload(`${tooLarge} ${t("upload.tooLarge.admin")}`);
      return;
    }
    setRejectedUpload(null);
    setFileName(file.name);
    session.analyze(file);
  }

  const analysis = session.analysis;
  const presetsError = errorInfoOf(presetsQuery.error);
  const jobError = cctvJob.error ? errorText(t, cctvJob.error) : null;

  return (
    <div className="grid grid-cols-[1fr_320px] gap-6 max-[900px]:grid-cols-1">
      <div className="flex flex-col gap-6">
        <CctvModeToggle checked onChange={(checked) => !checked && onExit()} />
        <ModeUnavailable capabilities={capabilitiesQuery.data} />
        <CctvDropzone fileName={fileName} disabled={isSessionBusy(session)} onFileSelected={handleFileSelected} />
        {rejectedUpload && <p role="alert" className="text-xs text-danger">{rejectedUpload}</p>}
        <SessionStatus session={session} />
        {presetsError && <p role="alert" className="text-xs text-danger">{errorText(t, presetsError)}</p>}
        {analysis && <CctvDiagnosisCard analysis={analysis} />}
        {analysis && presetsQuery.data && (
          <CctvJobSetup
            key={analysis.token}
            analysis={analysis}
            presets={presetsQuery.data}
            capabilities={capabilitiesQuery.data}
            busy={isJobBusy(cctvJob.phase)}
            onSubmit={cctvJob.submit}
          />
        )}
      </div>
      <div className="flex flex-col gap-4">
        <JobCard
          phase={cctvJob.phase}
          job={cctvJob.job}
          fileName={fileName ?? undefined}
          errorMessage={jobError}
          onCancel={cctvJob.cancel}
          uploadPercent={null}
        />
        <CompletedResult job={cctvJob.job} retentionHours={engineQuery.data?.outputTtlHours ?? null} />
      </div>
    </div>
  );
}
