import type { ComponentProps } from "react";
import { JobCard } from "../../components/JobCard";
import type { GenerationJobPhase } from "../../hooks/useGenerationJob";
import { useTranslation } from "../../i18n/LocaleProvider";
import type { EraseZone, GenerationJob, InitImageResponse } from "../../lib/apiTypes";
import { EditorActionControls } from "./EditorActionControls";
import type { EditorSidePanelView } from "./EditorSidePanel";
import { PeoplePanel } from "./PeoplePanel";
import { PeopleSummary } from "./PeopleSummary";
import { personNumbers } from "./peopleState";
import { ReviewPanel } from "./ReviewPanel";
import type { usePeopleSelection } from "./usePeopleSelection";
import type { useSilhouetteBatch } from "./useSilhouetteBatch";
import type { RedoEngine } from "./zones";

export interface EditorSidePeople {
  selection: ReturnType<typeof usePeopleSelection>;
  batch: ReturnType<typeof useSilhouetteBatch>;
  personIds: number[];
  onBack: () => void;
}

export interface EditorSideReview {
  zones: EraseZone[] | null;
  hoveredZone: number | null;
  onHoverZone: (index: number | null) => void;
  onRedo: (zone: EraseZone, engine: RedoEngine, prompt?: string) => void;
  sdxlAvailable: boolean;
  eraserAvailable: boolean;
  redoingIndex: number | null;
  redoProgress: number | null;
  errors: Record<number, string>;
  resultUrl: string | null;
}

export interface EditorSideJob {
  phase: GenerationJobPhase;
  job: GenerationJob | undefined;
  errorMessage: string | null;
  onCancel: () => void;
  onUseAsBase: () => void;
}

interface EditorSideContentProps {
  view: EditorSidePanelView;
  baseInfo: InitImageResponse;
  baseUrl: string | null;
  uploadError: string | null;
  people: EditorSidePeople;
  review: EditorSideReview;
  job: EditorSideJob;
  actionControls: ComponentProps<typeof EditorActionControls>;
}

function PeopleView({ people, baseInfo, baseUrl }: Pick<EditorSideContentProps, "people" | "baseInfo" | "baseUrl">) {
  const { t } = useTranslation();
  const { selection, batch } = people;
  return (
    <PeoplePanel
      people={selection.people}
      choices={selection.choices}
      hoveredId={selection.hoveredId}
      onHover={selection.setHoveredId}
      onToggle={selection.toggle}
      onGroup={selection.setGroupChoice}
      protectKept={selection.protectKept}
      onProtectKeptChange={selection.setProtectKept}
      onAccept={() => void batch.accept()}
      progress={batch.progress}
      onRetry={() => void selection.refetch()}
      status={selection.status}
      imageUrl={baseUrl ?? ""}
      imageWidth={baseInfo.width}
      imageHeight={baseInfo.height}
      unavailableReason={selection.error ?? t("editor.tool.peopleUnavailable")}
      error={selection.error}
    />
  );
}

function ReviewView({ review, people, baseInfo, baseUrl }: Pick<EditorSideContentProps, "review" | "people" | "baseInfo" | "baseUrl">) {
  if (!review.zones) return null;
  return (
    <ReviewPanel
      zones={review.zones}
      people={people.selection.people}
      choices={people.selection.choices}
      hoveredZone={review.hoveredZone}
      onHoverZone={review.onHoverZone}
      onRedo={review.onRedo}
      sdxlAvailable={review.sdxlAvailable}
      eraserAvailable={review.eraserAvailable}
      redoingIndex={review.redoingIndex}
      redoProgress={review.redoProgress}
      errors={review.errors}
      beforeUrl={baseUrl}
      afterUrl={review.resultUrl}
      imageSize={baseInfo}
    />
  );
}

function ReviewJob({ job, fileName }: { job: EditorSideJob; fileName: string }) {
  const { t } = useTranslation();
  const shown = job.job;
  return (
    <div className="mb-3">
      <JobCard phase={job.phase} job={shown} fileName={fileName} errorMessage={job.errorMessage} onCancel={job.onCancel} />
      {shown?.status === "completed" && shown.downloadUrl && (
        <button
          type="button"
          onClick={job.onUseAsBase}
          className="mt-3 min-h-9 rounded border border-border px-4 py-2 text-sm text-text hover:bg-surface-2"
        >
          {t("editor.useAsBase")}
        </button>
      )}
    </div>
  );
}

export function EditorSideContent(props: EditorSideContentProps) {
  const { view, baseInfo, uploadError, people, job, actionControls } = props;
  return (
    <>
      {uploadError && <p className="mb-3 text-sm text-danger">{uploadError}</p>}
      {view === "action" && (
        <PeopleSummary
          personIds={people.personIds}
          numbers={personNumbers(people.selection.people)}
          failed={people.batch.failed}
          onBack={people.onBack}
        />
      )}
      {view === "people" && <PeopleView {...props} />}
      {view === "review" && <ReviewView {...props} />}
      {view === "review" && <ReviewJob job={job} fileName={baseInfo.originalFilename} />}
      {view !== "people" && <EditorActionControls {...actionControls} />}
      {view !== "people" && (
        <JobCard
          phase={job.phase}
          job={job.job}
          fileName={baseInfo.originalFilename}
          errorMessage={job.errorMessage}
          onCancel={job.onCancel}
        />
      )}
    </>
  );
}
