import { Play, TriangleAlert } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { VideoCapabilities } from "../../../lib/apiTypes";
import type {
  CctvAnalysis,
  CctvJobRequest,
  CctvLane,
  CctvParamValue,
  CctvPresetsResponse,
  CctvStepSchema,
  CctvTask,
} from "../../../services/cctv";
import { AiLaneConfirmDialog } from "./AiLaneConfirmDialog";
import { CaseDetailsForm } from "./CaseDetailsForm";
import { CctvFrameTools } from "./CctvFrameTools";
import { CctvLaneSelector } from "./CctvLaneSelector";
import { CctvPresetPicker } from "./CctvPresetPicker";
import { CctvStepCard } from "./CctvStepCard";
import { isCaseDetailsValid } from "./cctvCase";
import {
  buildCctvJobRequest,
  initialChoices,
  withCaseDetails,
  withLane,
  withPreset,
  type CctvChoices,
} from "./cctvChoices";
import { isTrimValid } from "./cctvFrames";
import { aiLaneState, isTaskReady, LANE_TASKS, startBlocker } from "./cctvLanes";
import { incompleteStepIds, visibleSteps, withStepEnabled, withStepFilter, withStepParam } from "./cctvSteps";

interface JobSetupProps {
  analysis: CctvAnalysis;
  presets: CctvPresetsResponse;
  capabilities: VideoCapabilities | undefined;
  busy: boolean;
  onSubmit: (request: CctvJobRequest) => void;
}

function taskTabClassName(isActive: boolean): string {
  const base =
    "rounded-sm border px-3 py-1.5 text-sm transition-[background-color,border-color,color] duration-fast focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent aria-disabled:cursor-not-allowed aria-disabled:opacity-50";
  return isActive ? `${base} border-accent bg-accent text-bg` : `${base} border-border bg-surface text-text-dim hover:text-text`;
}

function TaskTabs({ lane, value, onChange }: { lane: CctvLane; value: CctvTask; onChange: (task: CctvTask) => void }) {
  const { t } = useTranslation();
  return (
    <div role="tablist" aria-label={t("cctv.task.legend")} className="flex flex-wrap gap-2">
      {LANE_TASKS[lane].map((task) => (
        <button
          key={task}
          type="button"
          role="tab"
          aria-selected={task === value}
          aria-disabled={!isTaskReady(task)}
          title={isTaskReady(task) ? undefined : t("cctv.task.unavailable")}
          onClick={() => isTaskReady(task) && onChange(task)}
          className={taskTabClassName(task === value)}
        >
          {t(`cctv.task.${task}`)}
        </button>
      ))}
    </div>
  );
}

function StepList({
  steps,
  choices,
  onChange,
}: {
  steps: CctvStepSchema[];
  choices: CctvChoices;
  onChange: (next: CctvChoices["steps"]) => void;
}) {
  const { t } = useTranslation();
  const current = choices.steps;
  return (
    <div className="flex flex-col gap-2">
      <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">{t("cctv.steps.legend")}</span>
      <ol aria-label={t("cctv.steps.legend")} className="flex flex-col gap-2">
        {steps.map((step) => (
          <CctvStepCard
            key={step.id}
            step={step}
            choice={current[step.id] ?? null}
            onToggle={(enabled) => onChange(withStepEnabled(current, step, enabled))}
            onFilterChange={(filter) => onChange(withStepFilter(current, step.id, filter))}
            onParamChange={(name, value: CctvParamValue | null) => onChange(withStepParam(current, step.id, name, value))}
          />
        ))}
      </ol>
      <p className="text-xs text-text-faint">{t("cctv.fps.interpOff")}</p>
    </div>
  );
}

function AiBanner() {
  const { t } = useTranslation();
  return (
    <p role="note" className="flex items-start gap-2 rounded border border-warn bg-surface-2 p-2 text-xs text-text">
      <TriangleAlert aria-hidden="true" className="mt-0.5 h-3.5 w-3.5 shrink-0 text-warn" strokeWidth={1.75} />
      {t("cctv.ai.banner")}
    </p>
  );
}

export function CctvJobSetup({ analysis, presets, capabilities, busy, onSubmit }: JobSetupProps) {
  const { t } = useTranslation();
  const [choices, setChoices] = useState<CctvChoices>(() => initialChoices(analysis, presets));
  const [confirmingAi, setConfirmingAi] = useState(false);
  const ai = aiLaneState(capabilities, analysis);
  const catalog = presets.steps[choices.lane];
  const blocker = startBlocker({
    modeAvailable: (capabilities?.cctvAvailable ?? true) && analysis.modeAvailable && presets.modeAvailable,
    decodeFailed: analysis.decodeFailed,
    lane: choices.lane,
    aiAvailable: ai.available,
    task: choices.task,
    noOsd: choices.noOsd,
    osdBoxesConfirmed: choices.osdBoxesConfirmed,
    osdBoxCount: choices.osdBoxes.length,
    incompleteStepIds: incompleteStepIds(choices.steps, catalog),
    trimValid: isTrimValid(choices.trim, analysis.frameIndex?.frameCount ?? 0),
    caseDetailsValid: isCaseDetailsValid(choices.caseDetails),
  });

  function submit(): void {
    setConfirmingAi(false);
    onSubmit(buildCctvJobRequest(analysis.token, choices, presets));
  }

  function handleStart(): void {
    if (choices.lane === "ai") {
      setConfirmingAi(true);
      return;
    }
    submit();
  }

  return (
    <div className="flex flex-col gap-5">
      <CctvLaneSelector value={choices.lane} ai={ai} onChange={(lane) => setChoices(withLane(choices, lane, analysis, presets))} />
      {choices.lane === "ai" && <AiBanner />}
      <TaskTabs lane={choices.lane} value={choices.task} onChange={(task) => setChoices({ ...choices, task })} />
      <CctvPresetPicker
        presets={presets.presets}
        value={choices.presetId}
        suggested={analysis.suggestedPreset}
        onChange={(presetId) => setChoices(withPreset(choices, presetId, analysis, presets))}
      />
      <StepList steps={visibleSteps(catalog, choices.lane)} choices={choices} onChange={(steps) => setChoices({ ...choices, steps })} />
      <CctvFrameTools analysis={analysis} choices={choices} catalog={catalog} onChange={setChoices} />
      <CaseDetailsForm value={choices.caseDetails} onChange={(caseDetails) => setChoices(withCaseDetails(choices, caseDetails))} />
      <div className="flex flex-col gap-2">
        {blocker && (
          <p role="status" className="text-xs text-warn">
            {t(blocker.key)}
          </p>
        )}
        <button
          type="button"
          onClick={handleStart}
          disabled={blocker !== null || busy}
          className="inline-flex w-fit items-center gap-2 rounded bg-accent px-4 py-2 text-sm font-medium text-bg transition-[background-color,opacity] duration-fast hover:bg-accent-hover active:bg-accent-press disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          <Play aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
          {t("cctv.start")}
        </button>
      </div>
      {confirmingAi && <AiLaneConfirmDialog onConfirm={submit} onCancel={() => setConfirmingAi(false)} />}
    </div>
  );
}
