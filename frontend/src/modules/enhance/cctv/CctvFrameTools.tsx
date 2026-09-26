import { useState } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { CctvAnalysis, CctvStepSchema } from "../../../services/cctv";
import { BoxEditor } from "./BoxEditor";
import { CctvFilterPreview } from "./CctvFilterPreview";
import type { FrameSize } from "./cctvBoxes";
import { withRedaction, withRoi, withTrim, type CctvChoices } from "./cctvChoices";
import { displaySizeOf, storedSizeOf } from "./cctvFrames";
import { isVideoTask, usesFilters, usesTrim } from "./cctvLanes";
import { roiReferenceRequest, withRoiBox } from "./cctvRoi";
import { redactionSpan } from "./cctvRedaction";
import { hasChosenAiSteps, previewStepRequests } from "./cctvSteps";
import { FrameScrubber } from "./FrameScrubber";
import { RedactionBoxEditor, RedactionPanel } from "./RedactionPanel";
import { RoiFusionPanel } from "./RoiFusionPanel";
import { TrimControls } from "./TrimControls";
import { useOsdTools } from "./useOsdTools";
import { useRoiReferenceSuggestion } from "./useRoiReferenceSuggestion";

interface FrameToolsProps {
  analysis: CctvAnalysis;
  choices: CctvChoices;
  catalog: readonly CctvStepSchema[];
  onChange: (choices: CctvChoices) => void;
}

// La ROI se dibuja en coordenadas del cuadro guardado: el ancho que se lee es el que se grabo.
function RoiBoxEditor({
  choices,
  frame,
  frameSize,
  onChange,
}: {
  choices: CctvChoices;
  frame: number;
  frameSize: FrameSize;
  onChange: (choices: CctvChoices) => void;
}) {
  return (
    <BoxEditor
      kind="roi"
      frameSize={frameSize}
      boxes={choices.roi.box ? [choices.roi.box] : []}
      onChange={(boxes) => onChange(withRoi(choices, withRoiBox(choices.roi, boxes[0] ?? null, frame)))}
    />
  );
}

export function CctvFrameTools({ analysis, choices, catalog, onChange }: FrameToolsProps) {
  const { t } = useTranslation();
  const [frame, setFrame] = useState(0);
  const frameCount = analysis.frameIndex?.frameCount ?? 0;
  const hasFrames = frameCount > 0;
  const frameSize = storedSizeOf(analysis);
  const displaySize = displaySizeOf(analysis);
  const isVideo = isVideoTask(choices.task);
  const isRedact = choices.task === "redact";
  const isRoi = choices.task === "roi_fusion";
  const span = redactionSpan(choices.trim, frameCount);
  const osd = useOsdTools({ token: analysis.token, choices, frame, frameSize, onChange });
  const roiEditor = <RoiBoxEditor choices={choices} frame={frame} frameSize={frameSize} onChange={onChange} />;
  const redactEditor = (
    <RedactionBoxEditor
      redaction={choices.redaction}
      frame={frame}
      frameSize={frameSize}
      span={span}
      onChange={(redaction) => onChange(withRedaction(choices, redaction))}
    />
  );
  const overlay = isRedact ? redactEditor : isVideo ? osd.overlay : roiEditor;
  const stepRequests = previewStepRequests(choices.steps, catalog);
  const suggestion = useRoiReferenceSuggestion({
    token: analysis.token,
    roi: choices.roi,
    request: roiReferenceRequest(choices.roi, frameCount, stepRequests),
    onChange: (roi) => onChange(withRoi(choices, roi)),
    onShowFrame: setFrame,
  });

  return (
    <section aria-label={t("cctv.frames.legend")} className="flex flex-col gap-4">
      <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">{t("cctv.frames.legend")}</span>
      {hasFrames && (
        <FrameScrubber
          token={analysis.token}
          frameCount={frameCount}
          frame={frame}
          index={analysis.frameIndex}
          displaySize={displaySize}
          onFrameChange={setFrame}
          overlay={overlay}
        />
      )}
      {isVideo && osd.panel}
      {usesTrim(choices.task) && hasFrames && (
        <TrimControls
          frameCount={frameCount}
          currentFrame={frame}
          index={analysis.frameIndex}
          trim={choices.trim}
          onChange={(trim) => onChange(withTrim(choices, trim, frameCount))}
        />
      )}
      {isRedact && hasFrames && (
        <RedactionPanel
          redaction={choices.redaction}
          frame={frame}
          span={span}
          onChange={(redaction) => onChange(withRedaction(choices, redaction))}
          onShowFrame={setFrame}
        />
      )}
      {isRoi && hasFrames && (
        <RoiFusionPanel
          roi={choices.roi}
          frame={frame}
          frameCount={frameCount}
          index={analysis.frameIndex}
          onChange={(roi) => onChange(withRoi(choices, roi))}
          onShowFrame={setFrame}
          suggestion={suggestion}
        />
      )}
      {usesFilters(choices.task) && hasFrames && (
        <CctvFilterPreview
          token={analysis.token}
          frame={frame}
          steps={stepRequests}
          skipsAiSteps={hasChosenAiSteps(choices.steps, catalog)}
          displaySize={displaySize}
        />
      )}
    </section>
  );
}
