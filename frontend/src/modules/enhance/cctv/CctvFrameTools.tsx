import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import { checkCctvOsd, type CctvAnalysis, type CctvBox, type CctvStepSchema } from "../../../services/cctv";
import { BoxEditor } from "./BoxEditor";
import { CctvFilterPreview } from "./CctvFilterPreview";
import { CctvOsdPanel } from "./CctvOsdPanel";
import { notTextIndices, suggestedOsdBoxes, withBoxRemoved } from "./cctvBoxes";
import { withNoOsd, withOsdBoxes, withOsdConfirmed, withTrim, type CctvChoices } from "./cctvChoices";
import { displaySizeOf, storedSizeOf } from "./cctvFrames";
import { hasChosenAiSteps, previewStepRequests } from "./cctvSteps";
import { errorInfoOf, errorText } from "./cctvText";
import { FrameScrubber } from "./FrameScrubber";
import { TrimControls } from "./TrimControls";

interface FrameToolsProps {
  analysis: CctvAnalysis;
  choices: CctvChoices;
  catalog: readonly CctvStepSchema[];
  onChange: (choices: CctvChoices) => void;
}

interface OsdCheckInput {
  boxes: readonly CctvBox[];
  frame: number;
}

function useOsdCheck(token: string) {
  return useMutation({ mutationFn: ({ boxes, frame }: OsdCheckInput) => checkCctvOsd(token, boxes, frame) });
}

export function CctvFrameTools({ analysis, choices, catalog, onChange }: FrameToolsProps) {
  const { t } = useTranslation();
  const [frame, setFrame] = useState(0);
  const [confirmedFrame, setConfirmedFrame] = useState<number | null>(null);
  const osdCheck = useOsdCheck(analysis.token);
  const frameCount = analysis.frameIndex?.frameCount ?? 0;
  const hasFrames = frameCount > 0;
  const frameSize = storedSizeOf(analysis);
  const displaySize = displaySizeOf(analysis);
  const checkError = errorInfoOf(osdCheck.error);
  const flagged = notTextIndices(choices.osdBoxes, osdCheck.data?.checks ?? []);

  function changeBoxes(boxes: readonly CctvBox[]): void {
    osdCheck.reset();
    onChange(withOsdBoxes(choices, boxes));
  }

  // El chequeo de texto no bloquea: la confirmacion vale aunque diga que una caja no parece texto.
  function confirmBoxes(): void {
    setConfirmedFrame(frame);
    onChange(withOsdConfirmed(choices));
    osdCheck.mutate({ boxes: choices.osdBoxes, frame });
  }

  const boxEditor = (
    <BoxEditor
      kind="osd"
      frameSize={frameSize}
      boxes={choices.noOsd ? [] : choices.osdBoxes}
      onChange={changeBoxes}
      flagged={flagged}
      disabled={choices.noOsd}
    />
  );

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
          overlay={boxEditor}
        />
      )}
      <CctvOsdPanel
        boxes={choices.osdBoxes}
        confirmed={choices.osdBoxesConfirmed}
        confirmedFrame={confirmedFrame}
        noOsd={choices.noOsd}
        flagged={flagged}
        checking={osdCheck.isPending}
        checkError={checkError ? errorText(t, checkError) : null}
        onConfirm={confirmBoxes}
        onNoOsdChange={(noOsd) => onChange(withNoOsd(choices, noOsd))}
        onUseSuggested={() => changeBoxes(suggestedOsdBoxes(frameSize))}
        onRemove={(index) => changeBoxes(withBoxRemoved(choices.osdBoxes, index))}
      />
      {hasFrames && (
        <TrimControls
          frameCount={frameCount}
          currentFrame={frame}
          index={analysis.frameIndex}
          trim={choices.trim}
          onChange={(trim) => onChange(withTrim(choices, trim))}
        />
      )}
      {hasFrames && (
        <CctvFilterPreview
          token={analysis.token}
          frame={frame}
          steps={previewStepRequests(choices.steps, catalog)}
          skipsAiSteps={hasChosenAiSteps(choices.steps, catalog)}
          displaySize={displaySize}
        />
      )}
    </section>
  );
}
