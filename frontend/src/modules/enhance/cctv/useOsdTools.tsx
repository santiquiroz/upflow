import { useMutation } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import { checkCctvOsd, type CctvBox } from "../../../services/cctv";
import { BoxEditor } from "./BoxEditor";
import { CctvOsdPanel } from "./CctvOsdPanel";
import { notTextIndices, suggestedOsdBoxes, withBoxRemoved, type FrameSize } from "./cctvBoxes";
import { withNoOsd, withOsdBoxes, withOsdConfirmed, type CctvChoices } from "./cctvChoices";
import { errorInfoOf, errorText } from "./cctvText";

interface OsdCheckInput {
  boxes: readonly CctvBox[];
  frame: number;
}

interface OsdToolsInput {
  token: string;
  choices: CctvChoices;
  frame: number;
  frameSize: FrameSize;
  onChange: (choices: CctvChoices) => void;
}

export interface OsdTools {
  overlay: ReactNode;
  panel: ReactNode;
}

function useOsdCheck(token: string) {
  return useMutation({ mutationFn: ({ boxes, frame }: OsdCheckInput) => checkCctvOsd(token, boxes, frame) });
}

// El editor va encima del cuadro y el panel debajo; los dos comparten el chequeo de texto.
export function useOsdTools({ token, choices, frame, frameSize, onChange }: OsdToolsInput): OsdTools {
  const { t } = useTranslation();
  const [confirmedFrame, setConfirmedFrame] = useState<number | null>(null);
  const osdCheck = useOsdCheck(token);
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

  const overlay = (
    <BoxEditor
      kind="osd"
      frameSize={frameSize}
      boxes={choices.noOsd ? [] : choices.osdBoxes}
      onChange={changeBoxes}
      flagged={flagged}
      disabled={choices.noOsd}
    />
  );
  const panel = (
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
  );
  return { overlay, panel };
}
