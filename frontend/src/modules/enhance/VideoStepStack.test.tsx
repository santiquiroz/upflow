import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { en } from "../../i18n/en";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import { VideoStepStack } from "./VideoStepStack";
import type { VideoStep, VideoStepId } from "./videoSteps";

const ALL_STEPS: VideoStep[] = [
  {
    id: "upscale",
    modelName: "RealESRGAN x4plus",
    scale: 4,
  },
  {
    id: "interpolate",
    interpEngine: "rife",
    fpsMultiplier: 2,
    targetFps: null,
  },
  {
    id: "audio",
    audioEnhance: "deepfilter",
    audioRestore: "apollo",
  },
  { id: "subtitles" },
];

function copy(
  key: keyof typeof en,
  params: Record<string, string | number> = {},
): string {
  return en[key].replace(/\{\{(\w+)\}\}/g, (match, name: string) => {
    const value = params[name];
    return value === undefined ? match : String(value);
  });
}

function renderStack({
  steps = ALL_STEPS,
  addableStepIds = [],
  onRemove = vi.fn(),
  onAdd = vi.fn(),
}: {
  steps?: VideoStep[];
  addableStepIds?: VideoStepId[];
  onRemove?: (stepId: VideoStepId) => void;
  onAdd?: (stepId: VideoStepId) => void;
} = {}) {
  render(
    <LocaleProvider initialLocale="en">
      <VideoStepStack
        steps={steps}
        addableStepIds={addableStepIds}
        onRemove={onRemove}
        onAdd={onAdd}
      />
    </LocaleProvider>,
  );
}

describe("VideoStepStack", () => {
  it("shows the active steps in numbered execution order with visible descriptions", () => {
    renderStack();

    const list = screen.getByRole("list", { name: en["video.steps.listLabel"] });
    const items = within(list).getAllByRole("listitem");
    const expectedLabels = [
      en["video.steps.upscale.label"],
      en["video.steps.interpolate.label"],
      en["video.steps.audio.label"],
      en["video.steps.subtitles.label"],
    ];

    expect(items).toHaveLength(expectedLabels.length);
    expectedLabels.forEach((label, index) => {
      expect(items[index]).toHaveTextContent(label);
      expect(items[index]).toHaveTextContent(String(index + 1));
    });
    expect(
      screen.getByText(
        copy("video.steps.upscale.description", {
          model: "RealESRGAN x4plus",
          scale: 4,
        }),
      ),
    ).toBeVisible();
    expect(
      screen.getByText(
        copy("video.steps.interpolate.multiplierDescription", {
          engine: "RIFE",
          multiplier: 2,
        }),
      ),
    ).toBeVisible();
    expect(
      screen.getByText(
        copy("video.steps.audio.description", {
          modes: "DeepFilterNet + Apollo",
        }),
      ),
    ).toBeVisible();
    expect(screen.getByText(en["video.steps.subtitles.description"])).toBeVisible();
  });

  it("reports the step selected through its translated remove control", () => {
    const onRemove = vi.fn();
    renderStack({ onRemove });

    fireEvent.click(
      screen.getByRole("button", {
        name: copy("video.steps.removeStep", {
          step: en["video.steps.interpolate.label"],
        }),
      }),
    );

    expect(onRemove).toHaveBeenCalledWith("interpolate");
  });

  it("offers only missing steps and reports the selected addition", () => {
    const onAdd = vi.fn();
    renderStack({
      steps: [ALL_STEPS[0]],
      addableStepIds: ["audio", "subtitles"],
      onAdd,
    });

    expect(
      screen.queryByRole("button", {
        name: copy("video.steps.addStep", {
          step: en["video.steps.interpolate.label"],
        }),
      }),
    ).not.toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", {
        name: copy("video.steps.addStep", {
          step: en["video.steps.audio.label"],
        }),
      }),
    );

    expect(onAdd).toHaveBeenCalledWith("audio");
  });

  it("shows the translated empty state when no configuration fields generate a step", () => {
    renderStack({ steps: [], addableStepIds: [] });

    expect(screen.getByText(en["video.steps.empty"])).toBeVisible();
    expect(
      screen.queryByRole("list", { name: en["video.steps.listLabel"] }),
    ).not.toBeInTheDocument();
  });
});
