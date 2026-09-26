import { fireEvent, render, screen, within } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { en } from "../../../i18n/en";
import type { CctvAiUpscaleModel } from "../../../services/cctv";
import { AiUpscalePicker } from "./AiUpscalePicker";
import type { AiUpscaleChoice } from "./cctvAiUpscale";
import { AI_UPSCALE_MODELS } from "./cctvFixtures";

function StatefulPicker({ models, scaleHint }: { models: CctvAiUpscaleModel[]; scaleHint: number | null }) {
  const [value, setValue] = useState<AiUpscaleChoice | null>(null);
  return <AiUpscalePicker models={models} value={value} scaleHint={scaleHint} onChange={setValue} />;
}

function modelSelect(): HTMLSelectElement {
  return screen.getByLabelText(en["cctv.ai.upscale.model"]);
}

describe("AiUpscalePicker", () => {
  it("starts at None and tags every model as generative or not", () => {
    render(<StatefulPicker models={AI_UPSCALE_MODELS} scaleHint={null} />);

    expect(modelSelect()).toHaveValue("");
    const options = within(modelSelect()).getAllByRole("option").map((option) => option.textContent);
    expect(options).toEqual([
      en["cctv.ai.upscale.none"],
      "Real-ESRGAN x4plus — Generative (invents texture)",
      "Plain x2 — Non-generative",
    ]);
    expect(screen.queryByRole("radiogroup")).not.toBeInTheDocument();
  });

  it("offers only the scales the model's export has, starting at the preset's hint", () => {
    render(<StatefulPicker models={AI_UPSCALE_MODELS} scaleHint={4} />);

    fireEvent.change(modelSelect(), { target: { value: "realesrgan-x4plus" } });

    const scales = within(screen.getByRole("radiogroup", { name: en["cctv.ai.upscale.scale"] })).getAllByRole("radio");
    expect(scales.map((radio) => radio.textContent)).toEqual(["2x", "4x"]);
    expect(screen.getByRole("radio", { name: "4x" })).toHaveAttribute("aria-checked", "true");
    expect(screen.getByText(en["cctv.ai.upscale.hint"])).toBeInTheDocument();
  });

  it("falls back to a scale the new model can run", () => {
    render(<StatefulPicker models={AI_UPSCALE_MODELS} scaleHint={4} />);
    fireEvent.change(modelSelect(), { target: { value: "realesrgan-x4plus" } });

    fireEvent.change(modelSelect(), { target: { value: "plain-x2" } });

    expect(screen.getByRole("radio", { name: "2x" })).toHaveAttribute("aria-checked", "true");
  });

  it("goes back to no AI upscale with None", () => {
    render(<StatefulPicker models={AI_UPSCALE_MODELS} scaleHint={null} />);
    fireEvent.change(modelSelect(), { target: { value: "plain-x2" } });

    fireEvent.change(modelSelect(), { target: { value: "" } });

    expect(screen.queryByRole("radiogroup")).not.toBeInTheDocument();
  });

  it("explains why there is nothing to pick without a stream-capable model", () => {
    render(<StatefulPicker models={[]} scaleHint={null} />);

    expect(modelSelect()).toBeDisabled();
    expect(screen.getByText(en["cctv.ai.upscale.noModels"])).toBeInTheDocument();
  });
});
