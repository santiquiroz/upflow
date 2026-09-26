import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import type { RestoreStepCapability } from "../../../lib/restoreApiTypes";
import { RestorePresetPicker } from "./RestorePresetPicker";
import { RestoreStepCard } from "./RestoreStepCard";
import { makeCapabilities } from "./restoreTestFixtures";

function stepCapability(id: string, missingPacks: string[] = []): RestoreStepCapability {
  const step = makeCapabilities(missingPacks).steps.find((candidate) => candidate.id === id);
  if (!step) throw new Error(id);
  return step;
}

function withQueryClient(children: ReactNode) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}

function renderCard(id: string, enabled: boolean, options: Record<string, unknown> = {}) {
  const onToggle = vi.fn();
  const onOptionChange = vi.fn();
  render(
    withQueryClient(
      <RestoreStepCard
        step={stepCapability(id)}
        position={4}
        isLast={false}
        enabled={enabled}
        options={options}
        onToggle={onToggle}
        onOptionChange={onOptionChange}
      />,
    ),
  );
  return { onToggle, onOptionChange };
}

describe("RestoreStepCard", () => {
  it("shows the intensity only while the step is on", () => {
    renderCard("denoise", false, { strength: 0.3, keep_grain: 0.25 });

    expect(screen.queryByRole("slider")).not.toBeInTheDocument();
    expect(screen.getByText("4")).toBeInTheDocument();
  });

  it("toggles the step from its checkbox", () => {
    const { onToggle } = renderCard("denoise", false);

    fireEvent.click(screen.getByRole("checkbox", { name: "Reduce noise" }));

    expect(onToggle).toHaveBeenCalledWith(true);
  });

  it("changes the intensity and shows it as a percentage", () => {
    const { onOptionChange } = renderCard("denoise", true, { strength: 0.3, keep_grain: 0.25 });

    expect(screen.getByText("30%")).toBeInTheDocument();
    fireEvent.change(screen.getByRole("slider", { name: "Strength" }), { target: { value: "0.5" } });

    expect(onOptionChange).toHaveBeenCalledWith("strength", 0.5);
  });

  it("keeps the advanced settings folded until asked, with Keep grain at 25%", () => {
    const { onOptionChange } = renderCard("denoise", true, { strength: 0.3, keep_grain: 0.25 });
    const advanced = screen.getByRole("button", { name: "Advanced" });
    expect(advanced).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("slider", { name: "Keep grain" })).not.toBeInTheDocument();

    fireEvent.click(advanced);
    const grain = screen.getByRole("slider", { name: "Keep grain" });

    expect(grain).toHaveValue("0.25");
    fireEvent.change(grain, { target: { value: "0.4" } });
    expect(onOptionChange).toHaveBeenCalledWith("keep_grain", 0.4);
  });

  it("offers the choices and toggles of each step", () => {
    const repair = renderCard("repair", true, { engine: "fast", sensitivity: 0.5, grow_px: 1 });
    fireEvent.click(screen.getByRole("button", { name: "Advanced" }));

    fireEvent.change(screen.getByRole("combobox", { name: "Fill method" }), { target: { value: "classic" } });
    expect(repair.onOptionChange).toHaveBeenCalledWith("engine", "classic");
    expect(screen.getByText("+1 px")).toBeInTheDocument();
  });

  it("says before choosing that a step can invent detail", () => {
    renderCard("faces", false);

    expect(screen.getByText("AI-generated facial detail. It can change how a person looks.")).toBeInTheDocument();
  });

  it("can't be turned on without its pack", () => {
    render(
      withQueryClient(
        <RestoreStepCard
          step={stepCapability("colorize", ["restore-colorize"])}
          position={7}
          isLast
          enabled={false}
          options={{}}
          onToggle={vi.fn()}
          onOptionChange={vi.fn()}
        />,
      ),
    );

    expect(screen.getByRole("checkbox", { name: "Colorize" })).toBeDisabled();
    expect(screen.getAllByText("This fix needs the restore-colorize pack.").length).toBeGreaterThan(0);
  });
});

describe("RestorePresetPicker", () => {
  const presets = makeCapabilities().presets;

  it("selects a preset and marks the suggested ones", () => {
    const onSelect = vi.fn();
    render(
      <RestorePresetPicker presets={presets} selectedId="gentle" suggestedIds={["portrait"]} customized={false} onSelect={onSelect} />,
    );

    expect(screen.getByRole("radio", { name: "Gentle" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Portrait" }).parentElement).toHaveTextContent("Suggested");
    fireEvent.click(screen.getByRole("radio", { name: "Faded color print" }));

    expect(onSelect).toHaveBeenCalledWith("faded_color_print");
  });

  it("goes back to the untouched preset after manual changes", () => {
    const onSelect = vi.fn();
    render(<RestorePresetPicker presets={presets} selectedId="gentle" suggestedIds={[]} customized onSelect={onSelect} />);

    fireEvent.click(screen.getByRole("button", { name: "Back to the preset" }));

    expect(onSelect).toHaveBeenCalledWith("gentle");
  });
});
