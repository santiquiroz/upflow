import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import type { RestoreAnalysis, RestoreCapabilities, RestoreFinding } from "../../../lib/restoreApiTypes";
import { RestoreSummary } from "./RestoreSummary";
import { makeAnalysis, makeCapabilities } from "./restoreTestFixtures";
import { useRestoreSelection } from "./useRestoreSelection";

function finding(key: string, reasonKey: string, params: RestoreFinding["params"] = {}): RestoreFinding {
  return { key, value: null, reasonKey, params, proposes: [], missingPack: null, message: null };
}

const ANALYSIS = makeAnalysis({
  diagnosis: {
    findings: [
      finding("tone", "restore.diag.mono"),
      finding("damage", "restore.diag.damage", { pct: 4, largeAreas: 0 }),
      finding("faded", "restore.diag.faded", { cast: "cyan/green" }),
      finding("faces", "restore.diag.faces", { count: 1 }),
    ],
    suggestedPresets: ["heavy_damage"],
    toneKind: "mono",
  },
  proposedPreset: "gentle",
  proposedSteps: ["repair", "denoise"],
  proposedOptions: { repair: { sensitivity: 0.5 }, denoise: { strength: 0.3 } },
  presetSelections: {
    gentle: { steps: ["repair", "denoise"], options: { repair: { sensitivity: 0.5 }, denoise: { strength: 0.3 } } },
    newspaper: {
      steps: ["descreen", "denoise"],
      options: { descreen: { mode: "halftone", strength: 1 }, denoise: { strength: 0.2 } },
    },
  },
  eta: {
    gpuSeconds: 40,
    cpuSeconds: 240,
    perStep: { repair: { gpuSeconds: 10, cpuSeconds: 60 }, denoise: { gpuSeconds: 30, cpuSeconds: 180 } },
  },
});

interface HarnessProps {
  analysis?: RestoreAnalysis;
  capabilities?: RestoreCapabilities;
  canRestore?: boolean;
  onRestore?: () => void;
}

function Harness({ analysis = ANALYSIS, capabilities = makeCapabilities(), canRestore = true, onRestore = vi.fn() }: HarnessProps) {
  const selection = useRestoreSelection(analysis, capabilities);
  return (
    <RestoreSummary
      analysis={analysis}
      capabilities={capabilities}
      selection={selection}
      canRestore={canRestore}
      onRestore={onRestore}
    />
  );
}

function renderSummary(props: HarnessProps = {}) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<Harness {...props} />, { wrapper: Wrapper });
}

function openSteps() {
  fireEvent.click(screen.getByRole("button", { name: "Customize" }));
}

describe("RestoreSummary", () => {
  it("shows the summary first with Restore visible and the steps folded", () => {
    renderSummary();

    expect(screen.getByRole("button", { name: "Restore" })).toBeEnabled();
    expect(screen.getByText("2 fixes selected")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Customize" })).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("checkbox", { name: "Repair damage" })).not.toBeInTheDocument();
  });

  it("lists what the analysis found, in words", () => {
    renderSummary();

    const list = screen.getByRole("list", { name: "What the analysis found" });
    expect(within(list).getByText("Looks black & white")).toBeInTheDocument();
    expect(within(list).getByText("Scratches and dust: about 4% of the photo")).toBeInTheDocument();
    expect(within(list).getByText("Faded colors (cyan/green cast)")).toBeInTheDocument();
    expect(within(list).getByText("1 face found")).toBeInTheDocument();
  });

  it("adds the capture advice under a phone photo of a print", () => {
    const analysis = makeAnalysis({
      diagnosis: {
        findings: [finding("capture", "restore.diag.phoneCapture", { glare: 1, perspective: 0 })],
        suggestedPresets: [],
        toneKind: "color",
      },
    });

    renderSummary({ analysis });

    const list = screen.getByRole("list", { name: "What the analysis found" });
    expect(within(list).getByText("Looks like a phone photo of a print (glare/perspective)")).toBeInTheDocument();
    expect(screen.getByText(/rescan the print at 600 dpi or more/)).toBeInTheDocument();
  });

  it("gives no capture advice for a plain scan", () => {
    renderSummary();

    expect(screen.queryByText(/rescan the print/)).not.toBeInTheDocument();
  });

  it("estimates the time of the selected fixes on GPU and CPU", () => {
    renderSummary();

    expect(screen.getByText("About 40 s on your GPU · about 4 min on CPU")).toBeInTheDocument();
  });

  it("calls onRestore from the summary without opening the steps", () => {
    const onRestore = vi.fn();
    renderSummary({ onRestore });

    fireEvent.click(screen.getByRole("button", { name: "Restore" }));

    expect(onRestore).toHaveBeenCalledTimes(1);
  });

  it("disables Restore while the parent says it can't run or nothing is selected", () => {
    const { unmount } = renderSummary({ canRestore: false });
    expect(screen.getByRole("button", { name: "Restore" })).toBeDisabled();
    unmount();

    renderSummary({ analysis: { ...ANALYSIS, proposedSteps: [] } });
    expect(screen.getByText("No fixes selected")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Restore" })).toBeDisabled();
  });

  it("opens the steps in the backend order and the count and time follow the toggles", () => {
    renderSummary();
    openSteps();

    const names = screen.getAllByRole("checkbox").map((box) => (box as HTMLInputElement).labels?.[0]?.textContent);
    expect(names).toEqual([
      "Remove print pattern",
      "Repair damage",
      "Remove JPEG artifacts",
      "Reduce noise",
      "Fix colors and tone",
      "Restore faces",
      "Colorize",
    ]);

    fireEvent.click(screen.getByRole("checkbox", { name: "Reduce noise" }));

    expect(screen.getByText("1 fix selected")).toBeInTheDocument();
    expect(screen.getByText("About 10 s on your GPU · about 60 s on CPU")).toBeInTheDocument();
    expect(screen.getByText("You changed this preset's fixes.")).toBeInTheDocument();
  });

  it("switches preset and marks the suggested one", () => {
    renderSummary();
    openSteps();

    const heavy = screen.getByRole("radio", { name: "Heavy damage" });
    expect(heavy.parentElement).toHaveTextContent("Suggested");
    fireEvent.click(screen.getByRole("radio", { name: "Newspaper / magazine clipping" }));

    expect(screen.getByRole("checkbox", { name: "Remove print pattern" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "Repair damage" })).not.toBeChecked();
  });

  it("warns about overprocessing when a halftone descreen meets strong denoise", () => {
    renderSummary();
    openSteps();
    fireEvent.click(screen.getByRole("radio", { name: "Newspaper / magazine clipping" }));
    expect(screen.queryByText(/can blur it twice/)).not.toBeInTheDocument();

    const strengths = screen.getAllByRole("slider", { name: "Strength" });
    fireEvent.change(strengths[strengths.length - 1], { target: { value: "0.6" } });

    expect(screen.getByText(/Noise reduction above 30% can blur it twice/)).toBeInTheDocument();
  });

  it("keeps a step without its pack off and offers the download", () => {
    renderSummary({ capabilities: makeCapabilities(["restore-core"]) });

    expect(screen.getByText("No fixes selected")).toBeInTheDocument();
    openSteps();
    expect(screen.getByRole("checkbox", { name: "Repair damage" })).toBeDisabled();
    expect(screen.getAllByText("This fix needs the restore-core pack.").length).toBeGreaterThan(0);
  });
});
