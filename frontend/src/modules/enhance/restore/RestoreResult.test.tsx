import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { createEditorHandoffStore } from "../../../lib/editorHandoffStore";
import { createJobQueueStore } from "../../../lib/jobQueueStore";
import type { CreateRestoreJobParams } from "../../../services/restore";
import { RestoreResult } from "./RestoreResult";
import { readRestoreSummary } from "./restoreResultModel";
import { makeCompletedRestoreJob, makeRestoreMetadata } from "./restoreTestFixtures";

const BEFORE_URL = "/api/v1/restore/analysis/tok-1/preview.jpg?v=0";
const WITH_COLOR = {
  artifacts: ["preview", "view", "beforeafter", "sidecar", "uncolored"],
  downloadNames: {
    restored: "grandma_colorized.png",
    uncolored: "grandma_uncolored.png",
    before_after: "grandma_before-after.jpg",
    sidecar: "grandma_restore.json",
  },
  compositeReasons: ["colorize"],
};

const WITH_FACES = {
  compositeReasons: ["faces"],
  recomposeAvailable: true,
  faces: [{ index: 0, enabled: true, restored: true, blend: 0.6 }],
};

const BATCH_BASE: CreateRestoreJobParams = {
  source: { token: "tok-1" },
  steps: ["repair"],
  options: { repair: { sensitivity: 0.5 } },
  scale: 1,
  modelId: null,
  device: null,
  outputFormat: "png",
};

function renderResult(
  overrides: Record<string, unknown> = {},
  recompose = vi.fn(),
  batchBase: CreateRestoreJobParams | null = null,
) {
  const job = makeCompletedRestoreJob(makeRestoreMetadata(overrides));
  const summary = readRestoreSummary(job);
  if (summary === null) {
    throw new Error("fixture should parse");
  }
  const handoffStore = createEditorHandoffStore();
  render(
    <MemoryRouter initialEntries={["/enhance/restore"]}>
      <Routes>
        <Route
          path="/enhance/restore"
          element={
            <RestoreResult
              job={job}
              summary={summary}
              beforeUrl={BEFORE_URL}
              handoffStore={handoffStore}
              recompose={recompose}
              batchBase={batchBase}
              batchDeps={{ createJob: vi.fn(), queue: createJobQueueStore() }}
            />
          }
        />
        <Route path="/editor" element={<p>Editor page</p>} />
      </Routes>
    </MemoryRouter>,
  );
  return handoffStore;
}

function link(name: string) {
  return screen.getByRole("link", { name });
}

describe("RestoreResult", () => {
  it("compares the working copy against the viewing copy of the result", () => {
    renderResult();
    expect(screen.getByAltText("Before: grandma.jpg")).toHaveAttribute("src", BEFORE_URL);
    expect(screen.getByAltText("After: grandma.jpg")).toHaveAttribute("src", "/api/v1/jobs/job-1/artifacts/view");
    expect(screen.getByRole("button", { name: "100%" })).toBeInTheDocument();
  });

  it("labels the zoom Preview when the viewing copy is reduced", () => {
    renderResult({ viewFullResolution: false });
    expect(screen.queryByRole("button", { name: "100%" })).not.toBeInTheDocument();
    expect(screen.getByRole("status", { name: "Zoom level" })).toHaveTextContent("Preview");
  });

  it("offers readable download names", () => {
    renderResult();
    expect(link("Download restored")).toHaveAttribute("href", "/api/v1/jobs/job-1/download");
    expect(link("Download restored")).toHaveAttribute("download", "grandma_restored.png");
    expect(link("Download before/after")).toHaveAttribute("href", "/api/v1/jobs/job-1/artifacts/beforeafter");
    expect(link("Download before/after")).toHaveAttribute("download", "grandma_before-after.jpg");
    expect(link("Restoration details (JSON)")).toHaveAttribute("href", "/api/v1/jobs/job-1/artifacts/sidecar");
    expect(link("Restoration details (JSON)")).toHaveAttribute("download", "grandma_restore.json");
    expect(screen.queryByRole("link", { name: "Download uncolored" })).not.toBeInTheDocument();
  });

  it("names a colorized result and offers the uncolored version", () => {
    renderResult(WITH_COLOR);
    expect(link("Download restored")).toHaveAttribute("download", "grandma_colorized.png");
    expect(link("Download uncolored")).toHaveAttribute("href", "/api/v1/jobs/job-1/artifacts/uncolored");
    expect(link("Download uncolored")).toHaveAttribute("download", "grandma_uncolored.png");
  });

  it("switches the comparison to the uncolored version", () => {
    renderResult(WITH_COLOR);
    fireEvent.click(screen.getByRole("checkbox", { name: "Show uncolored" }));
    expect(screen.getByAltText("After: grandma.jpg")).toHaveAttribute("src", "/api/v1/jobs/job-1/artifacts/uncolored");
    fireEvent.click(screen.getByRole("checkbox", { name: "Show uncolored" }));
    expect(screen.getByAltText("After: grandma.jpg")).toHaveAttribute("src", "/api/v1/jobs/job-1/artifacts/view");
  });

  it("has no uncolored switch without colorization", () => {
    renderResult();
    expect(screen.queryByRole("checkbox", { name: "Show uncolored" })).not.toBeInTheDocument();
  });

  it("warns about AI reconstructions only when faces, colors or large fills were involved", () => {
    renderResult({ compositeReasons: ["faces"] });
    expect(screen.getByRole("note")).toHaveTextContent(
      "Restored faces and filled areas are AI reconstructions and may not match how the person actually looked. Colors are AI guesses, not the real colors.",
    );
  });

  it("has no info card for a faithful restoration", () => {
    renderResult();
    expect(screen.queryByRole("note")).not.toBeInTheDocument();
  });

  it("opens the restored photo in the Editor", () => {
    const handoffStore = renderResult();
    fireEvent.click(screen.getByRole("button", { name: "Open in Editor" }));
    expect(screen.getByText("Editor page")).toBeInTheDocument();
    expect(handoffStore.getSnapshot()).toEqual({
      url: "/api/v1/jobs/job-1/download",
      fileName: "grandma_restored.png",
    });
  });

  it("says the photo never left the computer", () => {
    renderResult();
    expect(screen.getByText("Runs on your computer. Your photos are not uploaded anywhere.")).toBeInTheDocument();
  });

  it("offers to apply the same settings to more photos", () => {
    renderResult({}, vi.fn(), BATCH_BASE);
    expect(screen.getByLabelText("Apply these settings to more photos")).toHaveAttribute("type", "file");
  });

  it("has no batch without the settings that produced the result", () => {
    renderResult();
    expect(screen.queryByLabelText("Apply these settings to more photos")).not.toBeInTheDocument();
  });

  it("has no batch when the photo only restored faces", () => {
    renderResult({}, vi.fn(), { ...BATCH_BASE, steps: ["faces"] });
    expect(screen.queryByLabelText("Apply these settings to more photos")).not.toBeInTheDocument();
  });

  it("has no face controls when no face was restored", () => {
    renderResult();
    expect(screen.queryByRole("heading", { name: "Faces, one by one" })).not.toBeInTheDocument();
  });

  it("recomposes the faces and shows the rebuilt photo and downloads", async () => {
    const recompose = vi.fn().mockResolvedValue({
      sidecar: { faces: [{ index: 0, enabled: false, restored: true, blend: 0.6 }], compositeReasons: [] },
    });
    renderResult(WITH_FACES, recompose);
    expect(screen.getByRole("note")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("checkbox", { name: "Show original face" }));
    fireEvent.click(screen.getByRole("button", { name: "Apply" }));

    await waitFor(() =>
      expect(screen.getByAltText("After: grandma.jpg")).toHaveAttribute("src", "/api/v1/jobs/job-1/artifacts/view?v=1"),
    );
    expect(recompose).toHaveBeenCalledWith("job-1", { 0: { enabled: false, blend: 0.6 } });
    expect(link("Download restored")).toHaveAttribute("href", "/api/v1/jobs/job-1/download?v=1");
    expect(link("Download before/after")).toHaveAttribute("href", "/api/v1/jobs/job-1/artifacts/beforeafter?v=1");
    expect(screen.queryByRole("note")).not.toBeInTheDocument();
    expect(screen.getByText("Faces updated.")).toBeInTheDocument();
  });
});
