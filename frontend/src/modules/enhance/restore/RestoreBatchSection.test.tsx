import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { JobResponse } from "../../../lib/apiTypes";
import { createJobQueueStore } from "../../../lib/jobQueueStore";
import type { CreateRestoreJobParams } from "../../../services/restore";
import { RestoreBatchSection } from "./RestoreBatchSection";
import { makeCompletedRestoreJob, makeRestoreMetadata } from "./restoreTestFixtures";
import type { BatchResultDeps } from "./useBatchJobs";

const BASE: CreateRestoreJobParams = {
  source: { token: "tok-1" },
  steps: ["repair", "faces"],
  options: {
    repair: { sensitivity: 0.5, use_user_mask: true },
    faces: { blend: 0.6, selected: [0], per_face: { 0: 0.6 } },
  },
  scale: 1,
  modelId: null,
  device: null,
  outputFormat: "png",
};

const ONE_FACE = { faces: [{ index: 0, enabled: true, restored: true, blend: 0.6 }], recomposeAvailable: true };

function photo(name: string): File {
  return new File(["x"], name, { type: "image/jpeg" });
}

function queued(jobId: string): JobResponse {
  return { jobId, status: "queued" } as JobResponse;
}

function finished(jobId: string, fileName: string, restore: Record<string, unknown> = {}): JobResponse {
  return {
    ...makeCompletedRestoreJob(makeRestoreMetadata({ batch: true, ...restore })),
    jobId,
    originalFilename: fileName,
    downloadUrl: `/api/v1/jobs/${jobId}/download`,
    restoreSteps: ["repair", "faces"],
  };
}

interface RenderOptions {
  base?: CreateRestoreJobParams;
  createJob?: ReturnType<typeof vi.fn>;
  fetchJob?: BatchResultDeps["fetchJob"];
  recompose?: BatchResultDeps["recompose"];
}

function renderSection({ base = BASE, createJob = vi.fn(), fetchJob, recompose = vi.fn() }: RenderOptions = {}) {
  const queue = createJobQueueStore();
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const resultDeps: BatchResultDeps = {
    fetchJob: fetchJob ?? ((jobId) => Promise.resolve(queued(jobId))),
    pollIntervalMs: 10,
    recompose,
  };
  render(
    <QueryClientProvider client={queryClient}>
      <RestoreBatchSection base={base} deps={{ createJob, queue }} resultDeps={resultDeps} />
    </QueryClientProvider>,
  );
  return { createJob, queue, recompose };
}

function choose(files: File[]) {
  fireEvent.change(screen.getByLabelText("Apply these settings to more photos"), { target: { files } });
}

function turnFacesOn() {
  fireEvent.click(screen.getByLabelText("Also restore faces in these photos"));
}

describe("RestoreBatchSection", () => {
  it("sends every chosen photo with the same settings, without faces or the painted mask", async () => {
    const createJob = vi.fn().mockResolvedValueOnce(queued("job-a")).mockResolvedValueOnce(queued("job-b"));
    const { queue } = renderSection({ createJob });

    choose([photo("a.jpg"), photo("b.jpg")]);

    expect(await screen.findByText("2 photos added to the job queue.")).toBeInTheDocument();
    expect(createJob).toHaveBeenCalledTimes(2);
    expect(createJob.mock.calls[0][0]).toEqual({
      ...BASE,
      source: { file: expect.any(File) },
      steps: ["repair"],
      options: { repair: { sensitivity: 0.5 }, batch: true },
    });
    expect(queue.getSnapshot()).toHaveLength(2);
  });

  it("restores faces in the batch only when asked, with the shared face settings", async () => {
    const createJob = vi.fn().mockResolvedValue(queued("job-a"));
    renderSection({ createJob });
    turnFacesOn();

    choose([photo("a.jpg")]);

    expect(await screen.findByText("1 photo added to the job queue.")).toBeInTheDocument();
    expect(createJob.mock.calls[0][0]).toMatchObject({
      steps: ["repair", "faces"],
      options: { repair: { sensitivity: 0.5 }, faces: { blend: 0.6 }, batch: true },
    });
  });

  it("warns about AI faces and asks for a review once faces are on", () => {
    renderSection();
    expect(screen.getByText("Faces are left as they are unless you turn them on.")).toBeInTheDocument();

    turnFacesOn();

    expect(screen.getByText(/only large, blurry ones are restored/)).toBeInTheDocument();
    expect(screen.getByText("AI-generated facial detail. It can change how a person looks.")).toBeInTheDocument();
  });

  it("does not offer faces when the photo did not restore any", () => {
    renderSection({ base: { ...BASE, steps: ["repair"] } });
    expect(screen.queryByLabelText("Also restore faces in these photos")).not.toBeInTheDocument();
  });

  it("lists each photo of the batch with its progress", async () => {
    const createJob = vi.fn().mockResolvedValueOnce(queued("job-a")).mockResolvedValueOnce(queued("job-b"));
    renderSection({ createJob });

    choose([photo("a.jpg"), photo("b.jpg")]);

    const list = await screen.findByRole("region", { name: "Photos in this batch" });
    expect(await within(list).findByText("a.jpg")).toBeInTheDocument();
    expect(within(list).getByText("b.jpg")).toBeInTheDocument();
    expect(within(list).getAllByText("Queued")).toHaveLength(2);
  });

  it("asks to review each result with restored faces and marks it reviewed once opened", async () => {
    const createJob = vi.fn().mockResolvedValueOnce(queued("job-a")).mockResolvedValueOnce(queued("job-b"));
    const fetchJob = vi.fn((jobId: string) =>
      Promise.resolve(jobId === "job-a" ? finished("job-a", "a.jpg", ONE_FACE) : finished("job-b", "b.jpg")),
    );
    renderSection({ createJob, fetchJob });
    turnFacesOn();

    choose([photo("a.jpg"), photo("b.jpg")]);

    expect(await screen.findByText("1 photo has restored faces to review")).toBeInTheDocument();
    expect(screen.getByText("1 face restored")).toBeInTheDocument();
    expect(screen.getByText("No faces restored")).toBeInTheDocument();
    expect(screen.getByText("Not reviewed")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Review faces" }));

    expect(screen.getByRole("heading", { name: "Faces, one by one" })).toBeInTheDocument();
    expect(screen.getByText("Reviewed")).toBeInTheDocument();
    expect(screen.getByText("All restored faces were reviewed")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Hide faces" })).toHaveAttribute("aria-expanded", "true");
  });

  it("recomposes a face of one batch result without touching the others", async () => {
    const createJob = vi.fn().mockResolvedValue(queued("job-a"));
    const fetchJob = vi.fn(() => Promise.resolve(finished("job-a", "a.jpg", ONE_FACE)));
    const recompose = vi.fn().mockResolvedValue({
      sidecar: { faces: [{ index: 0, enabled: false, restored: true, blend: 0.6 }], compositeReasons: [] },
    });
    renderSection({ createJob, fetchJob, recompose });
    turnFacesOn();
    choose([photo("a.jpg")]);
    fireEvent.click(await screen.findByRole("button", { name: "Review faces" }));

    fireEvent.click(screen.getByLabelText("Show original face"));
    fireEvent.click(screen.getByRole("button", { name: "Apply" }));

    expect(await screen.findByText("Restored faces turned off")).toBeInTheDocument();
    expect(recompose).toHaveBeenCalledWith("job-a", { 0: { enabled: false, blend: 0.6 } });
    expect(screen.getByRole("link", { name: "Download restored" })).toHaveAttribute(
      "href",
      "/api/v1/jobs/job-a/download?v=1",
    );
  });

  it("shows a finished batch without faces as completed, with no review", async () => {
    const createJob = vi.fn().mockResolvedValue(queued("job-a"));
    const fetchJob = vi.fn(() => Promise.resolve({ ...finished("job-a", "a.jpg"), restoreSteps: ["repair"] }));
    renderSection({ createJob, fetchJob });

    choose([photo("a.jpg")]);

    expect(await screen.findByText("Completed")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Restored a.jpg" })).toHaveAttribute(
      "src",
      "/api/v1/jobs/job-a/artifacts/preview",
    );
    expect(screen.queryByRole("button", { name: "Review faces" })).not.toBeInTheDocument();
    expect(screen.queryByText(/to review/)).not.toBeInTheDocument();
  });

  it("says how many photos could not be sent", async () => {
    const createJob = vi.fn().mockRejectedValue(new Error("Queue full"));
    renderSection({ createJob });

    choose([photo("a.jpg")]);

    expect(await screen.findByRole("alert")).toHaveTextContent("1 photo could not be sent");
    expect(screen.queryByRole("region", { name: "Photos in this batch" })).not.toBeInTheDocument();
  });

  it("ignores an empty choice", () => {
    const { createJob } = renderSection();

    choose([]);

    expect(createJob).not.toHaveBeenCalled();
  });
});
