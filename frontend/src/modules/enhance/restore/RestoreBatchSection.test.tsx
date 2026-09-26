import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { JobResponse } from "../../../lib/apiTypes";
import { createJobQueueStore } from "../../../lib/jobQueueStore";
import type { CreateRestoreJobParams } from "../../../services/restore";
import { RestoreBatchSection } from "./RestoreBatchSection";

const BASE: CreateRestoreJobParams = {
  source: { token: "tok-1" },
  steps: ["repair", "faces"],
  options: { repair: { sensitivity: 0.5, use_user_mask: true }, faces: { blend: 0.6 } },
  scale: 1,
  modelId: null,
  device: null,
  outputFormat: "png",
};

function photo(name: string): File {
  return new File(["x"], name, { type: "image/jpeg" });
}

function renderSection(base: CreateRestoreJobParams = BASE, createJob = vi.fn()) {
  const queue = createJobQueueStore();
  render(<RestoreBatchSection base={base} deps={{ createJob, queue }} />);
  return { createJob, queue };
}

function choose(files: File[]) {
  fireEvent.change(screen.getByLabelText("Apply these settings to more photos"), { target: { files } });
}

describe("RestoreBatchSection", () => {
  it("sends every chosen photo with the same settings, without faces or the painted mask", async () => {
    const createJob = vi
      .fn()
      .mockResolvedValueOnce({ jobId: "job-a", status: "queued" } as JobResponse)
      .mockResolvedValueOnce({ jobId: "job-b", status: "queued" } as JobResponse);
    const { queue } = renderSection(BASE, createJob);

    choose([photo("a.jpg"), photo("b.jpg")]);

    expect(await screen.findByText("2 photos added to the job queue.")).toBeInTheDocument();
    expect(createJob).toHaveBeenCalledTimes(2);
    expect(createJob.mock.calls[0][0]).toEqual({
      ...BASE,
      source: { file: expect.any(File) },
      steps: ["repair"],
      options: { repair: { sensitivity: 0.5 } },
    });
    expect(queue.getSnapshot()).toHaveLength(2);
  });

  it("says faces are skipped only when the photo restored faces", () => {
    renderSection();
    expect(screen.getByText(/Faces are not restored in a batch/)).toBeInTheDocument();
  });

  it("does not mention faces when they were off", () => {
    renderSection({ ...BASE, steps: ["repair"] });
    expect(screen.queryByText(/Faces are not restored in a batch/)).not.toBeInTheDocument();
  });

  it("says how many photos could not be sent", async () => {
    const createJob = vi.fn().mockRejectedValue(new Error("Queue full"));
    renderSection(BASE, createJob);

    choose([photo("a.jpg")]);

    expect(await screen.findByRole("alert")).toHaveTextContent("1 photo could not be sent");
  });

  it("ignores an empty choice", () => {
    const { createJob } = renderSection();

    choose([]);

    expect(createJob).not.toHaveBeenCalled();
  });
});
