import { describe, expect, it } from "vitest";
import type { JobResponse } from "../../../lib/apiTypes";
import {
  batchRowState,
  countToReview,
  faceCountKey,
  needsFaceReview,
  ranFaces,
  type BatchRowState,
} from "./batchResultModel";
import { makeCompletedRestoreJob, makeRestoreMetadata } from "./restoreTestFixtures";

function face(index: number, enabled = true) {
  return { index, enabled, restored: true, blend: 0.6 };
}

function doneWith(faces: unknown[], jobId = "job-1"): JobResponse {
  return { ...makeCompletedRestoreJob(makeRestoreMetadata({ faces, batch: true })), jobId };
}

function doneState(faces: unknown[], jobId = "job-1") {
  const state = batchRowState(doneWith(faces, jobId));
  if (state.kind !== "done") {
    throw new Error("fixture should parse");
  }
  return state;
}

describe("batchRowState", () => {
  it("waits while the job is unknown, queued or running", () => {
    expect(batchRowState(undefined).kind).toBe("waiting");
    expect(batchRowState({ ...doneWith([]), status: "running" }).kind).toBe("waiting");
  });

  it("ends without a review when the job failed", () => {
    expect(batchRowState({ ...doneWith([]), status: "failed", metadata: {} }).kind).toBe("ended");
  });

  it("reads the restored faces of a finished photo", () => {
    expect(doneState([face(0), face(1, false)]).summary.faces).toHaveLength(2);
  });
});

describe("ranFaces", () => {
  it("tells a batch with faces from one without them", () => {
    expect(ranFaces({ ...doneWith([]), restoreSteps: ["repair", "faces"] })).toBe(true);
    expect(ranFaces({ ...doneWith([]), restoreSteps: ["repair"] })).toBe(false);
    expect(ranFaces({ ...doneWith([]), restoreSteps: undefined })).toBe(false);
  });
});

describe("faceCountKey", () => {
  it("names how many faces are on, off or missing", () => {
    expect(faceCountKey(doneState([]).summary)).toBe("restore.batch.results.noFaces");
    expect(faceCountKey(doneState([face(0)]).summary)).toBe("restore.batch.results.facesRestoredOne");
    expect(faceCountKey(doneState([face(0), face(1)]).summary)).toBe("restore.batch.results.facesRestored");
    expect(faceCountKey(doneState([face(0, false)]).summary)).toBe("restore.batch.results.facesOff");
  });
});

describe("needsFaceReview", () => {
  it("asks for a review only for finished photos with restored faces not reviewed yet", () => {
    expect(needsFaceReview(doneState([face(0)]), false)).toBe(true);
    expect(needsFaceReview(doneState([face(0)]), true)).toBe(false);
    expect(needsFaceReview(doneState([]), false)).toBe(false);
    expect(needsFaceReview(batchRowState(undefined), false)).toBe(false);
  });
});

describe("countToReview", () => {
  it("counts the photos whose faces nobody looked at", () => {
    const states: BatchRowState[] = [
      doneState([face(0)], "job-a"),
      doneState([face(0)], "job-b"),
      doneState([], "job-c"),
      batchRowState(undefined),
    ];

    expect(countToReview(states, new Set(["job-b"]))).toBe(1);
  });
});
