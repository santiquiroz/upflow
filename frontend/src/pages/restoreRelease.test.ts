import { describe, expect, it } from "vitest";
import { isRestoreReleased } from "./restoreRelease";
import { treeWithRestore } from "./restoreReleaseTestUtils";

describe("isRestoreReleased", () => {
  it("is released when the backend lists image.restore as a live capability", () => {
    expect(isRestoreReleased(treeWithRestore(true))).toBe(true);
  });

  it("is not released when image.restore sits in the roadmap", () => {
    expect(isRestoreReleased(treeWithRestore(false))).toBe(false);
  });

  it("stays hidden until the capability tree arrives", () => {
    expect(isRestoreReleased(undefined)).toBe(false);
  });
});
