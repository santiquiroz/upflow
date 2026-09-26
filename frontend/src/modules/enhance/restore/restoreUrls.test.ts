import { describe, expect, it } from "vitest";
import { revisedUrl, versionedUrl } from "./restoreUrls";

describe("versionedUrl", () => {
  it("adds the revision as a query parameter", () => {
    expect(versionedUrl("/a/preview.jpg", 3)).toBe("/a/preview.jpg?v=3");
    expect(versionedUrl("/a/preview.jpg?x=1", 3)).toBe("/a/preview.jpg?x=1&v=3");
  });
});

describe("revisedUrl", () => {
  it("keeps the plain URL until the result is rebuilt", () => {
    expect(revisedUrl("/api/v1/jobs/job-1/download", 0)).toBe("/api/v1/jobs/job-1/download");
    expect(revisedUrl("/api/v1/jobs/job-1/download", 2)).toBe("/api/v1/jobs/job-1/download?v=2");
  });
});
