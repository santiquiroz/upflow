import { describe, expect, it } from "vitest";
import { isTerminalJobStatus } from "./jobStatus";

describe("isTerminalJobStatus", () => {
  it.each([
    ["queued", false],
    ["running", false],
    ["completed", true],
    ["failed", true],
  ] as const)("returns %s -> %s", (status, expected) => {
    expect(isTerminalJobStatus(status)).toBe(expected);
  });
});
