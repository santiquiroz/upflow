import { describe, expect, it } from "vitest";
import { isTerminalInstallStatus } from "./installStatus";

describe("isTerminalInstallStatus", () => {
  it("treats installed as terminal", () => {
    expect(isTerminalInstallStatus("installed")).toBe(true);
  });

  it("treats error as terminal", () => {
    expect(isTerminalInstallStatus("error")).toBe(true);
  });

  it("treats downloading as non-terminal", () => {
    expect(isTerminalInstallStatus("downloading")).toBe(false);
  });

  it("treats validating as non-terminal", () => {
    expect(isTerminalInstallStatus("validating")).toBe(false);
  });

  it("treats converting as non-terminal", () => {
    expect(isTerminalInstallStatus("converting")).toBe(false);
  });
});
