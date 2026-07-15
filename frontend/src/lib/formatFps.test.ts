import { describe, expect, it } from "vitest";
import { formatFps } from "./formatFps";

describe("formatFps", () => {
  it("normalizes an integer fraction to a whole number string", () => {
    expect(formatFps("60/1")).toBe("60");
  });

  it("normalizes a non-integer fraction to two decimal places", () => {
    expect(formatFps("24000/1001")).toBe("23.98");
  });

  it("passes through a value with no slash unchanged", () => {
    expect(formatFps("30")).toBe("30");
  });

  it("returns the raw value unchanged when the denominator is zero", () => {
    expect(formatFps("60/0")).toBe("60/0");
  });

  it("returns the raw value unchanged when given null", () => {
    expect(formatFps(null)).toBeNull();
  });

  it("returns the raw value unchanged when given undefined", () => {
    expect(formatFps(undefined)).toBeUndefined();
  });
});
