import { describe, expect, it } from "vitest";
import { formatModelSize } from "./sizeFormat";

describe("formatModelSize", () => {
  it("formats zero bytes", () => {
    expect(formatModelSize(0)).toBe("0 B");
  });

  it("formats sub-kilobyte sizes as whole bytes", () => {
    expect(formatModelSize(500)).toBe("500 B");
  });

  it("formats kilobyte sizes with one decimal", () => {
    expect(formatModelSize(1536)).toBe("1.5 KB");
  });

  it("formats megabyte sizes with one decimal", () => {
    expect(formatModelSize(5_242_880)).toBe("5.0 MB");
  });

  it("formats gigabyte sizes with one decimal", () => {
    expect(formatModelSize(2_147_483_648)).toBe("2.0 GB");
  });

  it("rounds to two significant fractional digits at most", () => {
    expect(formatModelSize(123_456_789)).toBe("117.7 MB");
  });
});
