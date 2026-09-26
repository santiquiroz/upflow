import { describe, expect, it } from "vitest";
import {
  acquisitionOf,
  caseRequestOf,
  EMPTY_CASE_DETAILS,
  isCaseDetailsValid,
  isClockOffsetValid,
  parseClockOffset,
  withCaseField,
} from "./cctvCase";

describe("parseClockOffset", () => {
  it.each([
    ["12", 12],
    ["-12", -12],
    ["+3.5", 3.5],
    [" 0.25 ", 0.25],
    [".5", 0.5],
  ])("reads %s as %d seconds", (text, seconds) => {
    expect(parseClockOffset(text)).toBe(seconds);
  });

  it.each(["", "abc", "1,5", "1e3", "12 s", "--1", "Infinity"])("rejects %j", (text) => {
    expect(parseClockOffset(text)).toBeNull();
  });
});

describe("isClockOffsetValid", () => {
  it("accepts an empty offset because every case field is optional", () => {
    expect(isClockOffsetValid("  ")).toBe(true);
  });

  it("rejects a decimal comma instead of guessing what it meant", () => {
    expect(isClockOffsetValid("1,500")).toBe(false);
  });
});

describe("isCaseDetailsValid", () => {
  it("is valid when nothing is filled in", () => {
    expect(isCaseDetailsValid(EMPTY_CASE_DETAILS)).toBe(true);
  });

  it("is invalid only because of an unreadable clock offset", () => {
    expect(isCaseDetailsValid(withCaseField(EMPTY_CASE_DETAILS, "clockOffset", "two"))).toBe(false);
  });
});

describe("withCaseField", () => {
  it("returns a new object and leaves the original untouched", () => {
    const next = withCaseField(EMPTY_CASE_DETAILS, "caseLabel", "Case 42");

    expect(next.caseLabel).toBe("Case 42");
    expect(EMPTY_CASE_DETAILS.caseLabel).toBe("");
  });
});

describe("acquisitionOf", () => {
  it("sends the recorder fields trimmed and the offset as a number", () => {
    const details = {
      ...EMPTY_CASE_DETAILS,
      recorderMake: " Hikvision ",
      recorderModel: "DS-7608NI",
      channel: "CH01",
      clockOffset: "-37.5",
      clockOffsetMethod: "Photo of the DVR clock next to horalegal.inm.gov.co",
    };

    expect(acquisitionOf(details)).toEqual({
      recorderMake: "Hikvision",
      recorderModel: "DS-7608NI",
      channel: "CH01",
      clockOffsetSeconds: -37.5,
      clockOffsetMethod: "Photo of the DVR clock next to horalegal.inm.gov.co",
    });
  });

  it("leaves out blank fields", () => {
    expect(acquisitionOf({ ...EMPTY_CASE_DETAILS, channel: "   " })).toEqual({});
  });

  it("keeps a zero offset, which is a real measurement", () => {
    expect(acquisitionOf({ ...EMPTY_CASE_DETAILS, clockOffset: "0" })).toEqual({ clockOffsetSeconds: 0 });
  });
});

describe("caseRequestOf", () => {
  it("adds nothing to the job request when the form is empty", () => {
    expect(caseRequestOf(EMPTY_CASE_DETAILS)).toEqual({});
  });

  it("sends the case label, the operator and the acquisition", () => {
    const details = { ...EMPTY_CASE_DETAILS, caseLabel: " 2026-114 ", operatorName: "S. Q.", channel: "2" };

    expect(caseRequestOf(details)).toEqual({
      caseLabel: "2026-114",
      operatorName: "S. Q.",
      acquisition: { channel: "2" },
    });
  });
});
