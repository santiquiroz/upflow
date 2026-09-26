import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { readCctvFlag, useCctvModeFlag } from "./useCctvModeFlag";
import { isLiteStorageSize, looksLikeDvrExport, readFileHead, useCctvSuggestion } from "./useCctvSuggestion";

function bytes(...values: number[]) {
  return new Uint8Array(values);
}

function ascii(text: string) {
  return new TextEncoder().encode(text);
}

describe("looksLikeDvrExport", () => {
  it("recognizes Hikvision and Dahua exports by their first bytes", () => {
    expect(looksLikeDvrExport(ascii("IMKH0100"))).toBe(true);
    expect(looksLikeDvrExport(ascii("DHAV\u00fd"))).toBe(true);
  });

  it("recognizes raw H.264 and H.265 streams", () => {
    expect(looksLikeDvrExport(bytes(0, 0, 0, 1, 0x67, 0x42))).toBe(true);
    expect(looksLikeDvrExport(bytes(0, 0, 1, 0x40, 0x01))).toBe(true);
  });

  it("ignores ordinary containers, including MPEG-PS that starts with a start code", () => {
    expect(looksLikeDvrExport(bytes(0, 0, 0, 0x18, ...ascii("ftypmp42")))).toBe(false);
    expect(looksLikeDvrExport(bytes(0x1a, 0x45, 0xdf, 0xa3))).toBe(false);
    expect(looksLikeDvrExport(bytes(0, 0, 1, 0xba, 0x44))).toBe(false);
  });
});

describe("isLiteStorageSize", () => {
  it("flags the half-width 'lite' storage sizes", () => {
    expect(isLiteStorageSize({ width: 960, height: 1080 })).toBe(true);
    expect(isLiteStorageSize({ width: 640, height: 720 })).toBe(true);
    expect(isLiteStorageSize({ width: 1920, height: 1080 })).toBe(false);
    expect(isLiteStorageSize(null)).toBe(false);
  });
});

describe("useCctvSuggestion", () => {
  it("reads only the head of the file", async () => {
    const head = await readFileHead(new File([ascii("IMKH-and-a-long-tail-of-video-bytes")], "a.mp4"), 4);

    expect(Array.from(head)).toEqual(Array.from(ascii("IMKH")));
  });

  it("suggests the mode for a recorder export whatever its extension", async () => {
    const file = new File([ascii("IMKH0100")], "export.mp4");

    const { result } = renderHook(() => useCctvSuggestion(file, null));

    await waitFor(() => expect(result.current).toBe(true));
  });

  it("suggests the mode for a lite resolution even in a normal container", () => {
    const file = new File([ascii("xxxxftypmp42")], "clip.mp4");

    const { result } = renderHook(() => useCctvSuggestion(file, { width: 960, height: 1080 }));

    expect(result.current).toBe(true);
  });

  it("stays quiet for an ordinary video and without a file", async () => {
    const file = new File([bytes(0, 0, 0, 0x18, ...ascii("ftypmp42"))], "holiday.mp4");

    const withFile = renderHook(() => useCctvSuggestion(file, { width: 1920, height: 1080 }));
    const withoutFile = renderHook(() => useCctvSuggestion(null, null));

    // Una segunda lectura del mismo archivo termina despues de la del hook: ya se evaluo la cabecera.
    await act(async () => {
      await readFileHead(file);
    });
    expect(withFile.result.current).toBe(false);
    expect(withoutFile.result.current).toBe(false);
  });
});

describe("useCctvModeFlag", () => {
  afterEach(() => {
    window.history.replaceState(null, "", "/");
  });

  it("reads ?cctv=1 as the entry point", () => {
    expect(readCctvFlag("?cctv=1")).toBe(true);
    expect(readCctvFlag("?cctv=0")).toBe(false);
    expect(readCctvFlag("")).toBe(false);
  });

  it("starts on when the page was opened with ?cctv=1", () => {
    window.history.replaceState(null, "", "/enhance/video?cctv=1");

    const { result } = renderHook(() => useCctvModeFlag());

    expect(result.current[0]).toBe(true);
  });
});
