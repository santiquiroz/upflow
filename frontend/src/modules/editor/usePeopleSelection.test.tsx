import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { PropsWithChildren } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { InitImageResponse } from "../../lib/apiTypes";
import * as editorService from "../../services/editor";
import type { EditorPerson } from "../../services/editor";
import { usePeopleSelection } from "./usePeopleSelection";

const PERSON: EditorPerson = {
  id: 7,
  box: [10, 20, 100, 200],
  score: 0.9,
  role: "subject",
  face: null,
  anchor: [50, 100],
};
const FIRST_IMAGE: InitImageResponse = {
  initImageToken: "first",
  originalFilename: "first.png",
  width: 640,
  height: 480,
};
const SECOND_IMAGE: InitImageResponse = {
  ...FIRST_IMAGE,
  initImageToken: "second",
  originalFilename: "second.png",
};

afterEach(() => {
  vi.restoreAllMocks();
});

describe("usePeopleSelection", () => {
  it("initializes choices from detection and resets them when the image changes", async () => {
    const detectSpy = vi.spyOn(editorService, "detectPeople").mockResolvedValue([PERSON]);
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const wrapper = ({ children }: PropsWithChildren) => (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
    const { result, rerender } = renderHook(
      ({ baseInfo }: { baseInfo: InitImageResponse }) => usePeopleSelection(baseInfo, true),
      { initialProps: { baseInfo: FIRST_IMAGE }, wrapper },
    );

    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current.choices[PERSON.id]).toBe("keep");
    act(() => {
      result.current.toggle(PERSON.id);
      result.current.setProtectKept(false);
    });
    expect(result.current.choices[PERSON.id]).toBe("remove");
    expect(result.current.protectKept).toBe(false);

    rerender({ baseInfo: SECOND_IMAGE });
    await waitFor(() => expect(detectSpy).toHaveBeenLastCalledWith("second"));
    await waitFor(() => expect(result.current.status).toBe("ready"));
    expect(result.current.choices[PERSON.id]).toBe("keep");
    expect(result.current.protectKept).toBe(true);
  });
});
