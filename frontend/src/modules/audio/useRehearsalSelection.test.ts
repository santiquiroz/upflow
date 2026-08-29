import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { SeparationModel } from "../../lib/apiTypes";
import { useRehearsalSelection } from "./useRehearsalSelection";

function modelo(id: string, stems: string[]): SeparationModel {
  return {
    id,
    name: id,
    installed: true,
    primaryStem: stems[0],
    category: "karaoke",
    architecture: "mdx",
    descriptionKey: `audio.karaoke.model.${id}.description`,
    stems: stems.map((stem) => ({ id: stem, labelKey: `audio.stem.${stem}` })),
  } as SeparationModel;
}

const CUATRO = modelo("umx_4stem", ["vocals", "drums", "bass", "other"]);
const KARAOKE = modelo("inst_hq_3", ["instrumental", "vocals"]);

function setup(model: SeparationModel | undefined = CUATRO) {
  return renderHook(({ current }) => useRehearsalSelection(current), {
    initialProps: { current: model },
  });
}

describe("useRehearsalSelection", () => {
  it("starts off, because minus-one is opt-in like the cleanup chain", () => {
    const { result } = setup();

    expect(result.current.active).toBe(false);
    expect(result.current.enabledStems).toEqual([]);
    expect(result.current.guidePercent).toBe(0);
  });

  it("reports nothing enabled while the section is off, even after ticking stems", () => {
    const { result } = setup();

    act(() => result.current.toggleStem("drums", true));

    expect(result.current.enabledStems).toEqual([]);
  });

  it("keeps the model's stem order no matter the order they were ticked", () => {
    const { result } = setup();
    act(() => result.current.setActive(true));

    act(() => result.current.toggleStem("bass", true));
    act(() => result.current.toggleStem("vocals", true));
    act(() => result.current.toggleStem("drums", true));

    expect(result.current.enabledStems).toEqual(["vocals", "drums", "bass"]);
  });

  it("unticking removes only that stem", () => {
    const { result } = setup();
    act(() => result.current.setActive(true));

    act(() => result.current.toggleStem("drums", true));
    act(() => result.current.toggleStem("bass", true));
    act(() => result.current.toggleStem("drums", false));

    expect(result.current.enabledStems).toEqual(["bass"]);
  });

  it("is unavailable for a two-stem model: there is no mix left to practise over", () => {
    const { result } = setup(KARAOKE);
    act(() => result.current.setActive(true));
    act(() => result.current.toggleStem("vocals", true));

    expect(result.current.available).toBe(false);
    expect(result.current.enabledStems).toEqual([]);
  });

  it("is available from three stems up", () => {
    const { result } = setup();

    expect(result.current.available).toBe(true);
  });

  it("drops picks that do not exist in the newly selected model", () => {
    const { result, rerender } = setup();
    act(() => result.current.setActive(true));
    act(() => result.current.toggleStem("drums", true));
    act(() => result.current.toggleStem("vocals", true));

    rerender({ current: modelo("otro", ["vocals", "guitar", "piano"]) });

    expect(result.current.enabledStems).toEqual(["vocals"]);
  });

  it("steps the guide percent through the fixed levels", () => {
    const { result } = setup();

    act(() => result.current.setGuidePercent(20));
    expect(result.current.guidePercent).toBe(20);

    act(() => result.current.setGuidePercent(0));
    expect(result.current.guidePercent).toBe(0);
  });

  it("ignores a percent outside the stepped scale", () => {
    // El backend valida 0-30 y el control es escalonado: un valor sin botón no
    // tiene forma legítima de llegar acá.
    const { result } = setup();
    act(() => result.current.setGuidePercent(20));

    act(() => result.current.setGuidePercent(15));

    expect(result.current.guidePercent).toBe(20);
  });

  it("survives an undefined model while capabilities are loading", () => {
    // renderHook directo: pasar undefined a setup() caeria en su default.
    const { result } = renderHook(() => useRehearsalSelection(undefined));

    act(() => result.current.setActive(true));
    act(() => result.current.toggleStem("drums", true));

    expect(result.current.available).toBe(false);
    expect(result.current.enabledStems).toEqual([]);
  });
});
