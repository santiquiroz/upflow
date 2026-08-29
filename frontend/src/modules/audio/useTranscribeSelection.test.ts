import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { SeparationModel } from "../../lib/apiTypes";
import { useTranscribeSelection } from "./useTranscribeSelection";

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
const SOLO_BATERIA = modelo("drum_only", ["drums"]);

function setup(model: SeparationModel | undefined = CUATRO) {
  return renderHook(({ current }) => useTranscribeSelection(current), {
    initialProps: { current: model },
  });
}

describe("useTranscribeSelection", () => {
  it("starts off, opt-in like minus-one", () => {
    const { result } = setup();

    expect(result.current.active).toBe(false);
    expect(result.current.enabledStems).toEqual([]);
  });

  it("excludes drums from the transcribable stems, mirroring the backend's rejection", () => {
    const { result } = setup(CUATRO);

    expect(result.current.transcribableStemIds).toEqual(["vocals", "bass", "other"]);
  });

  it("never enables drums even if toggled on directly", () => {
    const { result } = setup(CUATRO);
    act(() => result.current.setActive(true));

    act(() => result.current.toggleStem("drums", true));

    expect(result.current.enabledStems).toEqual([]);
  });

  it("reports nothing enabled while the subsection is off, even after ticking stems", () => {
    const { result } = setup();

    act(() => result.current.toggleStem("vocals", true));

    expect(result.current.enabledStems).toEqual([]);
  });

  it("keeps the model's stem order no matter the order they were ticked", () => {
    const { result } = setup();
    act(() => result.current.setActive(true));

    act(() => result.current.toggleStem("bass", true));
    act(() => result.current.toggleStem("vocals", true));
    act(() => result.current.toggleStem("other", true));

    expect(result.current.enabledStems).toEqual(["vocals", "bass", "other"]);
  });

  it("unticking removes only that stem", () => {
    const { result } = setup();
    act(() => result.current.setActive(true));

    act(() => result.current.toggleStem("bass", true));
    act(() => result.current.toggleStem("vocals", true));
    act(() => result.current.toggleStem("bass", false));

    expect(result.current.enabledStems).toEqual(["vocals"]);
  });

  it("is available even for a two-stem model, unlike minus-one: nothing there is drums", () => {
    const { result } = setup(KARAOKE);

    expect(result.current.available).toBe(true);
    expect(result.current.transcribableStemIds).toEqual(["instrumental", "vocals"]);
  });

  it("is unavailable for a model whose only stem is drums", () => {
    const { result } = setup(SOLO_BATERIA);

    expect(result.current.available).toBe(false);
  });

  it("drops picks that do not exist in the newly selected model", () => {
    const { result, rerender } = setup();
    act(() => result.current.setActive(true));
    act(() => result.current.toggleStem("bass", true));
    act(() => result.current.toggleStem("vocals", true));

    rerender({ current: modelo("otro", ["vocals", "guitar", "piano"]) });

    expect(result.current.enabledStems).toEqual(["vocals"]);
  });

  it("survives an undefined model while capabilities are loading", () => {
    const { result } = renderHook(() => useTranscribeSelection(undefined));

    act(() => result.current.setActive(true));
    act(() => result.current.toggleStem("vocals", true));

    expect(result.current.available).toBe(false);
    expect(result.current.enabledStems).toEqual([]);
  });
});
