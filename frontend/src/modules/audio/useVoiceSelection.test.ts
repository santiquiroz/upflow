import { act } from "@testing-library/react";
import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { VoiceCatalog } from "../../lib/apiTypes";
import { useVoiceSelection } from "./useVoiceSelection";

function step(id: string, defaultEnabled: boolean) {
  return {
    id,
    labelKey: `voice.step.${id}.label`,
    descriptionKey: `voice.step.${id}.description`,
    kind: "filter",
    defaultEnabled,
  };
}

const CATALOG: VoiceCatalog = {
  steps: [
    step("denoise", false),
    step("highpass", true),
    step("compress", true),
    step("presence", false),
    step("deesser", true),
    step("loudness", false),
  ],
  deliveries: [
    {
      id: "streaming",
      labelKey: "voice.delivery.streaming.label",
      descriptionKey: "voice.delivery.streaming.description",
      lufs: -14,
      truePeakDb: -1,
    },
    {
      id: "ebu_r128",
      labelKey: "voice.delivery.ebu_r128.label",
      descriptionKey: "voice.delivery.ebu_r128.description",
      lufs: -23,
      truePeakDb: -1,
    },
  ],
};

describe("useVoiceSelection", () => {
  it("starts off entirely, even though the catalog marks defaults", () => {
    // `defaultEnabled` describe la forma sensata de la cadena una vez que
    // alguien la pide, no que este pedida: encenderla sola aplicaria procesado
    // de voz a un job donde solo se pidio quitar ruido.
    const { result } = renderHook(() => useVoiceSelection(CATALOG));
    expect(result.current.active).toBe(false);
    expect(result.current.enabledIds).toEqual([]);
  });

  it("adopts the backend defaults once it is turned on", () => {
    const { result } = renderHook(() => useVoiceSelection(CATALOG));
    act(() => result.current.setActive(true));
    expect(result.current.enabledIds.sort()).toEqual(
      ["compress", "deesser", "highpass"].sort(),
    );
  });

  it("drops every step again when it is turned off", () => {
    const { result } = renderHook(() => useVoiceSelection(CATALOG));
    act(() => result.current.setActive(true));
    act(() => result.current.toggleStep("denoise", true));
    act(() => result.current.setActive(false));
    expect(result.current.enabledIds).toEqual([]);
  });

  it("has nothing enabled while the catalog has not arrived", () => {
    const { result } = renderHook(() => useVoiceSelection(undefined));
    act(() => result.current.setActive(true));
    expect(result.current.enabledIds).toEqual([]);
  });

  it("adopts the defaults when the catalog arrives late", () => {
    // El catalogo llega async. Se deriva en vez de inicializar con un efecto,
    // justamente para que este caso no dependa de un orden de renders.
    const { result, rerender } = renderHook(
      ({ catalog }: { catalog: VoiceCatalog | undefined }) => useVoiceSelection(catalog),
      { initialProps: { catalog: undefined as VoiceCatalog | undefined } },
    );
    act(() => result.current.setActive(true));
    expect(result.current.enabledIds).toEqual([]);

    rerender({ catalog: CATALOG });
    expect(result.current.enabledIds).toContain("highpass");
  });

  it("keeps the user's choice when the catalog refetches", () => {
    const { result, rerender } = renderHook(
      ({ catalog }: { catalog: VoiceCatalog | undefined }) => useVoiceSelection(catalog),
      { initialProps: { catalog: CATALOG as VoiceCatalog | undefined } },
    );
    act(() => result.current.setActive(true));
    act(() => result.current.toggleStep("highpass", false));
    expect(result.current.isEnabled("highpass")).toBe(false);

    // Un efecto de inicializacion volveria a encender highpass aca. Ese era el
    // motivo de derivar en vez de sincronizar.
    rerender({ catalog: { ...CATALOG } });
    expect(result.current.isEnabled("highpass")).toBe(false);
  });

  it("toggles a step on and off", () => {
    const { result } = renderHook(() => useVoiceSelection(CATALOG));
    act(() => result.current.setActive(true));
    act(() => result.current.toggleStep("denoise", true));
    expect(result.current.isEnabled("denoise")).toBe(true);

    act(() => result.current.toggleStep("denoise", false));
    expect(result.current.isEnabled("denoise")).toBe(false);
  });

  it("does not duplicate a step enabled twice", () => {
    const { result } = renderHook(() => useVoiceSelection(CATALOG));
    act(() => result.current.setActive(true));
    act(() => result.current.toggleStep("denoise", true));
    act(() => result.current.toggleStep("denoise", true));
    expect(result.current.enabledIds.filter((id) => id === "denoise")).toHaveLength(1);
  });

  it("preselects the most common delivery target when loudness turns on", () => {
    // El backend rechaza loudness sin destino, asi que encenderlo no puede dejar
    // la seleccion en un estado que ya se sabe invalido.
    const { result } = renderHook(() => useVoiceSelection(CATALOG));
    act(() => result.current.setActive(true));
    act(() => result.current.toggleStep("loudness", true));
    expect(result.current.delivery).toBe("streaming");
    expect(result.current.needsDelivery).toBe(false);
  });

  it("respects a delivery the user already picked", () => {
    const { result } = renderHook(() => useVoiceSelection(CATALOG));
    act(() => result.current.setActive(true));
    act(() => result.current.setDelivery("ebu_r128"));
    act(() => result.current.toggleStep("loudness", true));
    expect(result.current.delivery).toBe("ebu_r128");
  });

  it("reports no delivery while loudness is off", () => {
    // Mandar un destino sin el paso que lo usa haria creer al backend que hay
    // que ajustar loudness.
    const { result } = renderHook(() => useVoiceSelection(CATALOG));
    act(() => result.current.setActive(true));
    act(() => result.current.setDelivery("streaming"));
    expect(result.current.delivery).toBeNull();
    expect(result.current.needsDelivery).toBe(false);
  });

  it("carries the presence amount", () => {
    const { result } = renderHook(() => useVoiceSelection(CATALOG));
    act(() => result.current.setPresenceDb(4.5));
    expect(result.current.presenceDb).toBe(4.5);
  });
});
