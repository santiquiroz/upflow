import { fireEvent, render, screen } from "@testing-library/react";
import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { en } from "../../i18n/en";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import type { VoiceCatalog } from "../../lib/apiTypes";
import { VoiceChainPanel, voiceSummaryKey } from "./VoiceChainPanel";
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

const ORDER = [
  "denoise",
  "highpass",
  "compress",
  "presence",
  "deesser",
  "loudness",
] as const;

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

function Harness({ catalog = CATALOG }: { catalog?: VoiceCatalog }) {
  const selection = useVoiceSelection(catalog);
  return (
    <VoiceChainPanel
      catalog={catalog}
      isLoading={false}
      isError={false}
      selection={selection}
    />
  );
}

function renderPanel(catalog: VoiceCatalog = CATALOG) {
  return render(
    <LocaleProvider>
      <Harness catalog={catalog} />
    </LocaleProvider>,
  );
}

/** El primer checkbox es el opt-in maestro; los 6 siguientes son los pasos. */
function activate() {
  // El nombre accesible del maestro es su etiqueta MAS el parrafo de ayuda, que
  // comparten el mismo <label>: por eso se busca por substring y no por igualdad.
  fireEvent.click(
    screen.getByRole("checkbox", { name: new RegExp(en["voice.activate"]) }),
  );
}

function stepBoxes(): HTMLInputElement[] {
  return (screen.getAllByRole("checkbox") as HTMLInputElement[]).slice(1);
}

function emptySelection() {
  const { result } = renderHook(() => useVoiceSelection(undefined));
  return result.current;
}

describe("VoiceChainPanel", () => {
  it("shows every step of the chain in the order the backend sent", () => {
    // El orden es la informacion principal del panel: la cadena tiene
    // causalidad y presentarla desordenada la volveria una lista de opciones
    // sueltas.
    renderPanel();
    expect(stepBoxes()).toHaveLength(6);

    // Se asserta contra el catalogo, no contra copia literal: una clave mal
    // escrita rinde la clave cruda, que nunca es igual al valor del catalogo.
    const expected = ORDER.map((id) => en[`voice.step.${id}.label` as keyof typeof en]);
    const rendered = expected.map((copy) => screen.getByText(copy));
    expect(rendered.map((node) => node.textContent)).toEqual(expected);
  });

  it("keeps every description visible without hovering", () => {
    // Usar hover como unico mecanismo para la informacion que hace falta para
    // decidir es un fallo de accesibilidad, y este panel lo usa gente que nunca
    // vio un de-esser.
    renderPanel();
    for (const id of ORDER) {
      const copy = en[`voice.step.${id}.description` as keyof typeof en];
      expect(screen.getByText(copy)).toBeVisible();
    }
  });

  it("shows the steps but does not let them be chosen while it is off", () => {
    // Se ven para que se descubra que existen; se bloquean para que nadie
    // termine con procesado de voz sin haberlo pedido.
    renderPanel();
    for (const box of stepBoxes()) {
      expect(box).toBeDisabled();
      expect(box.checked).toBe(false);
    }
  });

  it("unlocks the steps once it is turned on", () => {
    renderPanel();
    activate();
    for (const box of stepBoxes()) {
      expect(box).toBeEnabled();
    }
  });

  it("starts with the backend defaults checked", () => {
    renderPanel();
    activate();
    const boxes = stepBoxes();
    // denoise, presence y loudness vienen apagados; los otros tres encendidos.
    expect(boxes.filter((box) => box.checked)).toHaveLength(3);
  });

  it("leaves a disabled step in place instead of hiding it", () => {
    // Si desapareciera, el orden dejaria de ser legible y no se veria donde
    // encajaria al volver a encenderlo.
    renderPanel();
    activate();
    fireEvent.click(stepBoxes()[1]);
    expect(stepBoxes()).toHaveLength(6);
    expect(screen.getByText(en["voice.step.highpass.label"])).toBeVisible();
  });

  it("hides the delivery target until loudness is on", () => {
    renderPanel();
    expect(screen.queryByText(en["voice.delivery.legend"])).not.toBeInTheDocument();

    activate();
    fireEvent.click(stepBoxes()[5]);
    expect(screen.getByText(en["voice.delivery.legend"])).toBeVisible();
  });

  it("shows each delivery target with its published numbers", () => {
    renderPanel();
    activate();
    fireEvent.click(stepBoxes()[5]);
    expect(screen.getByText("-14 LUFS · -1 dBTP")).toBeVisible();
    expect(screen.getByText("-23 LUFS · -1 dBTP")).toBeVisible();
  });

  it("clicking a delivery target does not toggle the step it is nested in", () => {
    // El config anidado quedo FUERA del label del checkbox justamente por esto.
    renderPanel();
    activate();
    const loudness = stepBoxes()[5];
    fireEvent.click(loudness);
    expect(loudness.checked).toBe(true);

    fireEvent.click(screen.getByRole("radio", { name: /EBU R128/ }));
    expect(loudness.checked).toBe(true);
  });

  it("hides the presence amount until the presence step is on", () => {
    renderPanel();
    expect(screen.queryByRole("slider")).not.toBeInTheDocument();

    activate();
    fireEvent.click(stepBoxes()[3]);
    expect(screen.getByRole("slider")).toBeVisible();
  });

  it("names the strategy each step actually runs", () => {
    // Badge, no selector: el backend todavia no publica variantes de modelo, asi
    // que ofrecer elegir seria un control sin nada detras.
    renderPanel();
    expect(screen.getAllByText(en["voice.strategy.dsp"])).toHaveLength(6);
  });

  it("shows a loading message while the catalog is in flight", () => {
    render(
      <LocaleProvider>
        <VoiceChainPanel
          catalog={undefined}
          isLoading
          isError={false}
          selection={emptySelection()}
        />
      </LocaleProvider>,
    );
    expect(screen.getByText(en["voice.loading"])).toBeVisible();
  });

  it("explains a failed catalog instead of rendering an empty chain", () => {
    render(
      <LocaleProvider>
        <VoiceChainPanel
          catalog={undefined}
          isLoading={false}
          isError
          selection={emptySelection()}
        />
      </LocaleProvider>,
    );
    expect(screen.getByText(en["voice.loadFailed"])).toBeVisible();
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
  });
});

describe("voiceSummaryKey", () => {
  it("uses a distinct key per count shape", () => {
    // "1 paso" y "3 pasos" no comparten forma y translate() no pluraliza.
    expect(voiceSummaryKey(0)).toBe("voice.summary.none");
    expect(voiceSummaryKey(1)).toBe("voice.summary.one");
    expect(voiceSummaryKey(4)).toBe("voice.summary.many");
  });
});
