import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { en } from "../../i18n/en";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import type {
  CapabilityResponse,
  CapabilityTreeResponse,
} from "../../lib/apiTypes";
import { CapabilityTree } from "./CapabilityTree";

function capability(
  overrides: Partial<CapabilityResponse> & Pick<CapabilityResponse, "id" | "domain" | "labelKey">,
): CapabilityResponse {
  return {
    status: "available",
    provisioning: "registry",
    jobKind: overrides.domain,
    strategies: ["model"],
    missingPacks: [],
    unavailableReasonKey: null,
    setupReasonKey: null,
    activatableSettings: [],
    ...overrides,
  };
}

const AVAILABLE = capability({
  id: "video.upscale",
  domain: "video",
  labelKey: "capability.video.upscale",
});

const NEEDS_PACK = capability({
  id: "video.interpolate",
  domain: "video",
  labelKey: "capability.video.interpolate",
  status: "needs_setup",
  provisioning: "vendored_pack",
  missingPacks: ["rife"],
  setupReasonKey: "capability.setup.missingPack",
});

// El ejemplo de "todavia no se puede" tiene que ser una capacidad que de verdad
// no este hecha. Antes era video.subtitles, que se envio (transcripcion alineada,
// muxeo y quemado) y dejo este fixture describiendo un estado inexistente.
const ROADMAP = capability({
  id: "print.slice",
  domain: "print",
  labelKey: "capability.print.slice",
  status: "not_implemented",
  provisioning: "none",
  jobKind: null,
  unavailableReasonKey: "capability.reason.noSlicerPack",
});

const NEEDS_FLAG = capability({
  id: "audio.restoreSr",
  domain: "audio",
  labelKey: "capability.audio.restoreSr",
  status: "needs_setup",
  provisioning: "vendored_pack",
  missingPacks: [],
  setupReasonKey: "capability.setup.missingSetting",
  activatableSettings: ["enable_audiosr"],
});

const NEEDS_PACK_AND_FLAG = capability({
  id: "audio.restore",
  domain: "audio",
  labelKey: "capability.audio.restore",
  status: "needs_setup",
  provisioning: "vendored_pack",
  missingPacks: ["apollo"],
  setupReasonKey: "capability.setup.missingPack",
  activatableSettings: ["enable_audio_restore"],
});

const NEEDS_MODEL = capability({
  id: "image.upscale",
  domain: "image",
  labelKey: "capability.image.upscale",
  status: "needs_setup",
  missingPacks: [],
  setupReasonKey: "capability.setup.missingModel",
});

const TREE: CapabilityTreeResponse = {
  domains: [
    {
      domain: "video",
      labelKey: "capability.domain.video",
      capabilities: [AVAILABLE, NEEDS_PACK],
      roadmap: [ROADMAP],
    },
    {
      domain: "image",
      labelKey: "capability.domain.image",
      capabilities: [NEEDS_MODEL],
      roadmap: [],
    },
    {
      domain: "audio",
      labelKey: "capability.domain.audio",
      capabilities: [
        capability({
          id: "audio.denoise",
          domain: "audio",
          labelKey: "capability.audio.denoise",
        }),
        NEEDS_FLAG,
        NEEDS_PACK_AND_FLAG,
      ],
      roadmap: [],
    },
    {
      domain: "generate",
      labelKey: "capability.domain.generate",
      capabilities: [
        capability({
          id: "generate.text_to_image",
          domain: "generate",
          labelKey: "capability.generate.textToImage",
          jobKind: "generation",
        }),
      ],
      roadmap: [],
    },
  ],
};

function queryClient(withData = true) {
  const client = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        staleTime: withData ? Infinity : 0,
      },
    },
  });
  if (withData) {
    client.setQueryData(["capability-tree"], TREE);
  }
  return client;
}

function Providers({
  children,
  client,
}: {
  children: ReactNode;
  client: QueryClient;
}) {
  return (
    <QueryClientProvider client={client}>
      <LocaleProvider initialLocale="en">{children}</LocaleProvider>
    </QueryClientProvider>
  );
}

function renderTree({
  onSelect = vi.fn(),
  onProvision = vi.fn(),
  onActivate = vi.fn(),
  withData = true,
}: {
  onSelect?: (capability: CapabilityResponse) => void;
  onProvision?: (capability: CapabilityResponse) => void;
  onActivate?: (capability: CapabilityResponse) => void;
  withData?: boolean;
} = {}) {
  const client = queryClient(withData);
  return render(
    <CapabilityTree
      onSelect={onSelect}
      onProvision={onProvision}
      onActivate={onActivate}
    />,
    {
      wrapper: ({ children }) => (
        <Providers client={client}>{children}</Providers>
      ),
    },
  );
}

function itemFor(label: string): HTMLElement {
  const item = screen.getByText(label).closest("li");
  if (!item) {
    throw new Error(`No capability item found for ${label}`);
  }
  return item;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("CapabilityTree", () => {
  it("shows the four domains first and switches the second-level capabilities", () => {
    renderTree();

    for (const domain of [
      "capability.domain.video",
      "capability.domain.image",
      "capability.domain.audio",
      "capability.domain.generate",
    ] as const) {
      expect(
        screen.getByRole("button", { name: en[domain] }),
      ).toBeVisible();
    }
    expect(screen.getByText(en["capability.video.upscale"])).toBeVisible();

    fireEvent.click(
      screen.getByRole("button", { name: en["capability.domain.image"] }),
    );

    expect(screen.getByText(en["capability.image.upscale"])).toBeVisible();
    expect(
      screen.queryByText(en["capability.video.upscale"]),
    ).not.toBeInTheDocument();
  });

  it("renders available, needs-setup and not-implemented capabilities differently", () => {
    renderTree();

    expect(itemFor(en["capability.video.upscale"])).toHaveAttribute(
      "data-status",
      "available",
    );
    expect(itemFor(en["capability.video.interpolate"])).toHaveAttribute(
      "data-status",
      "needs_setup",
    );
    expect(itemFor(en["capability.print.slice"])).toHaveAttribute(
      "data-status",
      "not_implemented",
    );
  });

  it("keeps a not-implemented reason in the DOM without prior interaction", () => {
    renderTree();

    expect(
      screen.getByText(en["capability.reason.noSlicerPack"]),
    ).toBeVisible();
  });

  it("does not render a button for a not-implemented capability", () => {
    renderTree();

    expect(
      within(itemFor(en["capability.print.slice"])).queryByRole("button"),
    ).not.toBeInTheDocument();
  });

  it("provisions a needs-setup capability when its pack is missing", () => {
    const onProvision = vi.fn();
    renderTree({ onProvision });

    fireEvent.click(
      within(itemFor(en["capability.video.interpolate"])).getByRole("button", {
        name: new RegExp(en["capability.tree.download"]),
      }),
    );

    expect(onProvision).toHaveBeenCalledOnce();
    expect(onProvision).toHaveBeenCalledWith(NEEDS_PACK);
  });

  it("shows the setup reason but no provision button when no pack is missing", () => {
    renderTree();
    fireEvent.click(
      screen.getByRole("button", { name: en["capability.domain.image"] }),
    );

    const item = itemFor(en["capability.image.upscale"]);
    expect(
      within(item).getByText(en["capability.setup.missingModel"]),
    ).toBeVisible();
    expect(within(item).queryByRole("button")).not.toBeInTheDocument();
  });

  it("offers turning the setting on right on the card when only the flag is missing", () => {
    // La friccion que arregla: el pack ya esta en disco y la tarjeta mandaba a
    // Ajustes a prender un flag a mano.
    const onActivate = vi.fn();
    renderTree({ onActivate });
    fireEvent.click(
      screen.getByRole("button", { name: en["capability.domain.audio"] }),
    );

    fireEvent.click(
      within(itemFor(en["capability.audio.restoreSr"])).getByRole("button", {
        name: new RegExp(en["capability.tree.activate"]),
      }),
    );

    expect(onActivate).toHaveBeenCalledOnce();
    expect(onActivate).toHaveBeenCalledWith(NEEDS_FLAG);
  });

  it("asks for the package first when the pack is missing too", () => {
    // Prender el flag sin el pack no arregla nada: el orden importa.
    renderTree();
    fireEvent.click(
      screen.getByRole("button", { name: en["capability.domain.audio"] }),
    );

    const item = itemFor(en["capability.audio.restore"]);
    expect(
      within(item).getByRole("button", {
        name: new RegExp(en["capability.tree.download"]),
      }),
    ).toBeVisible();
    expect(
      within(item).queryByRole("button", {
        name: new RegExp(en["capability.tree.activate"]),
      }),
    ).not.toBeInTheDocument();
  });

  it("does not offer to turn anything on when what is missing is a model", () => {
    renderTree();
    fireEvent.click(
      screen.getByRole("button", { name: en["capability.domain.image"] }),
    );

    expect(
      within(itemFor(en["capability.image.upscale"])).queryByRole("button", {
        name: new RegExp(en["capability.tree.activate"]),
      }),
    ).not.toBeInTheDocument();
  });

  it("selects an available capability", () => {
    const onSelect = vi.fn();
    renderTree({ onSelect });

    fireEvent.click(
      screen.getByRole("button", { name: en["capability.video.upscale"] }),
    );

    expect(onSelect).toHaveBeenCalledOnce();
    expect(onSelect).toHaveBeenCalledWith(AVAILABLE);
  });

  it("places roadmap capabilities under their own heading, outside the live list", () => {
    renderTree();

    const roadmapHeading = screen.getByRole("heading", {
      name: en["capability.tree.roadmap"],
    });
    const roadmapSection = roadmapHeading.closest("section");
    if (!roadmapSection) {
      throw new Error("Roadmap heading is not inside a section");
    }

    expect(
      within(roadmapSection).getByText(en["capability.print.slice"]),
    ).toBeVisible();
    expect(
      within(screen.getByTestId("live-capabilities")).queryByText(
        en["capability.print.slice"],
      ),
    ).not.toBeInTheDocument();
    expect(
      within(screen.getByTestId("roadmap-capabilities")).queryByText(
        en["capability.video.upscale"],
      ),
    ).not.toBeInTheDocument();
  });

  it("shows translated loading copy while the query is pending", () => {
    vi.stubGlobal("fetch", vi.fn(() => new Promise<Response>(() => undefined)));

    renderTree({ withData: false });

    expect(screen.getByText(en["capability.tree.loading"])).toBeVisible();
  });

  it("shows translated error copy when the query fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "boom" }), {
          status: 500,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    renderTree({ withData: false });

    await waitFor(() =>
      expect(screen.getByText(en["capability.tree.loadFailed"])).toBeVisible(),
    );
  });
});
