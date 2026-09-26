import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as licensesService from "../../services/licenses";
import type { LicensedModel, LicensesResponse } from "../../services/licenses";
import { LicensesView } from "./LicensesView";

vi.mock("../../services/licenses", () => ({ fetchLicenses: vi.fn() }));

const DRUNET: LicensedModel = {
  id: "drunet-color",
  name: "DRUNet (color)",
  licenseSpdx: "MIT",
  licenseUrl: "https://github.com/cszn/DPIR/blob/master/LICENSE",
  copyright: "Copyright (c) 2021 Kai Zhang",
  attribution: "DPIR by Kai Zhang et al.",
  dataLineage: "D1a",
  commercialUse: "unclear",
  sourceUrl: "https://github.com/cszn/DPIR",
  sourceRevision: "4b7d6a0e3c1f2a9b8c7d6e5f",
  modifications: ["Exported to ONNX opset 17", "Normalization baked in"],
  files: [{ name: "LICENSE", text: "MIT License\n\nPermission is hereby granted" }],
};

const LICENSES: LicensesResponse = {
  packs: [{ pack: "restore-core", models: [DRUNET] }],
  thirdParty: [
    {
      title: "Real-ESRGAN ONNX exports",
      section: "Components bundled in the release",
      fields: { License: ["BSD-3-Clause"], Copyright: ["Copyright (c) 2021, Xintao Wang"] },
      licenseText: "BSD 3-Clause License\n\nRedistribution and use",
    },
  ],
};

function renderView(response: LicensesResponse | Error = LICENSES) {
  if (response instanceof Error) {
    vi.mocked(licensesService.fetchLicenses).mockRejectedValue(response);
  } else {
    vi.mocked(licensesService.fetchLicenses).mockResolvedValue(response);
  }
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<LicensesView />, { wrapper: Wrapper });
}

async function modelCard(name: string): Promise<HTMLElement> {
  return screen.findByRole("article", { name });
}

afterEach(() => {
  vi.mocked(licensesService.fetchLicenses).mockReset();
});

describe("LicensesView", () => {
  it("titles the section Licenses", async () => {
    renderView();

    expect(screen.getByRole("heading", { name: "Licenses" })).toBeInTheDocument();
    await modelCard("DRUNet (color)");
  });

  it("shows each installed pack model with its license linked to the license text", async () => {
    renderView();

    const card = await modelCard("DRUNet (color)");

    expect(screen.getByRole("heading", { name: "restore-core" })).toBeInTheDocument();
    expect(within(card).getByRole("link", { name: "MIT" })).toHaveAttribute("href", DRUNET.licenseUrl);
    expect(within(card).getByText(DRUNET.copyright)).toBeInTheDocument();
    expect(within(card).getByText(DRUNET.attribution)).toBeInTheDocument();
  });

  it("explains the training data lineage in words, keeping its key", async () => {
    renderView();

    const card = await modelCard("DRUNet (color)");

    expect(
      within(card).getByText("D1a — Training images come from collections licensed for research or non-commercial use"),
    ).toBeInTheDocument();
  });

  it("explains every part of a combined lineage", async () => {
    renderView({ ...LICENSES, packs: [{ pack: "restore-faces", models: [{ ...DRUNET, dataLineage: "D1b + D1c" }] }] });

    const card = await modelCard("DRUNet (color)");
    const lineage = within(card).getByText(/^D1b \+ D1c — /);

    expect(lineage.textContent).toContain("no-derivatives or share-alike terms");
    expect(lineage.textContent).toContain("not declared by the authors");
  });

  it("says plainly whether commercial use is allowed", async () => {
    renderView();

    const card = await modelCard("DRUNet (color)");

    expect(within(card).getByText("Unclear — check the license and training data before commercial use")).toBeInTheDocument();
  });

  it("links the source at the exact revision and lists the changes made to the weights", async () => {
    renderView();

    const card = await modelCard("DRUNet (color)");

    expect(within(card).getByRole("link", { name: "github.com/cszn/DPIR @ 4b7d6a0e3c1f" })).toHaveAttribute(
      "href",
      DRUNET.sourceUrl,
    );
    expect(within(card).getByText("Exported to ONNX opset 17")).toBeInTheDocument();
    expect(within(card).getByText("Normalization baked in")).toBeInTheDocument();
  });

  it("shows the license files that shipped with the pack on demand", async () => {
    renderView();

    const card = await modelCard("DRUNet (color)");
    expect(within(card).getByText(/Permission is hereby granted/)).not.toBeVisible();
    fireEvent.click(within(card).getByText("LICENSE"));

    expect(within(card).getByText(/Permission is hereby granted/)).toBeVisible();
  });

  it("does not turn a non-web address into a link", async () => {
    const unsafe = { ...DRUNET, licenseUrl: "javascript:alert(1)" };
    renderView({ ...LICENSES, packs: [{ pack: "restore-core", models: [unsafe] }] });

    const card = await modelCard("DRUNet (color)");

    expect(within(card).queryByRole("link", { name: "MIT" })).toBeNull();
    expect(within(card).getByText("MIT")).toBeInTheDocument();
  });

  it("says when no model pack is installed", async () => {
    renderView({ ...LICENSES, packs: [] });

    expect(await screen.findByText("No restoration model packs are installed.")).toBeInTheDocument();
  });

  it("lists the third-party components bundled with Upflow, with their license text", async () => {
    renderView();

    const notice = await screen.findByRole("article", { name: "Real-ESRGAN ONNX exports" });

    expect(within(notice).getByText("BSD-3-Clause")).toBeInTheDocument();
    expect(within(notice).getByText("Copyright (c) 2021, Xintao Wang")).toBeInTheDocument();
    fireEvent.click(within(notice).getByText("License text"));
    expect(within(notice).getByText(/Redistribution and use/)).toBeVisible();
    expect(screen.getByRole("heading", { name: "Components bundled in the release" })).toBeInTheDocument();
  });

  it("reports a failed request instead of an empty list", async () => {
    renderView(new Error("network down"));

    expect(await screen.findByText("Could not load the licenses.")).toBeInTheDocument();
  });
});
