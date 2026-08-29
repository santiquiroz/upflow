import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { AudioStemDownload } from "../lib/apiTypes";
import { StemDownloadList } from "./StemDownloadList";

const LINK_CLASS = "link";
const ICON_CLASS = "h-4 w-4";
const CONTAINER_CLASS = "flex flex-wrap gap-2";

function renderList(stems: AudioStemDownload[] | null, vocalsUrl: string | null = null) {
  return render(
    <StemDownloadList
      stems={stems}
      downloadUrl="/api/v1/audio/jobs/j1/download"
      vocalsUrl={vocalsUrl}
      linkClassName={LINK_CLASS}
      iconClassName={ICON_CLASS}
      containerClassName={CONTAINER_CLASS}
    />,
  );
}

describe("StemDownloadList", () => {
  it("renders one labelled link per stem, in the backend's order", () => {
    renderList([
      { id: "vocals", labelKey: "audio.stem.vocals", url: "/d?stem=vocals" },
      { id: "drums", labelKey: "audio.stem.drums", url: "/d?stem=drums" },
    ]);

    const vocals = screen.getByRole("link", { name: "Vocals" });
    const drums = screen.getByRole("link", { name: "Drums" });
    expect(vocals).toHaveAttribute("href", "/d?stem=vocals");
    expect(drums).toHaveAttribute("href", "/d?stem=drums");
    expect(vocals.compareDocumentPosition(drums) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("labels the derived minus-one stems from the catalog", () => {
    // Los derivados llegan en la MISMA lista que los stems crudos, con su
    // labelKey minus_<id>: si la clave falta, el link mostraria la clave cruda.
    renderList([
      { id: "minus_drums", labelKey: "audio.stem.minus_drums", url: "/d?stem=minus_drums" },
      { id: "minus_bass", labelKey: "audio.stem.minus_bass", url: "/d?stem=minus_bass" },
    ]);

    expect(screen.getByRole("link", { name: "Without drums" })).toHaveAttribute(
      "href",
      "/d?stem=minus_drums",
    );
    expect(screen.getByRole("link", { name: "Without bass" })).toHaveAttribute(
      "href",
      "/d?stem=minus_bass",
    );
  });

  it("falls back to the labelled instrumental + vocals pair for pre-stems jobs", () => {
    renderList(null, "/d?stem=vocals");

    expect(screen.getByRole("link", { name: "Instrumental" })).toHaveAttribute(
      "href",
      "/api/v1/audio/jobs/j1/download",
    );
    expect(screen.getByRole("link", { name: "Vocals" })).toHaveAttribute("href", "/d?stem=vocals");
  });

  it("renders nothing without stems or a vocals url, leaving the generic link to the caller", () => {
    const { container } = renderList(null, null);

    expect(container).toBeEmptyDOMElement();
  });
});
