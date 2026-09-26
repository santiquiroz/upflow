import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { en } from "../../../i18n/en";
import { ROI_SUMMARY } from "./cctvFixtures";
import { RoiFusionResult } from "./RoiFusionResult";

const BASE = "/api/v1/video/jobs/job-1/artifacts";
const ARTIFACTS = ["roi:fused", "roi:reference", "roi:agreement", "roi:stack", "roi:samples"].map((name) => ({
  name,
  url: `${BASE}/${name}`,
}));

describe("RoiFusionResult", () => {
  it("says how many frames carried new information and flags near-copies", () => {
    render(<RoiFusionResult roi={ROI_SUMMARY} artifacts={ARTIFACTS} />);

    expect(screen.getByText("Frames used: 28 of 30 (only 2 carried new information)")).toBeInTheDocument();
    expect(screen.getByText(en["cctv.roi.littleToGain"])).toBeInTheDocument();
  });

  it("shows the combined image next to the untouched reference and the agreement map with its legend", () => {
    render(<RoiFusionResult roi={ROI_SUMMARY} artifacts={ARTIFACTS} />);

    expect(screen.getByRole("img", { name: "Combined frames (3x)" })).toHaveAttribute("src", `${BASE}/roi:fused`);
    expect(screen.getByRole("img", { name: "Reference frame 20, enlarged without changes (nearest neighbor)" })).toHaveAttribute(
      "src",
      `${BASE}/roi:reference`,
    );
    expect(screen.getByRole("img", { name: en["cctv.roi.result.agreement"] })).toHaveAttribute("src", `${BASE}/roi:agreement`);
    expect(screen.getByText(en["cctv.roi.agreement"])).toBeInTheDocument();
  });

  it("lists the automatic warnings with their measured values", () => {
    render(<RoiFusionResult roi={ROI_SUMMARY} artifacts={ARTIFACTS} />);

    const warnings = screen.getByRole("list", { name: en["cctv.result.warnings"] });
    expect(warnings).toHaveTextContent("The plate is about 14 px tall in the recording.");
    expect(warnings).toHaveTextContent(en["cctv.roi.nearCopies"]);
    expect(warnings).toHaveTextContent("Frames left out because they could not be aligned: 3, 17");
    expect(warnings).not.toHaveTextContent("Frames used");
  });

  it("does not flag a fusion that had enough distinct samples", () => {
    render(<RoiFusionResult roi={{ ...ROI_SUMMARY, nearCopies: false, notices: [], rejectedFrames: [] }} artifacts={ARTIFACTS} />);

    expect(screen.queryByText(en["cctv.roi.littleToGain"])).not.toBeInTheDocument();
    expect(screen.queryByRole("list", { name: en["cctv.result.warnings"] })).not.toBeInTheDocument();
  });

  it("skips an image the job did not produce", () => {
    render(<RoiFusionResult roi={ROI_SUMMARY} artifacts={ARTIFACTS.slice(0, 1)} />);

    expect(screen.getAllByRole("img")).toHaveLength(1);
  });
});
