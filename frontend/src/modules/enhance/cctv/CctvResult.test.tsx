import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { en } from "../../../i18n/en";
import { ApiError } from "../../../lib/api";
import type { CctvJobSummary } from "../../../lib/apiTypes";
import * as cctvService from "../../../services/cctv";
import { CctvResult } from "./CctvResult";

vi.mock("../../../services/cctv", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../services/cctv")>();
  return { ...actual, verifyCctvFiles: vi.fn() };
});

const BASE = "/api/v1/video/jobs/job-1/artifacts";
const SHA = "a".repeat(64);

function artifact(name: string) {
  return { name, url: `${BASE}/${name}` };
}

const SUMMARY: CctvJobSummary = {
  task: "clarify",
  lane: "classic",
  preset: "night_ir",
  sourceSha256: SHA,
  noOsd: false,
  osdBoxesConfirmed: true,
  warnings: [],
  artifacts: ["analysis", "viewing", "comparison", "package", "report_json", "report_html", "sha256sums"].map(artifact),
  verifyUrl: "/api/v1/video/jobs/job-1/verify",
};

function renderResult(summary: CctvJobSummary = SUMMARY, retentionHours: number | null = 24) {
  const queryClient = new QueryClient({ defaultOptions: { mutations: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  return render(<CctvResult jobId="job-1" summary={summary} retentionHours={retentionHours} />, { wrapper: Wrapper });
}

function clickVerify(): void {
  fireEvent.click(screen.getByRole("button", { name: en["cctv.verify.action"] }));
}

afterEach(() => {
  vi.mocked(cctvService.verifyCctvFiles).mockReset();
});

describe("CctvResult", () => {
  it("offers the handover package and the report as links to the backend artifacts", () => {
    renderResult();

    expect(screen.getByRole("link", { name: en["cctv.package.download"] })).toHaveAttribute("href", `${BASE}/package`);
    const report = screen.getByRole("link", { name: en["cctv.report.open"] });
    expect(report).toHaveAttribute("href", `${BASE}/report_html`);
    expect(report).toHaveAttribute("target", "_blank");
  });

  it("lists the produced files by what they are", () => {
    renderResult();

    const list = screen.getByRole("list", { name: en["cctv.result.files"] });
    expect(list).toHaveTextContent(en["cctv.result.file.analysis"]);
    expect(list).toHaveTextContent(en["cctv.result.file.comparison"]);
    expect(screen.getByRole("link", { name: en["cctv.result.file.sha256sums"] })).toHaveAttribute(
      "href",
      `${BASE}/sha256sums`,
    );
  });

  it("shows the hash of the file as Upflow received it", () => {
    renderResult();

    expect(screen.getByText(SHA)).toBeInTheDocument();
  });

  it("says when the results are deleted, with the server's retention", () => {
    renderResult(SUMMARY, 48);

    expect(screen.getByText(en["cctv.retention"].replace("{{hours}}", "48"))).toBeInTheDocument();
  });

  it("explains what the file check does not prove", () => {
    renderResult();

    const button = screen.getByRole("button", { name: en["cctv.verify.action"] });
    expect(button).toHaveAttribute("title", en["cctv.verify.tooltip"]);
    expect(button).toHaveAccessibleDescription(en["cctv.verify.tooltip"]);
  });

  it("reports no changes when every hash still matches", async () => {
    vi.mocked(cctvService.verifyCctvFiles).mockResolvedValue({ ok: true, checked: 9, mismatches: [], missing: [] });
    renderResult();

    clickVerify();

    expect(await screen.findByText(new RegExp(en["cctv.verify.ok"].replace(".", "\\.")))).toBeInTheDocument();
    expect(cctvService.verifyCctvFiles).toHaveBeenCalledWith("job-1");
  });

  it("lists every changed and missing file", async () => {
    vi.mocked(cctvService.verifyCctvFiles).mockResolvedValue({
      ok: false,
      checked: 9,
      mismatches: ["analysis.mkv"],
      missing: ["report.html"],
    });
    renderResult();

    clickVerify();

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(en["cctv.verify.mismatch"].replace("{{path}}", "analysis.mkv"));
    expect(alert).toHaveTextContent(en["cctv.verify.missing"].replace("{{path}}", "report.html"));
  });

  it("shows the translated error when the check fails", async () => {
    vi.mocked(cctvService.verifyCctvFiles).mockRejectedValue(
      new ApiError(409, "The CCTV job has not finished yet.", "cctv.error.jobNotCompleted"),
    );
    renderResult();

    clickVerify();

    expect(await screen.findByRole("alert")).toHaveTextContent("The CCTV job has not finished yet.");
  });

  it("has no handover package link when the job couldn't build one, and says why", () => {
    renderResult({
      ...SUMMARY,
      artifacts: SUMMARY.artifacts.filter((item) => item.name !== "package"),
      warnings: ["cctv.package.noDiskRoom"],
    });

    expect(screen.queryByRole("link", { name: en["cctv.package.download"] })).not.toBeInTheDocument();
    expect(screen.getByText(en["cctv.package.noDiskRoom"])).toBeInTheDocument();
  });

  it("hides the file check until the job has a verify endpoint", () => {
    renderResult({ ...SUMMARY, verifyUrl: null });

    expect(screen.queryByRole("button", { name: en["cctv.verify.action"] })).not.toBeInTheDocument();
  });
});
