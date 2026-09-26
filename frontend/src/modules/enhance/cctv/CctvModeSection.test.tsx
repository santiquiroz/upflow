import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { en } from "../../../i18n/en";
import * as api from "../../../lib/api";
import type { EngineInfoResponse, VideoCapabilities, VideoJobResponse } from "../../../lib/apiTypes";
import * as cctvService from "../../../services/cctv";
import { ANALYSIS, PRESETS_RESPONSE } from "./cctvFixtures";
import { CctvModeSection } from "./CctvModeSection";

vi.mock("../../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../lib/api")>();
  return { ...actual, getVideoCapabilities: vi.fn(), getVideoJob: vi.fn(), getEngineInfo: vi.fn() };
});

vi.mock("../../../services/cctv", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../services/cctv")>();
  return {
    ...actual,
    analyzeCctv: vi.fn(),
    getCctvAnalysis: vi.fn(),
    getCctvPresets: vi.fn(),
    createCctvJob: vi.fn(),
  };
});

const CAPS: VideoCapabilities = {
  interpEngines: ["rife"],
  cctvAvailable: true,
  cctvReasonKey: null,
  cctvAiAvailable: false,
  cctvAiReasonKey: "capability.setup.needsGpu",
  cctvUnavailableSteps: [],
};

const QUEUED_JOB = { jobId: "job-1", status: "queued", originalFilename: "ch01.dav" } as VideoJobResponse;
const COMPLETED_JOB = {
  ...QUEUED_JOB,
  status: "completed",
  metadata: {},
  cctv: {
    task: "clarify",
    lane: "classic",
    preset: "night_ir",
    sourceSha256: "b".repeat(64),
    noOsd: true,
    osdBoxesConfirmed: false,
    warnings: [],
    artifacts: [{ name: "package", url: "/api/v1/video/jobs/job-1/artifacts/package" }],
    verifyUrl: "/api/v1/video/jobs/job-1/verify",
  },
} as VideoJobResponse;

function renderSection(
  caps: VideoCapabilities = CAPS,
  initialFile: File | null = null,
  presets: cctvService.CctvPresetsResponse = PRESETS_RESPONSE,
) {
  vi.mocked(api.getVideoCapabilities).mockResolvedValue(caps);
  vi.mocked(api.getVideoJob).mockResolvedValue(QUEUED_JOB);
  vi.mocked(api.getEngineInfo).mockResolvedValue({ maxVideoUploadMb: 1, outputTtlHours: 36 } as EngineInfoResponse);
  vi.mocked(cctvService.getCctvPresets).mockResolvedValue(presets);
  vi.mocked(cctvService.analyzeCctv).mockResolvedValue({ kind: "done", analysis: ANALYSIS });
  vi.mocked(cctvService.createCctvJob).mockResolvedValue(QUEUED_JOB);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const onExit = vi.fn();
  function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  }
  const view = render(<CctvModeSection initialFile={initialFile} onExit={onExit} />, { wrapper: Wrapper });
  return { ...view, onExit, queryClient };
}

function dropRecording(name = "ch01.dav") {
  const input = document.getElementById("cctv-file-input") as HTMLInputElement;
  fireEvent.change(input, { target: { files: [new File(["IMKH"], name)] } });
}

async function analyzedSection(caps: VideoCapabilities = CAPS) {
  const view = renderSection(caps);
  dropRecording();
  await screen.findByText(en["cctv.diag.title"]);
  await screen.findByRole("button", { name: en["cctv.start"] });
  return view;
}

function stepLabels(): string[] {
  const list = screen.getByRole("list", { name: en["cctv.steps.legend"] });
  return within(list)
    .getAllByRole("checkbox")
    .map((box) => box.closest("label")?.textContent ?? "");
}

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("CctvModeSection", () => {
  it("analyzes any dropped recorder export and shows the hash and the diagnosis", async () => {
    await analyzedSection();

    expect(cctvService.analyzeCctv).toHaveBeenCalledWith(expect.any(File), expect.anything());
    expect(screen.getByText(en["cctv.hash.recorded"])).toBeInTheDocument();
    expect(screen.getByText(ANALYSIS.sourceSha256)).toBeInTheDocument();
    expect(screen.getByText("Hikvision PS")).toBeInTheDocument();
    expect(screen.getByText(en["cctv.lite"])).toBeInTheDocument();
  });

  it("analyzes the file it was opened with", async () => {
    renderSection(CAPS, new File(["IMKH"], "from-video-panel.mp4"));

    await screen.findByText(en["cctv.diag.title"]);
    expect(cctvService.analyzeCctv).toHaveBeenCalledTimes(1);
  });

  it("shows no AI step and no interpolation in the classic lane", async () => {
    await analyzedSection();

    const labels = stepLabels();
    expect(labels).not.toContain(en["cctv.step.ai_deblock"]);
    expect(labels).not.toContain(en["cctv.step.ai_upscale"]);
    expect(labels.join(" ")).not.toMatch(/interpolat/i);
    expect(screen.getByText(en["cctv.fps.interpOff"])).toBeInTheDocument();
  });

  it("starts with the suggested preset and its steps checked", async () => {
    await analyzedSection();

    expect(screen.getByRole("button", { name: /Night \/ IR/ })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("checkbox", { name: en["cctv.step.deblock"] })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: en["cctv.step.gray"] })).toBeChecked();
  });

  it("keeps Start disabled until the on-screen text is decided", async () => {
    await analyzedSection();
    const start = screen.getByRole("button", { name: en["cctv.start"] });

    expect(start).toBeDisabled();
    expect(screen.getAllByText(en["cctv.osd.confirm"]).length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("checkbox", { name: en["cctv.osd.none"] }));

    expect(start).toBeEnabled();
  });

  it("creates the classic job directly with the ordered steps", async () => {
    await analyzedSection();
    fireEvent.click(screen.getByRole("checkbox", { name: en["cctv.osd.none"] }));

    fireEvent.click(screen.getByRole("button", { name: en["cctv.start"] }));

    await waitFor(() => expect(cctvService.createCctvJob).toHaveBeenCalled());
    expect(vi.mocked(cctvService.createCctvJob).mock.calls[0][0]).toEqual({
      token: "tok-1",
      task: "clarify",
      preset: "night_ir",
      steps: [
        { id: "deblock", params: { filter: "deblock", filter_type: "strong", block: 8 } },
        { id: "gray", params: { filter: "gray" } },
      ],
      osdBoxes: [],
      osdBoxesConfirmed: false,
      noOsd: true,
      trim: null,
    });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("sends the case details with the job", async () => {
    await analyzedSection();
    fireEvent.click(screen.getByRole("checkbox", { name: en["cctv.osd.none"] }));
    fireEvent.change(screen.getByLabelText(en["cctv.case.caseLabel"]), { target: { value: "2026-114" } });
    fireEvent.change(screen.getByLabelText(en["cctv.case.clockOffset"]), { target: { value: "-12" } });

    fireEvent.click(screen.getByRole("button", { name: en["cctv.start"] }));

    await waitFor(() => expect(cctvService.createCctvJob).toHaveBeenCalled());
    const request = vi.mocked(cctvService.createCctvJob).mock.calls[0][0];
    expect(request.caseLabel).toBe("2026-114");
    expect(request.acquisition).toEqual({ clockOffsetSeconds: -12 });
  });

  it("blocks Start while the clock offset isn't a number", async () => {
    await analyzedSection();
    fireEvent.click(screen.getByRole("checkbox", { name: en["cctv.osd.none"] }));

    fireEvent.change(screen.getByLabelText(en["cctv.case.clockOffset"]), { target: { value: "1,5" } });

    expect(screen.getByRole("button", { name: en["cctv.start"] })).toBeDisabled();
    expect(screen.getAllByText(en["cctv.case.offsetInvalid"]).length).toBeGreaterThan(0);
  });

  it("shows the result with the server's retention once the job completes", async () => {
    await analyzedSection();
    vi.mocked(cctvService.createCctvJob).mockResolvedValue(COMPLETED_JOB);
    vi.mocked(api.getVideoJob).mockResolvedValue(COMPLETED_JOB);
    fireEvent.click(screen.getByRole("checkbox", { name: en["cctv.osd.none"] }));

    fireEvent.click(screen.getByRole("button", { name: en["cctv.start"] }));

    expect(await screen.findByRole("link", { name: en["cctv.package.download"] })).toBeInTheDocument();
    expect(screen.getByText(en["cctv.retention"].replace("{{hours}}", "36"))).toBeInTheDocument();
  });

  it("disables the AI lane without a GPU and says how long the CPU would take", async () => {
    await analyzedSection();

    const aiLane = screen.getByRole("radio", { name: new RegExp(en["cctv.lane.ai"].replace(/[()]/g, "\\$&")) });
    expect(aiLane).toBeDisabled();
    expect(within(aiLane).getByText(/AI enhancement needs a GPU\. On this computer's CPU it would take about 2 h\./)).toBeInTheDocument();
  });

  it("asks for confirmation before an AI lane job and warns on the lane", async () => {
    await analyzedSection({ ...CAPS, cctvAiAvailable: true, cctvAiReasonKey: null });

    fireEvent.click(screen.getByRole("radio", { name: new RegExp(en["cctv.lane.ai"].replace(/[()]/g, "\\$&")) }));
    expect(screen.getByText(en["cctv.ai.banner"])).toBeInTheDocument();
    expect(screen.getByText(en["cctv.plates.noAi"])).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: en["cctv.step.ai_deblock"] })).toBeChecked();
    fireEvent.click(screen.getByRole("checkbox", { name: en["cctv.osd.none"] }));
    fireEvent.click(screen.getByRole("button", { name: en["cctv.start"] }));

    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByText(en["cctv.ai.confirm"])).toBeInTheDocument();
    expect(cctvService.createCctvJob).not.toHaveBeenCalled();

    fireEvent.click(within(dialog).getByRole("button", { name: en["cctv.ai.confirm.continue"] }));

    await waitFor(() => expect(cctvService.createCctvJob).toHaveBeenCalled());
    expect(vi.mocked(cctvService.createCctvJob).mock.calls[0][0].task).toBe("enhance");
  });

  it("sends the chosen AI upscale model and scale with the AI job", async () => {
    await analyzedSection({ ...CAPS, cctvAiAvailable: true, cctvAiReasonKey: null });
    fireEvent.click(screen.getByRole("radio", { name: new RegExp(en["cctv.lane.ai"].replace(/[()]/g, "\\$&")) }));
    fireEvent.click(screen.getByRole("checkbox", { name: en["cctv.osd.none"] }));

    fireEvent.change(screen.getByLabelText(en["cctv.ai.upscale.model"]), { target: { value: "realesrgan-x4plus" } });
    fireEvent.click(screen.getByRole("radio", { name: "4x" }));
    fireEvent.click(screen.getByRole("button", { name: en["cctv.start"] }));
    fireEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: en["cctv.ai.confirm.continue"] }));

    await waitFor(() => expect(cctvService.createCctvJob).toHaveBeenCalled());
    const request = vi.mocked(cctvService.createCctvJob).mock.calls[0][0];
    expect(request).toMatchObject({ task: "enhance", modelId: "realesrgan-x4plus", scale: 4 });
    expect(request.steps.map((step) => step.id)).toEqual(["ai_deblock", "gray", "ai_upscale"]);
  });

  it("opens the multi-frame still in either lane", async () => {
    await analyzedSection();

    const roiTab = screen.getByRole("tab", { name: en["cctv.task.roi_fusion"] });
    fireEvent.click(roiTab);

    expect(roiTab).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("region", { name: en["cctv.roi.legend"] })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: en["cctv.start"] })).toBeDisabled();
    expect(screen.getAllByText(en["cctv.roi.blocked.box"]).length).toBeGreaterThan(0);
  });

  it("switches the steps when another preset is picked", async () => {
    await analyzedSection();

    fireEvent.click(screen.getByRole("button", { name: "Day" }));

    expect(screen.getByRole("checkbox", { name: en["cctv.step.gray"] })).not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: en["cctv.step.deinterlace"] })).toBeChecked();
  });

  it("explains a step this ffmpeg build can't run", async () => {
    const withoutDenoise = {
      ...PRESETS_RESPONSE,
      steps: {
        ...PRESETS_RESPONSE.steps,
        classic: PRESETS_RESPONSE.steps.classic.map((step) =>
          step.id === "denoise"
            ? { ...step, available: false, filters: step.filters.map((filter) => ({ ...filter, available: false })) }
            : step,
        ),
      },
    };
    renderSection(CAPS, null, withoutDenoise);
    dropRecording();

    const denoise = await screen.findByRole("checkbox", { name: en["cctv.step.denoise"] });
    expect(denoise).toBeDisabled();
    expect(screen.getByText("This ffmpeg build doesn't include hqdn3d, atadenoise, so this step is off.")).toBeInTheDocument();
  });

  it("translates a keyed analysis error", async () => {
    renderSection();
    vi.mocked(cctvService.analyzeCctv).mockRejectedValue(
      new api.ApiError(400, "The recording can't be decoded.", "cctv.undecodable"),
    );
    dropRecording();

    expect(await screen.findByRole("alert")).toHaveTextContent(en["cctv.undecodable"]);
  });

  it("polls a long analysis until it completes", async () => {
    renderSection();
    vi.mocked(cctvService.analyzeCctv).mockResolvedValue({ kind: "pending", analysisJobId: "an-1" });
    vi.mocked(cctvService.getCctvAnalysis).mockResolvedValue({
      analysisJobId: "an-1",
      status: "completed",
      statusUrl: "/api/v1/video/cctv/analysis/an-1",
      result: ANALYSIS,
      error: null,
      errorKey: null,
    });
    dropRecording();

    await screen.findByText(en["cctv.diag.title"]);
    expect(cctvService.getCctvAnalysis).toHaveBeenCalledWith("an-1");
  });

  it("refuses a recording over the video upload limit before uploading it", async () => {
    const { queryClient } = renderSection();
    await waitFor(() => expect(queryClient.getQueryData(["engine"])).toBeDefined());
    const input = document.getElementById("cctv-file-input") as HTMLInputElement;
    const huge = new File([new Uint8Array(2 * 1024 * 1024)], "huge.dav");

    fireEvent.change(input, { target: { files: [huge] } });

    expect(screen.getByRole("alert")).toHaveTextContent("That file is 2 MB. The limit is 1 MB.");
    expect(cctvService.analyzeCctv).not.toHaveBeenCalled();
  });

  it("leaves the mode when the switch is turned off", async () => {
    const { onExit } = renderSection();

    fireEvent.click(screen.getByRole("switch", { name: en["cctv.toggle"] }));

    expect(onExit).toHaveBeenCalled();
  });
});
