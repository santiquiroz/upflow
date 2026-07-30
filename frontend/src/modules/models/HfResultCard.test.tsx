import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { en } from "../../i18n/en";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import * as api from "../../lib/api";
import type {
  HfModelSearchResultResponse,
  InstallStatusResponse,
  UpscalerPreflightResponse,
} from "../../lib/apiTypes";
import { HfResultCard } from "./HfResultCard";

vi.mock("../../lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../lib/api")>();
  return {
    ...actual,
    installModel: vi.fn(),
    getInstallStatus: vi.fn(),
    preflightUpscalerModel: vi.fn(),
  };
});

const POLL_INTERVAL_MS = 10;

const RESULT: HfModelSearchResultResponse = {
  id: "example/anime-2x",
  author: "example",
  pipelineTag: "image-to-image",
  downloads: 12000,
  likes: 42,
  tags: ["onnx"],
  compat: null,
  compatReasonKey: null,
  compatReasonParams: {},
  availablePrecisions: [],
};

const PREFLIGHT: UpscalerPreflightResponse = {
  repoId: RESULT.id,
  compat: "ready_onnx",
  compatReasonKey: "compat.upscaler.readyOnnx",
  compatReasonParams: {},
  degraded: false,
  downloadBytes: 2 * 1024 ** 3,
  devices: [
    {
      id: "dml:0",
      name: "RX 7800 XT",
      kind: "gpu",
      freeVramBytes: 12 * 1024 ** 3,
    },
  ],
  disk: { targetPath: "D:\\models", freeBytes: 20 * 1024 ** 3 },
  freeRamBytes: 24 * 1024 ** 3,
};

function renderCard(result: HfModelSearchResultResponse = RESULT) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <LocaleProvider initialLocale="en">{children}</LocaleProvider>
      </QueryClientProvider>
    );
  }
  return render(<HfResultCard result={result} pollIntervalMs={POLL_INTERVAL_MS} />, { wrapper: Wrapper });
}

afterEach(() => {
  vi.mocked(api.installModel).mockReset();
  vi.mocked(api.getInstallStatus).mockReset();
  vi.mocked(api.preflightUpscalerModel).mockReset();
});

describe("HfResultCard", () => {
  it("shows repo id, author and tabular download/like counts", () => {
    renderCard();

    expect(screen.getByText("example/anime-2x")).toBeInTheDocument();
    expect(screen.getByText("example")).toBeInTheDocument();
    const downloads = screen.getByText("12,000");
    expect(downloads).toHaveClass("font-mono-tabular");
  });

  it("starts an install and shows a determinate progress bar while downloading", async () => {
    vi.mocked(api.installModel).mockResolvedValue({ installId: "install-1", statusUrl: "/x" });
    vi.mocked(api.getInstallStatus).mockResolvedValue({
      installId: "install-1",
      repoId: RESULT.id,
      status: "downloading",
      progressPct: 42,
      modelId: null,
      error: null,
    } satisfies InstallStatusResponse);

    renderCard();
    fireEvent.click(screen.getByRole("button", { name: /install/i }));

    await waitFor(() => expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "42"));
    expect(screen.getByText("42%")).toBeInTheDocument();
  });

  it("transitions from downloading through validating, converting to installed", async () => {
    vi.mocked(api.installModel).mockResolvedValue({ installId: "install-2", statusUrl: "/x" });
    const base: InstallStatusResponse = {
      installId: "install-2",
      repoId: RESULT.id,
      status: "downloading",
      progressPct: 10,
      modelId: null,
      error: null,
    };
    vi.mocked(api.getInstallStatus)
      .mockResolvedValueOnce({ ...base, status: "downloading", progressPct: 80 })
      .mockResolvedValueOnce({ ...base, status: "validating", progressPct: null })
      .mockResolvedValueOnce({ ...base, status: "converting", progressPct: null })
      .mockResolvedValue({ ...base, status: "installed", progressPct: null, modelId: "example--anime-2x" });

    renderCard();
    fireEvent.click(screen.getByRole("button", { name: /install/i }));

    await waitFor(() => expect(screen.getByText(/installed/i)).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: /install/i })).not.toBeInTheDocument();
  });

  it("shows a clear error message and a retry action when the install fails", async () => {
    vi.mocked(api.installModel).mockResolvedValue({ installId: "install-3", statusUrl: "/x" });
    vi.mocked(api.getInstallStatus).mockResolvedValue({
      installId: "install-3",
      repoId: RESULT.id,
      status: "error",
      progressPct: null,
      modelId: null,
      error: "Repository does not contain a supported model file",
    });

    renderCard();
    fireEvent.click(screen.getByRole("button", { name: /install/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/does not contain a supported model file/i);
    expect(screen.getByRole("button", { name: /try again/i })).toBeInTheDocument();
  });

  it("returns to the install button after retrying from an error", async () => {
    vi.mocked(api.installModel).mockResolvedValue({ installId: "install-4", statusUrl: "/x" });
    vi.mocked(api.getInstallStatus).mockResolvedValue({
      installId: "install-4",
      repoId: RESULT.id,
      status: "error",
      progressPct: null,
      modelId: null,
      error: "repo_id must look like 'owner/name'",
    });

    renderCard();
    fireEvent.click(screen.getByRole("button", { name: /install/i }));
    await screen.findByRole("alert");

    fireEvent.click(screen.getByRole("button", { name: /try again/i }));

    expect(screen.getByRole("button", { name: /install/i })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("shows the compat badge without fetching preflight on render", () => {
    renderCard({ ...RESULT, compat: "needs_conversion" });

    expect(
      screen.getByText(en["upscaler.compat.needsConversion"]),
    ).toBeInTheDocument();
    expect(api.preflightUpscalerModel).not.toHaveBeenCalled();
  });

  it("shows measured preflight data without turning VRAM into a verdict", async () => {
    vi.mocked(api.preflightUpscalerModel).mockResolvedValue(PREFLIGHT);

    renderCard({ ...RESULT, compat: "ready_onnx" });
    fireEvent.click(
      screen.getByRole("button", {
        name: en["upscaler.details.show"],
      }),
    );

    expect(
      await screen.findByText(en["compat.upscaler.readyOnnx"]),
    ).toBeInTheDocument();
    expect(screen.getByText("RX 7800 XT")).toBeInTheDocument();
    expect(
      screen.getByText(
        en["upscaler.capacity.vramFree"].replace("{{free}}", "12.0 GB"),
      ),
    ).toBeInTheDocument();
    expect(
      screen.queryByLabelText(en["upscaler.warnings.ariaLabel"]),
    ).not.toBeInTheDocument();
  });

  it("keeps Install enabled when gated and disk warnings are visible", async () => {
    vi.mocked(api.preflightUpscalerModel).mockResolvedValue({
      ...PREFLIGHT,
      compat: "gated",
      compatReasonKey: "compat.gated",
      disk: { targetPath: "D:\\models", freeBytes: 1 * 1024 ** 3 },
    });

    renderCard({ ...RESULT, compat: "gated" });
    fireEvent.click(
      screen.getByRole("button", {
        name: en["upscaler.details.show"],
      }),
    );

    expect(
      await screen.findByText(en["upscaler.warning.gated"]),
    ).toBeInTheDocument();
    expect(
      screen.getByText(
        en["upscaler.warning.diskLow"]
          .replace("{{free}}", "1.0 GB")
          .replace("{{path}}", "D:\\models")
          .replace("{{needed}}", "2.0 GB"),
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /^install$/i }),
    ).toBeEnabled();
  });

  it("fetches preflight once when details are toggled repeatedly", async () => {
    vi.mocked(api.preflightUpscalerModel).mockResolvedValue(PREFLIGHT);

    renderCard();
    fireEvent.click(
      screen.getByRole("button", {
        name: en["upscaler.details.show"],
      }),
    );
    await screen.findByText(en["compat.upscaler.readyOnnx"]);

    fireEvent.click(
      screen.getByRole("button", {
        name: en["upscaler.details.hide"],
      }),
    );
    fireEvent.click(
      screen.getByRole("button", {
        name: en["upscaler.details.show"],
      }),
    );

    await waitFor(() =>
      expect(api.preflightUpscalerModel).toHaveBeenCalledTimes(1),
    );
  });

  it("keeps Install enabled when preflight is degraded", async () => {
    vi.mocked(api.preflightUpscalerModel).mockResolvedValue({
      ...PREFLIGHT,
      degraded: true,
      compat: null,
      compatReasonKey: null,
      downloadBytes: null,
      devices: [],
      disk: null,
      freeRamBytes: null,
    });

    renderCard();
    fireEvent.click(
      screen.getByRole("button", {
        name: en["upscaler.details.show"],
      }),
    );

    expect(
      await screen.findByText(en["upscaler.warning.degraded"]),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /^install$/i }),
    ).toBeEnabled();
  });
});
