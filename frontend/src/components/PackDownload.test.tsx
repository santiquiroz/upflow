import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ApiError } from "../lib/api";
import type { ProvisionJob } from "../lib/apiTypes";
import * as capabilitiesService from "../services/capabilities";
import * as licensesService from "../services/licenses";
import { PackDownload } from "./PackDownload";
import { LICENSE_REQUIRED_KEY, LICENSE_UNAVAILABLE_KEY } from "./PackLicenseGate";

vi.mock("../services/capabilities", () => ({
  provisionPack: vi.fn(),
  getProvisionStatus: vi.fn(),
}));

vi.mock("../services/licenses", () => ({
  fetchPackLicense: vi.fn(),
}));

const FULL_LICENSE = "S-Lab License 1.0\n\nRedistribution and use for non-commercial purpose only.\n";

const QUEUED: ProvisionJob = {
  jobId: "job-1",
  pack: "restore-faces-nc",
  status: "queued",
  error: null,
  statusUrl: "/api/v1/capabilities/provision/job-1",
};

function renderWithQuery(node: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{node}</QueryClientProvider>);
}

function refuseUntilAccepted(key = LICENSE_REQUIRED_KEY) {
  vi.mocked(capabilitiesService.provisionPack).mockImplementation(async (_pack, _variant, accepted) => {
    if (!accepted) throw new ApiError(403, "Accept the license first.", key);
    return QUEUED;
  });
}

beforeEach(() => {
  vi.mocked(capabilitiesService.provisionPack).mockReset();
  vi.mocked(capabilitiesService.getProvisionStatus).mockReset();
  vi.mocked(capabilitiesService.getProvisionStatus).mockResolvedValue(QUEUED);
  vi.mocked(licensesService.fetchPackLicense).mockReset();
  vi.mocked(licensesService.fetchPackLicense).mockResolvedValue({
    pack: "restore-faces-nc",
    gated: true,
    licenseText: FULL_LICENSE,
  });
});

describe("PackDownload", () => {
  it("downloads a pack without a license gate straight away", async () => {
    vi.mocked(capabilitiesService.provisionPack).mockResolvedValue({ ...QUEUED, pack: "rife" });
    renderWithQuery(<PackDownload pack="rife" />);

    fireEvent.click(screen.getByRole("button", { name: "Download" }));

    await waitFor(() => expect(capabilitiesService.provisionPack).toHaveBeenCalledWith("rife", undefined, false));
    expect(licensesService.fetchPackLicense).not.toHaveBeenCalled();
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
  });

  it("shows the full license text and a checkbox when the pack is license gated", async () => {
    refuseUntilAccepted();
    renderWithQuery(<PackDownload pack="restore-faces-nc" />);

    fireEvent.click(screen.getByRole("button", { name: "Download" }));

    const region = await screen.findByRole("region", { name: "Full license text" });
    expect(region.textContent).toBe(FULL_LICENSE);
    expect(screen.getByRole("checkbox", { name: "I have read the full license and accept its terms" })).not.toBeChecked();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(licensesService.fetchPackLicense).toHaveBeenCalledWith("restore-faces-nc");
  });

  it("keeps the download disabled until the checkbox is ticked", async () => {
    refuseUntilAccepted();
    renderWithQuery(<PackDownload pack="restore-faces-nc" />);
    fireEvent.click(screen.getByRole("button", { name: "Download" }));

    const accept = await screen.findByRole("button", { name: "Accept and download" });
    expect(accept).toBeDisabled();

    fireEvent.click(screen.getByRole("checkbox"));
    expect(accept).toBeEnabled();
  });

  it("sends the acceptance once the license is accepted", async () => {
    refuseUntilAccepted();
    renderWithQuery(<PackDownload pack="restore-faces-nc" variant={undefined} />);
    fireEvent.click(screen.getByRole("button", { name: "Download" }));

    fireEvent.click(await screen.findByRole("checkbox"));
    fireEvent.click(screen.getByRole("button", { name: "Accept and download" }));

    await waitFor(() =>
      expect(capabilitiesService.provisionPack).toHaveBeenLastCalledWith("restore-faces-nc", undefined, true),
    );
    expect(await screen.findByRole("button", { name: "Downloading…" })).toBeDisabled();
    expect(screen.queryByRole("region", { name: "Full license text" })).not.toBeInTheDocument();
  });

  it("blocks the download when the license text is not available", async () => {
    refuseUntilAccepted(LICENSE_UNAVAILABLE_KEY);
    vi.mocked(licensesService.fetchPackLicense).mockResolvedValue({
      pack: "restore-faces-nc",
      gated: true,
      licenseText: null,
    });
    renderWithQuery(<PackDownload pack="restore-faces-nc" />);

    fireEvent.click(screen.getByRole("button", { name: "Download" }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The license text of this package is not available, so it cannot be accepted or downloaded.",
    );
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Accept and download" })).not.toBeInTheDocument();
  });

  it("still reports other download errors as errors", async () => {
    vi.mocked(capabilitiesService.provisionPack).mockRejectedValue(new ApiError(400, "Unknown pack.", null));
    renderWithQuery(<PackDownload pack="nope" />);

    fireEvent.click(screen.getByRole("button", { name: "Download" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Unknown pack.");
    expect(licensesService.fetchPackLicense).not.toHaveBeenCalled();
  });
});
