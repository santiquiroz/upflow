import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { UpdateCheck } from "../lib/apiTypes";
import * as useUpdateCheckModule from "../hooks/useUpdateCheck";
import { UpdateBanner } from "./UpdateBanner";

vi.mock("../hooks/useUpdateCheck");

const DISMISS_STORAGE_KEY = "upflow.dismissedUpdate";
const RELEASE_URL = "https://github.com/santiquiroz/upflow/releases/tag/v0.2.0";

function buildStatus(overrides: Partial<UpdateCheck> = {}): UpdateCheck {
  return {
    currentVersion: "0.1.0",
    latestVersion: "0.2.0",
    updateAvailable: true,
    releaseUrl: RELEASE_URL,
    publishedAt: "2026-07-16T10:00:00Z",
    checkedAt: "2026-07-16T10:00:00Z",
    error: null,
    ...overrides,
  };
}

function mockStatus(data: UpdateCheck | undefined): void {
  vi.mocked(useUpdateCheckModule.useUpdateCheck).mockReturnValue({
    data,
  } as unknown as ReturnType<typeof useUpdateCheckModule.useUpdateCheck>);
}

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  vi.mocked(useUpdateCheckModule.useUpdateCheck).mockReset();
  localStorage.clear();
});

describe("UpdateBanner", () => {
  it("announces the latest version and links to the release when an update is available", () => {
    mockStatus(buildStatus());

    render(<UpdateBanner />);

    expect(screen.getByRole("status")).toHaveTextContent(/new version 0\.2\.0 available/i);
    const link = screen.getByRole("link", { name: /view release/i });
    expect(link).toHaveAttribute("href", RELEASE_URL);
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("renders nothing when no update is available", () => {
    mockStatus(buildStatus({ updateAvailable: false }));

    render(<UpdateBanner />);

    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("renders nothing when the check reported an error", () => {
    mockStatus(buildStatus({ error: "rate limited" }));

    render(<UpdateBanner />);

    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("renders nothing while the check has not resolved", () => {
    mockStatus(undefined);

    render(<UpdateBanner />);

    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("hides the banner and persists the dismissed version when dismissed", () => {
    mockStatus(buildStatus());

    render(<UpdateBanner />);
    fireEvent.click(screen.getByRole("button", { name: /dismiss/i }));

    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(localStorage.getItem(DISMISS_STORAGE_KEY)).toBe("0.2.0");
  });

  it("stays hidden on re-mount for a version already dismissed", () => {
    localStorage.setItem(DISMISS_STORAGE_KEY, "0.2.0");
    mockStatus(buildStatus());

    render(<UpdateBanner />);

    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("re-appears when the latest version is newer than the dismissed one", () => {
    localStorage.setItem(DISMISS_STORAGE_KEY, "0.2.0");
    mockStatus(buildStatus({ latestVersion: "0.3.0", releaseUrl: RELEASE_URL.replace("0.2.0", "0.3.0") }));

    render(<UpdateBanner />);

    expect(screen.getByRole("status")).toHaveTextContent(/new version 0\.3\.0 available/i);
  });
});
