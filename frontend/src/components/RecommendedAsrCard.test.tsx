import { fireEvent, render, screen } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LocaleProvider } from "../i18n/LocaleProvider";
import { en } from "../i18n/en";
import {
  RECOMMENDED_ASR_REPO_ID,
  RecommendedAsrCard,
} from "./RecommendedAsrCard";

const installMock = vi.fn();
let installedModels: unknown[] = [];
let installPhase = "idle";

vi.mock("../hooks/useTranscribeJob", () => ({
  useInstalledAsrModels: () => ({ data: installedModels, isLoading: false }),
  useAsrModelInstall: () => ({
    phase: installPhase,
    progressPct: null,
    errorMessage: null,
    modelId: null,
    install: installMock,
    reset: vi.fn(),
  }),
}));

function Wrapper({ children }: { children: ReactNode }) {
  return <LocaleProvider initialLocale="en">{children}</LocaleProvider>;
}

beforeEach(() => {
  installMock.mockReset();
  installedModels = [];
  installPhase = "idle";
});

describe("RecommendedAsrCard", () => {
  it("offers the recommended model when nothing is installed", () => {
    render(<RecommendedAsrCard />, { wrapper: Wrapper });

    fireEvent.click(
      screen.getByRole("button", { name: en["asrDefault.button"] }),
    );

    expect(installMock).toHaveBeenCalledWith(RECOMMENDED_ASR_REPO_ID);
  });

  it("renders nothing when a model is already installed", () => {
    installedModels = [{ id: "asr-1" }];
    const { container } = render(<RecommendedAsrCard />, { wrapper: Wrapper });

    expect(container).toBeEmptyDOMElement();
  });

  it("announces success once installed", () => {
    installPhase = "installed";
    render(<RecommendedAsrCard />, { wrapper: Wrapper });

    expect(screen.getByRole("status")).toHaveTextContent(en["asrDefault.done"]);
  });
});
