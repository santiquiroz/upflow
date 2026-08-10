import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { en } from "../../i18n/en";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import type { EditableSettingStatus } from "../../lib/apiTypes";
import { CapabilitySettingsSection } from "./CapabilitySettingsSection";

const fetchEditableSettings = vi.hoisted(() => vi.fn());
const patchSetting = vi.hoisted(() => vi.fn());

vi.mock("../../services/settings", () => ({ fetchEditableSettings, patchSetting }));

function setting(overrides: Partial<EditableSettingStatus> & { key: string }): EditableSettingStatus {
  return {
    configured: false,
    value: null,
    requiresRestart: false,
    ...overrides,
  };
}

const AUDIOSR_OFF = setting({ key: "enable_audiosr", value: "false" });
const RESTORE_ON = setting({ key: "enable_audio_restore", configured: true, value: "true" });
const TOKEN = setting({ key: "hf_token", configured: true });
const CAD_URL = setting({ key: "cad_llm_base_url", requiresRestart: true });
const REBAR = setting({ key: "rebar_confirmed", value: "false" });

function renderSection(settings: EditableSettingStatus[]) {
  fetchEditableSettings.mockResolvedValue({ settings });
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>
        <LocaleProvider initialLocale="en">{children}</LocaleProvider>
      </QueryClientProvider>
    );
  }
  return render(<CapabilitySettingsSection />, { wrapper: Wrapper });
}

afterEach(() => {
  fetchEditableSettings.mockReset();
  patchSetting.mockReset();
});

describe("CapabilitySettingsSection", () => {
  it("shows a switch per flag with the state the server reports", async () => {
    renderSection([AUDIOSR_OFF, RESTORE_ON, TOKEN]);

    const audiosr = await screen.findByRole("checkbox", {
      name: en["settings.flag.enable_audiosr"],
    });
    expect(audiosr).not.toBeChecked();
    expect(
      screen.getByRole("checkbox", { name: en["settings.flag.enable_audio_restore"] }),
    ).toBeChecked();
  });

  it("never draws a switch for a setting without a value", async () => {
    // hf_token no devuelve valor: sin el, un interruptor solo podria mentir
    // sobre su estado.
    renderSection([AUDIOSR_OFF, TOKEN]);

    await screen.findByRole("checkbox", { name: en["settings.flag.enable_audiosr"] });
    expect(screen.getAllByRole("checkbox")).toHaveLength(1);
  });

  it("does not repeat a flag that already has its own card elsewhere", async () => {
    renderSection([AUDIOSR_OFF, REBAR]);

    await screen.findByRole("checkbox", { name: en["settings.flag.enable_audiosr"] });
    expect(screen.getAllByRole("checkbox")).toHaveLength(1);
  });

  it("turns a flag on with a PATCH", async () => {
    patchSetting.mockResolvedValue({ key: "enable_audiosr" });
    renderSection([AUDIOSR_OFF]);

    fireEvent.click(
      await screen.findByRole("checkbox", { name: en["settings.flag.enable_audiosr"] }),
    );

    await waitFor(() => expect(patchSetting).toHaveBeenCalledWith("enable_audiosr", "true"));
  });

  it("turns a flag off with a PATCH", async () => {
    patchSetting.mockResolvedValue({ key: "enable_audio_restore" });
    renderSection([RESTORE_ON]);

    fireEvent.click(
      await screen.findByRole("checkbox", { name: en["settings.flag.enable_audio_restore"] }),
    );

    await waitFor(() =>
      expect(patchSetting).toHaveBeenCalledWith("enable_audio_restore", "false"),
    );
  });

  it("shows the backend message when the switch cannot be saved", async () => {
    patchSetting.mockRejectedValue(new Error("no es editable desde la UI"));
    renderSection([AUDIOSR_OFF]);

    fireEvent.click(
      await screen.findByRole("checkbox", { name: en["settings.flag.enable_audiosr"] }),
    );

    expect(await screen.findByRole("alert")).toHaveTextContent("no es editable");
  });

  it("saves the CAD server URL and says plainly that it needs a restart", async () => {
    patchSetting.mockResolvedValue({ key: "cad_llm_base_url" });
    renderSection([CAD_URL]);

    const input = await screen.findByLabelText(en["settings.cadServer.label"]);
    fireEvent.change(input, { target: { value: "http://localhost:11434/v1" } });
    fireEvent.click(screen.getByRole("button", { name: en["settings.videoLimit.save"] }));

    await waitFor(() =>
      expect(patchSetting).toHaveBeenCalledWith("cad_llm_base_url", "http://localhost:11434/v1"),
    );
    expect(screen.getByText(en["settings.cadServer.restartNeeded"])).toBeVisible();
  });
});
