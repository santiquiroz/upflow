import { afterEach, describe, expect, it, vi } from "vitest";
import { changePassword, getMe, login, logout, logoutAll, setup } from "./auth";

vi.mock("../lib/api", () => ({ apiGet: vi.fn(), apiPostJson: vi.fn() }));

import { apiGet, apiPostJson } from "../lib/api";

afterEach(() => {
  vi.mocked(apiGet).mockReset();
  vi.mocked(apiPostJson).mockReset();
});

describe("auth service", () => {
  it("login posts username/password to /auth/login", async () => {
    vi.mocked(apiPostJson).mockResolvedValue({ ok: true });
    await login("alice", "pw");
    expect(apiPostJson).toHaveBeenCalledWith("/auth/login", { username: "alice", password: "pw" });
  });

  it("logout posts to /auth/logout", async () => {
    vi.mocked(apiPostJson).mockResolvedValue({ ok: true });
    await logout();
    expect(apiPostJson).toHaveBeenCalledWith("/auth/logout", {});
  });

  it("logoutAll posts to /auth/logout-all", async () => {
    vi.mocked(apiPostJson).mockResolvedValue({ ok: true });
    await logoutAll();
    expect(apiPostJson).toHaveBeenCalledWith("/auth/logout-all", {});
  });

  it("getMe fetches /auth/me", async () => {
    vi.mocked(apiGet).mockResolvedValue({ username: "alice" });
    await getMe();
    expect(apiGet).toHaveBeenCalledWith("/auth/me");
  });

  it("changePassword posts camelCase body", async () => {
    vi.mocked(apiPostJson).mockResolvedValue({ ok: true });
    await changePassword("old", "new");
    expect(apiPostJson).toHaveBeenCalledWith("/auth/change-password", {
      currentPassword: "old",
      newPassword: "new",
    });
  });

  it("setup posts username/password to /auth/setup", async () => {
    vi.mocked(apiPostJson).mockResolvedValue({ ok: true });
    await setup("admin", "pw12345678");
    expect(apiPostJson).toHaveBeenCalledWith("/auth/setup", { username: "admin", password: "pw12345678" });
  });
});
