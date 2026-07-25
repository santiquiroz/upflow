import { apiGet, apiPostJson } from "../lib/api";
import type { MeResponse } from "../lib/apiTypes";

export function login(username: string, password: string): Promise<{ ok: boolean }> {
  return apiPostJson<{ ok: boolean }>("/auth/login", { username, password });
}

export function logout(): Promise<{ ok: boolean }> {
  return apiPostJson<{ ok: boolean }>("/auth/logout", {});
}

export function logoutAll(): Promise<{ ok: boolean }> {
  return apiPostJson<{ ok: boolean }>("/auth/logout-all", {});
}

export function getMe(): Promise<MeResponse> {
  return apiGet<MeResponse>("/auth/me");
}

export function changePassword(currentPassword: string, newPassword: string): Promise<{ ok: boolean }> {
  return apiPostJson<{ ok: boolean }>("/auth/change-password", { currentPassword, newPassword });
}

export function setup(username: string, password: string): Promise<{ ok: boolean }> {
  return apiPostJson<{ ok: boolean }>("/auth/setup", { username, password });
}
