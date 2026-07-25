import { afterEach, describe, expect, it, vi } from "vitest";
import { createUser, getUserJobs, listUsers, updateUser } from "./users";

vi.mock("../lib/api", () => ({ apiGet: vi.fn(), apiPostJson: vi.fn(), apiPatchJson: vi.fn() }));

import { apiGet, apiPatchJson, apiPostJson } from "../lib/api";

afterEach(() => {
  vi.mocked(apiGet).mockReset();
  vi.mocked(apiPostJson).mockReset();
  vi.mocked(apiPatchJson).mockReset();
});

describe("users service", () => {
  it("listUsers fetches /users", async () => {
    vi.mocked(apiGet).mockResolvedValue({ users: [] });
    await listUsers();
    expect(apiGet).toHaveBeenCalledWith("/users");
  });

  it("createUser posts to /users", async () => {
    vi.mocked(apiPostJson).mockResolvedValue({ user: {}, temporaryPassword: "x" });
    await createUser({ username: "bob", role: "user" });
    expect(apiPostJson).toHaveBeenCalledWith("/users", { username: "bob", role: "user" });
  });

  it("updateUser patches /users/{id}", async () => {
    vi.mocked(apiPatchJson).mockResolvedValue({ user: {}, temporaryPassword: null });
    await updateUser("u1", { disabled: true });
    expect(apiPatchJson).toHaveBeenCalledWith("/users/u1", { disabled: true });
  });

  it("getUserJobs fetches /users/{id}/jobs", async () => {
    vi.mocked(apiGet).mockResolvedValue({ jobs: [] });
    await getUserJobs("u1");
    expect(apiGet).toHaveBeenCalledWith("/users/u1/jobs");
  });
});
