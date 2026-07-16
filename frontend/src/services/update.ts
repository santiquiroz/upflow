import type { UpdateCheck } from "../lib/apiTypes";

const UPDATE_CHECK_URL = "/api/v1/update-check";

// Mirrors the fetch style of lib/api.ts. The endpoint always answers 200 with
// an `error` field on failure, so a non-ok response is genuinely unexpected.
export async function fetchUpdateCheck(): Promise<UpdateCheck> {
  const response = await fetch(UPDATE_CHECK_URL, { method: "GET" });
  if (!response.ok) {
    throw new Error(`Update check failed with status ${response.status}`);
  }
  return (await response.json()) as UpdateCheck;
}
