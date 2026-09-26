import type { CapabilityResponse, CapabilityStatus, CapabilityTreeResponse } from "../lib/apiTypes";
import { RESTORE_CAPABILITY_ID } from "./restoreRelease";

function restoreCapability(status: CapabilityStatus): CapabilityResponse {
  return {
    id: RESTORE_CAPABILITY_ID,
    domain: "image",
    labelKey: "capability.image.restore",
    status,
    provisioning: "builtin",
    jobKind: "image",
    strategies: ["dsp"],
    missingPacks: [],
    unavailableReasonKey: status === "not_implemented" ? "capability.reason.pendingModelRelease" : null,
    setupReasonKey: null,
    activatableSettings: [],
  };
}

export function treeWithRestore(released: boolean): CapabilityTreeResponse {
  const capability = restoreCapability(released ? "available" : "not_implemented");
  return {
    domains: [
      {
        domain: "image",
        labelKey: "capability.domain.image",
        capabilities: released ? [capability] : [],
        roadmap: released ? [] : [capability],
      },
    ],
  };
}
