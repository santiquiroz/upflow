import type { Tool } from "./EditorToolRail";

const MASK_TOOLS: Tool[] = ["people", "tap", "protect", "brush", "eraser", "lasso", "rect"];
const RETOUCH_TOOLS: Tool[] = ["restore", "clone", "heal"];

export interface ToolLockState {
  tapAvailable: boolean;
  resultShown: boolean;
  lamaInstalled: boolean;
}

export function toolLocks(state: ToolLockState, t: (key: string) => string): Partial<Record<Tool, string>> {
  const locks: Partial<Record<Tool, string>> = {};
  if (!state.tapAvailable) {
    locks.tap = t("editor.tool.tapUnavailable");
    locks.protect = t("editor.tool.tapUnavailable");
  }
  if (!state.lamaInstalled) locks.heal = t("editor.heal.needsLama");
  if (state.resultShown) {
    const reason = t("editor.review.keepEditingFirst");
    for (const tool of [...MASK_TOOLS, ...RETOUCH_TOOLS]) locks[tool] = reason;
  }
  return locks;
}
