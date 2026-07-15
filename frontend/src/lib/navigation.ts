import type { LucideIcon } from "lucide-react";
import { Boxes, Sliders, Wand2, Zap } from "lucide-react";

export interface NavEntry {
  label: string;
  path: string;
  icon: LucideIcon;
}

export const NAV_ENTRIES: readonly NavEntry[] = [
  { label: "Enhance", path: "/", icon: Wand2 },
  { label: "Models", path: "/models", icon: Boxes },
  { label: "Realtime", path: "/realtime", icon: Zap },
  { label: "Settings", path: "/settings", icon: Sliders },
];
