import type { LucideIcon } from "lucide-react";
import { AudioWaveform, Boxes, Sliders, Sparkles, Users as UsersIcon, Wand2, Zap } from "lucide-react";

export interface NavEntry {
  label: string;
  path: string;
  icon: LucideIcon;
  requiredPermission?: string;
}

export const NAV_ENTRIES: readonly NavEntry[] = [
  { label: "Enhance", path: "/", icon: Wand2 },
  { label: "Audio", path: "/audio", icon: AudioWaveform },
  { label: "Generate", path: "/generate", icon: Sparkles },
  { label: "Models", path: "/models", icon: Boxes },
  { label: "Realtime", path: "/realtime", icon: Zap },
  { label: "Settings", path: "/settings", icon: Sliders },
  { label: "Users", path: "/users", icon: UsersIcon, requiredPermission: "users:manage" },
];
