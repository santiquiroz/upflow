import type { LucideIcon } from "lucide-react";
import { AudioLines, AudioWaveform, Boxes, Download, Eraser, LayoutGrid, Sliders, Sparkles, Users as UsersIcon, Wand2, Zap } from "lucide-react";

export interface NavEntry {
  label: string;
  path: string;
  icon: LucideIcon;
  requiredPermission?: string;
}

export const NAV_ENTRIES: readonly NavEntry[] = [
  { label: "Tasks", path: "/", icon: LayoutGrid },
  { label: "Enhance", path: "/enhance", icon: Wand2 },
  { label: "Audio", path: "/audio", icon: AudioWaveform },
  { label: "Transcribe", path: "/transcribe", icon: AudioLines },
  { label: "Download", path: "/download", icon: Download },
  { label: "Generate", path: "/generate", icon: Sparkles },
  { label: "Editor", path: "/editor", icon: Eraser },
  { label: "Models", path: "/models", icon: Boxes },
  { label: "Realtime", path: "/realtime", icon: Zap },
  { label: "Settings", path: "/settings", icon: Sliders },
  { label: "Users", path: "/users", icon: UsersIcon, requiredPermission: "users:manage" },
];
