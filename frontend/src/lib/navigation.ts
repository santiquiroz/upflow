import type { LucideIcon } from "lucide-react";
import { AudioLines, AudioWaveform, Box, Boxes, Download, Eraser, LayoutGrid, Mic, MicVocal, Sliders, Sparkles, Users as UsersIcon, Wand2, Zap } from "lucide-react";

export interface NavEntry {
  labelKey: string;
  path: string;
  icon: LucideIcon;
  requiredPermission?: string;
}

export const NAV_ENTRIES: readonly NavEntry[] = [
  { labelKey: "nav.tasks", path: "/", icon: LayoutGrid },
  { labelKey: "nav.enhance", path: "/enhance", icon: Wand2 },
  { labelKey: "nav.audio", path: "/audio", icon: AudioWaveform },
  { labelKey: "nav.transcribe", path: "/transcribe", icon: AudioLines },
  { labelKey: "nav.karaoke", path: "/karaoke", icon: MicVocal },
  { labelKey: "nav.download", path: "/download", icon: Download },
  { labelKey: "nav.generate", path: "/generate", icon: Sparkles },
  { labelKey: "nav.editor", path: "/editor", icon: Eraser },
  { labelKey: "nav.models", path: "/models", icon: Boxes },
  { labelKey: "nav.voice", path: "/voice", icon: Mic },
  { labelKey: "nav.print", path: "/print", icon: Box },
  { labelKey: "nav.realtime", path: "/realtime", icon: Zap },
  { labelKey: "nav.settings", path: "/settings", icon: Sliders },
  { labelKey: "nav.users", path: "/users", icon: UsersIcon, requiredPermission: "users:manage" },
];
