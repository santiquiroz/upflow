import { useState } from "react";
import { ImagePanel } from "../modules/enhance/ImagePanel";
import { VideoPanel } from "../modules/enhance/VideoPanel";

type EnhanceMedium = "image" | "video";

const MEDIUM_TABS: readonly { value: EnhanceMedium; label: string }[] = [
  { value: "image", label: "Image" },
  { value: "video", label: "Video" },
];

function tabClassName(isActive: boolean): string {
  const base =
    "rounded-sm border px-3 py-1.5 text-sm transition-[background-color,border-color,color] duration-fast focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";
  if (isActive) {
    return `${base} border-accent bg-accent text-bg`;
  }
  return `${base} border-border bg-surface text-text-dim hover:border-text-faint hover:text-text`;
}

export function EnhancePage() {
  const [medium, setMedium] = useState<EnhanceMedium>("image");

  return (
    <div className="flex flex-col gap-6">
      <div role="tablist" aria-label="Enhance medium" className="flex w-fit gap-2">
        {MEDIUM_TABS.map((tab) => (
          <button
            key={tab.value}
            type="button"
            role="tab"
            aria-selected={medium === tab.value}
            className={tabClassName(medium === tab.value)}
            onClick={() => setMedium(tab.value)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      {medium === "image" ? <ImagePanel /> : <VideoPanel />}
    </div>
  );
}
