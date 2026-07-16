import { useRef, useState, type KeyboardEvent } from "react";
import { ImagePanel } from "../modules/enhance/ImagePanel";
import { VideoPanel } from "../modules/enhance/VideoPanel";

type EnhanceMedium = "image" | "video";

interface MediumTab {
  value: EnhanceMedium;
  label: string;
  subtitle: string;
}

const MEDIUM_TABS: readonly MediumTab[] = [
  { value: "image", label: "Image", subtitle: "Upscale a single image." },
  { value: "video", label: "Video", subtitle: "Upscale, interpolate and clean up a video." },
];

function tabId(value: EnhanceMedium): string {
  return `enhance-tab-${value}`;
}

function panelId(value: EnhanceMedium): string {
  return `enhance-panel-${value}`;
}

function tabClassName(isActive: boolean): string {
  const base =
    "rounded-sm border px-3 py-1.5 text-sm transition-[background-color,border-color,color] duration-fast focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";
  if (isActive) {
    return `${base} border-accent bg-accent text-bg`;
  }
  return `${base} border-border bg-surface text-text-dim hover:border-text-faint hover:text-text`;
}

// WAI-ARIA APG tabs pattern: Right/Left roving focus wraps around the tab
// list and activates immediately (automatic selection model), matching the
// click behavior these tabs already have.
function resolveNextTabIndex(currentIndex: number, key: string): number | null {
  if (key === "ArrowRight") {
    return (currentIndex + 1) % MEDIUM_TABS.length;
  }
  if (key === "ArrowLeft") {
    return (currentIndex - 1 + MEDIUM_TABS.length) % MEDIUM_TABS.length;
  }
  return null;
}

export function EnhancePage() {
  const [medium, setMedium] = useState<EnhanceMedium>("image");
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const activeTab = MEDIUM_TABS.find((tab) => tab.value === medium) ?? MEDIUM_TABS[0];

  function handleTabKeyDown(event: KeyboardEvent<HTMLButtonElement>, currentIndex: number): void {
    const nextIndex = resolveNextTabIndex(currentIndex, event.key);
    if (nextIndex === null) {
      return;
    }
    event.preventDefault();
    const nextTab = MEDIUM_TABS[nextIndex];
    setMedium(nextTab.value);
    tabRefs.current[nextIndex]?.focus();
  }

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="font-heading text-2xl font-semibold text-text">Enhance</h1>
        <p className="mt-1 text-sm text-text-dim">{activeTab.subtitle}</p>
      </div>
      <div role="tablist" aria-label="Enhance medium" className="flex w-fit gap-2">
        {MEDIUM_TABS.map((tab, index) => {
          const isActive = medium === tab.value;
          return (
            <button
              key={tab.value}
              ref={(el) => {
                tabRefs.current[index] = el;
              }}
              type="button"
              role="tab"
              id={tabId(tab.value)}
              aria-selected={isActive}
              aria-controls={panelId(tab.value)}
              tabIndex={isActive ? 0 : -1}
              onClick={() => setMedium(tab.value)}
              onKeyDown={(event) => handleTabKeyDown(event, index)}
              className={tabClassName(isActive)}
            >
              {tab.label}
            </button>
          );
        })}
      </div>
      <div id={panelId(medium)} role="tabpanel" aria-labelledby={tabId(medium)} tabIndex={0}>
        {medium === "image" ? <ImagePanel /> : <VideoPanel />}
      </div>
    </div>
  );
}
