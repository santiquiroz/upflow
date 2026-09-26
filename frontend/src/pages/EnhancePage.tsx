import { useRef, useState, type KeyboardEvent } from "react";
import { useTranslation } from "../i18n/LocaleProvider";
import { ImagePanel } from "../modules/enhance/ImagePanel";
import { VideoPanel } from "../modules/enhance/VideoPanel";
import { PhotoRestorePanel } from "../modules/enhance/restore/PhotoRestorePanel";
import { useRestoreReleased } from "./restoreRelease";

export type EnhanceMedium = "image" | "video" | "restore";

interface MediumTab {
  value: EnhanceMedium;
  labelKey: string;
  subtitleKey: string;
}

const MEDIUM_TABS: readonly MediumTab[] = [
  { value: "image", labelKey: "enhance.tab.image", subtitleKey: "enhance.subtitle.image" },
  { value: "video", labelKey: "enhance.tab.video", subtitleKey: "enhance.subtitle.video" },
  { value: "restore", labelKey: "restore.tab", subtitleKey: "restore.subtitle" },
];

export function isEnhanceMedium(value: string | undefined): value is EnhanceMedium {
  return MEDIUM_TABS.some((tab) => tab.value === value);
}

export function visibleTabs(restoreReleased: boolean): readonly MediumTab[] {
  return restoreReleased ? MEDIUM_TABS : MEDIUM_TABS.filter((tab) => tab.value !== "restore");
}

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

function MediumPanel({ medium }: { medium: EnhanceMedium }) {
  if (medium === "video") {
    return <VideoPanel />;
  }
  if (medium === "restore") {
    return <PhotoRestorePanel />;
  }
  return <ImagePanel />;
}

// WAI-ARIA APG tabs pattern: Right/Left roving focus wraps around the tab
// list and activates immediately (automatic selection model), matching the
// click behavior these tabs already have.
function resolveNextTabIndex(currentIndex: number, key: string, tabCount: number): number | null {
  if (key === "ArrowRight") {
    return (currentIndex + 1) % tabCount;
  }
  if (key === "ArrowLeft") {
    return (currentIndex - 1 + tabCount) % tabCount;
  }
  return null;
}

// La URL fija el punto de ENTRADA (el arbol de capacidades entra directo a
// /enhance/video) y de ahi en mas las pestañas son estado local. Navegar en cada
// pestaña metaria una entrada de historial por cada flecha del teclado, y el
// componente ademas queda testeable sin router.
export function EnhancePage({
  initialMedium = "image",
}: {
  initialMedium?: EnhanceMedium;
}) {
  const { t } = useTranslation();
  const [medium, setMedium] = useState<EnhanceMedium>(initialMedium);
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const tabs = visibleTabs(useRestoreReleased());
  const activeTab = tabs.find((tab) => tab.value === medium) ?? tabs[0];

  function handleTabKeyDown(event: KeyboardEvent<HTMLButtonElement>, currentIndex: number): void {
    const nextIndex = resolveNextTabIndex(currentIndex, event.key, tabs.length);
    if (nextIndex === null) {
      return;
    }
    event.preventDefault();
    const nextTab = tabs[nextIndex];
    setMedium(nextTab.value);
    tabRefs.current[nextIndex]?.focus();
  }

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="font-heading text-2xl font-semibold text-text">Enhance</h1>
        <p className="mt-1 text-sm text-text-dim">{t(activeTab.subtitleKey)}</p>
      </div>
      <div role="tablist" aria-label="Enhance medium" className="flex w-fit gap-2">
        {tabs.map((tab, index) => {
          const isActive = activeTab.value === tab.value;
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
              {t(tab.labelKey)}
            </button>
          );
        })}
      </div>
      <div id={panelId(activeTab.value)} role="tabpanel" aria-labelledby={tabId(activeTab.value)} tabIndex={0}>
        <MediumPanel medium={activeTab.value} />
      </div>
    </div>
  );
}
