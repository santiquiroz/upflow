import type { ReactNode } from "react";
import { useTranslation } from "../../i18n/LocaleProvider";

const EDITOR_SIDE_PANEL_MAX_HEIGHT = "calc(100dvh - 12rem)";

export type EditorSidePanelView = "people" | "action" | "review";

export function EditorSidePanel({
  view,
  children,
}: {
  view: EditorSidePanelView;
  children: ReactNode;
}) {
  const { t } = useTranslation();
  return (
    <aside
      aria-label={t("editor.sidePanel.label")}
      data-view={view}
      style={{ maxHeight: EDITOR_SIDE_PANEL_MAX_HEIGHT }}
      className="min-h-0 min-w-0 w-full flex-none overflow-x-hidden overflow-y-auto rounded-lg border border-border bg-surface p-3 lg:w-[340px]"
    >
      {children}
    </aside>
  );
}
