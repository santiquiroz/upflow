import type { ReactNode } from "react";
import { useTranslation } from "../../i18n/LocaleProvider";

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
      className="max-h-[calc(100dvh-12rem)] min-h-0 w-[340px] flex-none overflow-y-auto rounded-lg border border-border bg-surface p-3"
    >
      {children}
    </aside>
  );
}
