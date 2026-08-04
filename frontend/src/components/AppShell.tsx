import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { useAuth } from "../hooks/useAuth";
import { useTranslation } from "../i18n/LocaleProvider";
import { NAV_ENTRIES } from "../lib/navigation";
import { Header } from "./Header";
import { JobQueue } from "./JobQueue";
import { UpdateBanner } from "./UpdateBanner";

interface AppShellProps {
  children: ReactNode;
}

const NAV_LINK_BASE =
  "flex items-center gap-3 rounded px-3 py-2 text-sm font-body text-text-dim " +
  "transition-[background-color,color] duration-fast hover:bg-surface-2 hover:text-text " +
  "focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";

const NAV_LINK_ACTIVE = "bg-surface-2 text-text before:absolute before:inset-y-0 before:left-0 before:w-[3px] before:bg-accent";

function navLinkClassName({ isActive }: { isActive: boolean }): string {
  return `relative ${NAV_LINK_BASE} ${isActive ? NAV_LINK_ACTIVE : ""}`;
}

export function AppShell({ children }: AppShellProps) {
  const { hasPermission } = useAuth();
  const { t } = useTranslation();
  const visibleEntries = NAV_ENTRIES.filter(
    (entry) => !entry.requiredPermission || hasPermission(entry.requiredPermission),
  );
  return (
    <div className="flex h-screen flex-col">
      <UpdateBanner />
      <div className="grid min-h-0 flex-1 grid-cols-[240px_1fr_320px] max-[900px]:grid-cols-[72px_1fr_320px]">
        <aside
          aria-label={t("nav.mainLabel")}
          className="flex flex-col gap-1 border-r border-border bg-surface p-2"
        >
          <div className="flex items-center justify-between px-2 py-4 max-[900px]:hidden">
            <span className="font-heading text-lg font-semibold tracking-tight text-text">Upflow</span>
            <Header />
          </div>
          <nav className="flex flex-col gap-1">
            {visibleEntries.map((entry) => {
              const Icon = entry.icon;
              const label = t(entry.labelKey);
              return (
                // El `title` es lo unico que le queda al usuario con mouse bajo
                // los 900px, donde la etiqueta pasa a sr-only y quedan once
                // iconos parecidos entre si (Audio y Transcribe casi identicos).
                <NavLink
                  key={entry.path}
                  to={entry.path}
                  end={entry.path === "/"}
                  title={label}
                  className={navLinkClassName}
                >
                  <Icon aria-hidden="true" className="h-[18px] w-[18px] shrink-0" strokeWidth={1.75} />
                  <span className="max-[900px]:sr-only">{label}</span>
                </NavLink>
              );
            })}
          </nav>
        </aside>
        <main className="overflow-y-auto p-6">
          <div className="mx-auto w-full max-w-[1200px]">{children}</div>
        </main>
        <aside aria-label={t("nav.queueLabel")} className="border-l border-border bg-surface p-4">
          <JobQueue />
        </aside>
      </div>
    </div>
  );
}
