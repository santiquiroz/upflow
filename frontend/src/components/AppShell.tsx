import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { NAV_ENTRIES } from "../lib/navigation";
import { JobQueue } from "./JobQueue";

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
  return (
    <div className="grid h-screen grid-cols-[240px_1fr_320px] max-[900px]:grid-cols-[72px_1fr_320px]">
      <aside aria-label="Main navigation" className="flex flex-col gap-1 border-r border-border bg-surface p-2">
        <div className="px-2 py-4 font-heading text-lg font-semibold tracking-tight text-text max-[900px]:hidden">
          Upflow
        </div>
        <nav className="flex flex-col gap-1">
          {NAV_ENTRIES.map((entry) => {
            const Icon = entry.icon;
            return (
              <NavLink key={entry.path} to={entry.path} end={entry.path === "/"} className={navLinkClassName}>
                <Icon aria-hidden="true" className="h-[18px] w-[18px] shrink-0" strokeWidth={1.75} />
                <span className="max-[900px]:sr-only">{entry.label}</span>
              </NavLink>
            );
          })}
        </nav>
      </aside>
      <main className="overflow-y-auto p-6">
        <div className="mx-auto w-full max-w-[1200px]">{children}</div>
      </main>
      <aside aria-label="Job queue" className="border-l border-border bg-surface p-4">
        <JobQueue />
      </aside>
    </div>
  );
}
