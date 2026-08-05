import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "../i18n/LocaleProvider";
import { LogOut, User } from "lucide-react";
import { useState } from "react";
import { useAuth } from "../hooks/useAuth";
import { logout } from "../services/auth";

export function Header() {
  const { t } = useTranslation();
  const { me } = useAuth();
  const queryClient = useQueryClient();
  const [menuOpen, setMenuOpen] = useState(false);

  if (!me || me.authMode === "off") {
    return null;
  }

  async function handleLogout() {
    await logout();
    await queryClient.invalidateQueries({ queryKey: ["me"] });
  }

  return (
    <div className="relative flex items-center gap-2">
      <button
        type="button"
        onClick={() => setMenuOpen((open) => !open)}
        aria-label={t("common.userMenu")}
        className="flex items-center gap-1.5 rounded-sm px-2 py-1 text-xs text-text-dim transition-colors duration-fast hover:text-text"
      >
        <User aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
        {me.username}
      </button>
      {menuOpen && (
        <div
          role="menu"
          className="absolute right-0 top-full z-10 mt-1 flex flex-col gap-1 rounded border border-border bg-surface p-2 shadow-lg"
        >
          <button
            type="button"
            role="menuitem"
            onClick={handleLogout}
            className="flex items-center gap-1.5 px-2 py-1 text-left text-xs text-text-dim transition-colors duration-fast hover:text-text"
          >
            <LogOut aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
            Log out
          </button>
        </div>
      )}
    </div>
  );
}
