import { useQuery } from "@tanstack/react-query";
import { createContext, useContext, type ReactNode } from "react";
import { ApiError } from "../lib/api";
import type { MeResponse } from "../lib/apiTypes";
import { getMe } from "../services/auth";

interface AuthContextValue {
  me: MeResponse | undefined;
  isLoading: boolean;
  isError: boolean;
  needsSetup: boolean;
  hasPermission: (permission: string) => boolean;
  refetch: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function isSetupRequiredError(error: unknown): boolean {
  return error instanceof ApiError && error.message === "setup_required";
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const query = useQuery({ queryKey: ["me"], queryFn: getMe, retry: false });

  function hasPermission(permission: string): boolean {
    return query.data?.permissions.includes(permission) ?? false;
  }

  const value: AuthContextValue = {
    me: query.data,
    isLoading: query.isLoading,
    isError: query.isError,
    needsSetup: isSetupRequiredError(query.error),
    hasPermission,
    refetch: () => void query.refetch(),
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
