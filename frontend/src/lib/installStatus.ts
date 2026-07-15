const TERMINAL_INSTALL_STATUSES: readonly string[] = ["installed", "error"];

export function isTerminalInstallStatus(status: string): boolean {
  return TERMINAL_INSTALL_STATUSES.includes(status);
}
