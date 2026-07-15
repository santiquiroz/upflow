// Ports the formatFps() algorithm pinned by tests/test_web_ui.py (app/templates/index.html)
// so the React UI normalizes outputFps identically to the legacy Jinja page.

function isFraction(value: string): boolean {
  return value.includes("/");
}

export function formatFps<T extends string | null | undefined>(rawValue: T): T {
  if (typeof rawValue !== "string" || !isFraction(rawValue)) {
    return rawValue;
  }
  const [numerator, denominator] = rawValue.split("/").map(Number);
  if (!denominator) {
    return rawValue as T;
  }
  const decimal = numerator / denominator;
  return (Number.isInteger(decimal) ? String(decimal) : decimal.toFixed(2)) as T;
}
