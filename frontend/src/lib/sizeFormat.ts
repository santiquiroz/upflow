const BYTE_UNITS = ["B", "KB", "MB", "GB", "TB"] as const;
const BYTES_PER_UNIT_STEP = 1024;

function unitExponent(bytes: number): number {
  if (bytes < BYTES_PER_UNIT_STEP) {
    return 0;
  }
  const maxExponent = BYTE_UNITS.length - 1;
  const exponent = Math.floor(Math.log(bytes) / Math.log(BYTES_PER_UNIT_STEP));
  return Math.min(exponent, maxExponent);
}

export function formatModelSize(bytes: number): string {
  const exponent = unitExponent(bytes);
  const value = bytes / BYTES_PER_UNIT_STEP ** exponent;
  const decimals = exponent === 0 ? 0 : 1;
  return `${value.toFixed(decimals)} ${BYTE_UNITS[exponent]}`;
}
