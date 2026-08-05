const BYTES_PER_MB = 1024 * 1024;
const MB_PER_GB = 1024;

// Misma cuenta que `storage.save_upload` en el backend, incluido el `>`: el
// archivo que da JUSTO en el limite entra. Usar `>=` aca haria que la app
// rechace archivos que el servidor si acepta.
export function exceedsUploadLimit(sizeBytes: number, limitMb: number | null): boolean {
  if (limitMb === null) {
    return false;
  }
  return sizeBytes > limitMb * BYTES_PER_MB;
}

export function formatMegabytes(sizeBytes: number): string {
  const megabytes = sizeBytes / BYTES_PER_MB;
  if (megabytes < MB_PER_GB) {
    return `${Math.round(megabytes)} MB`;
  }
  const gigabytes = megabytes / MB_PER_GB;
  const rounded = Number.isInteger(gigabytes) ? gigabytes : Number(gigabytes.toFixed(1));
  return `${rounded} GB`;
}
