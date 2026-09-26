export function versionedUrl(url: string, revision: number): string {
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}v=${revision}`;
}

// Recomponer reescribe los archivos del job con la misma URL: sin la version el
// navegador mostraria (o bajaria) la copia de antes.
export function revisedUrl(url: string, revision: number): string {
  return revision === 0 ? url : versionedUrl(url, revision);
}
