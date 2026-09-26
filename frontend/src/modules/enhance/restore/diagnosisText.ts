import type { RestoreFinding } from "../../../lib/restoreApiTypes";

export interface FindingText {
  key: string;
  params: Record<string, string | number>;
}

// translate() no pluraliza: "1 faces found" necesita su propia clave.
function reasonKey(finding: RestoreFinding): string {
  if (finding.reasonKey === "restore.diag.faces" && finding.params.count === 1) {
    return "restore.diag.faces.one";
  }
  return finding.reasonKey;
}

// El backend nombra la dominante en ingles ("cyan/green"); la clave la traduce.
export function castKey(cast: string): string {
  return `restore.diag.cast.${cast.replace(/\W+/g, "_")}`;
}

export function findingText(finding: RestoreFinding, translate: (key: string) => string): FindingText {
  const { cast } = finding.params;
  const params = typeof cast === "string" ? { ...finding.params, cast: translate(castKey(cast)) } : finding.params;
  return { key: reasonKey(finding), params };
}
