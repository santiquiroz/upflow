import { Ruler } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "../../i18n/LocaleProvider";
import { estimatePrintSize, type SizeEstimate } from "../../services/print";

// Palabras que pone la cámara o el gestor de archivos, no el usuario: no
// describen ningún objeto. "IMG_20260808_143255.jpg" no es una pista.
const CAMERA_NOISE = new Set([
  "img",
  "image",
  "images",
  "imagen",
  "dsc",
  "dscn",
  "pxl",
  "photo",
  "foto",
  "picture",
  "pic",
  "screenshot",
  "captura",
  "whatsapp",
  "copy",
  "copia",
  "final",
]);

const MIN_HINT_LETTERS = 3;

/**
 * Lo que el nombre de un archivo dice sobre el objeto, si dice algo.
 *
 * En el carril de foto no hay descripción: el nombre es la única señal. Cuando
 * no la hay, devuelve vacío y no se ofrece estimar — pedirle al modelo que
 * adivine sobre nada da una medida inventada, que es peor que el valor por
 * defecto.
 */
export function objectHintFromFileName(name: string): string {
  const withoutExtension = name.replace(/\.[^.]+$/, "");
  const words = withoutExtension
    .split(/[^\p{L}\p{N}]+/u)
    .filter((word) => word.length > 1)
    .filter((word) => !/^\d+$/.test(word))
    .filter((word) => !CAMERA_NOISE.has(word.toLowerCase()));
  const hint = words.join(" ").trim();
  const letters = hint.match(/\p{L}/gu) ?? [];
  return letters.length >= MIN_HINT_LETTERS ? hint : "";
}

/**
 * Ofrece cuánto mide el objeto REAL. Nunca lo aplica: el número entra al campo
 * de medida solo cuando el usuario hace clic, y hasta entonces lo que haya
 * tipeado queda intacto.
 */
export function SizeSuggestion({
  hint,
  onAccept,
}: {
  hint: string;
  onAccept: (longestMm: number, reference: string) => void;
}) {
  const { t } = useTranslation();
  const [estimate, setEstimate] = useState<SizeEstimate | null>(null);
  const [estimatedFor, setEstimatedFor] = useState("");
  const [state, setState] = useState<"idle" | "loading" | "failed">("idle");

  async function handleEstimate() {
    setState("loading");
    setEstimate(null);
    try {
      setEstimate(await estimatePrintSize(hint));
      setEstimatedFor(hint);
      setState("idle");
    } catch {
      // Que no haya sugerencia no es un error del flujo: la medida por defecto
      // sigue valiendo y el trabajo se puede generar igual.
      setState("failed");
    }
  }

  // Una sugerencia para "una taza" no puede quedar en pantalla cuando el pedido
  // ya dice "un tornillo M3".
  const fresh = estimate !== null && estimatedFor === hint;

  return (
    <div className="flex flex-col gap-2">
      <button
        type="button"
        onClick={() => void handleEstimate()}
        disabled={hint.length === 0 || state === "loading"}
        className="inline-flex w-fit items-center gap-2 rounded border border-border bg-surface px-3 py-1.5 text-sm text-text hover:border-text-faint disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        <Ruler aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
        {state === "loading" ? t("gen3d.estimate.working") : t("gen3d.estimate.action")}
      </button>

      {state === "failed" && (
        <p role="status" className="text-xs text-text-dim">
          {t("gen3d.estimate.failed")}
        </p>
      )}

      {fresh && estimate && (
        <div className="flex flex-col items-start gap-1 rounded border border-border bg-surface-2 px-3 py-2">
          <p className="text-sm text-text">
            {t("gen3d.estimate.suggestion", { mm: estimate.longestMm })}
          </p>
          {estimate.reference && (
            <p className="text-xs text-text-dim">
              {t("gen3d.estimate.reference", { reference: estimate.reference })}
            </p>
          )}
          <p className="text-xs text-text-faint">{t("gen3d.estimate.disclaimer")}</p>
          <button
            type="button"
            onClick={() => onAccept(estimate.longestMm, estimate.reference)}
            className="mt-1 inline-flex items-center gap-2 rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg hover:bg-accent-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
          >
            {t("gen3d.estimate.apply", { mm: estimate.longestMm })}
          </button>
        </div>
      )}
    </div>
  );
}
