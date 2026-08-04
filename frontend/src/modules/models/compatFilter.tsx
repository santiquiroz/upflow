import { useTranslation } from "../../i18n/LocaleProvider";
import type { HfModelSearchResultResponse } from "../../lib/apiTypes";

// Filtro de compatibilidad compartido por TODOS los buscadores de la app
// (generación, escalado y transcripción). Nació en el buscador de generación y
// vivía solo ahí; los otros dos mostraban el veredicto en cada tarjeta pero
// obligaban a leerlas una por una para encontrar las que corren tal cual.

export type CompatFilter = "all" | "ready" | "conversion";

const COMPAT_FILTERS: { value: CompatFilter; labelKey: string }[] = [
  { value: "all", labelKey: "models.compat.all" },
  { value: "ready", labelKey: "models.compat.ready" },
  { value: "conversion", labelKey: "models.compat.conversion" },
];

// ready_onnx corre tal cual; needs_conversion y single_file pasan por la
// conversión. El resto (gated, incompatible) solo aparece en "Todos", con su
// motivo en la tarjeta.
export function matchesCompatFilter(
  result: Pick<HfModelSearchResultResponse, "compat">,
  filter: CompatFilter,
): boolean {
  if (filter === "all") {
    return true;
  }
  if (filter === "ready") {
    return result.compat === "ready_onnx";
  }
  return result.compat === "needs_conversion" || result.compat === "single_file";
}

export function CompatFilterChips({
  value,
  onChange,
  name,
}: {
  value: CompatFilter;
  onChange: (next: CompatFilter) => void;
  // Un name por buscador: dos grupos de radios con el mismo name se pisan
  // cuando conviven en la misma pantalla.
  name: string;
}) {
  const { t } = useTranslation();
  return (
    <fieldset className="flex flex-wrap gap-2">
      <legend className="sr-only">{t("models.compat.legend")}</legend>
      {COMPAT_FILTERS.map((option) => (
        <label
          key={option.value}
          className={`cursor-pointer rounded border px-3 py-1.5 text-sm ${
            value === option.value
              ? "border-accent bg-surface-2 text-text"
              : "border-border bg-surface text-text-dim"
          }`}
        >
          <input
            type="radio"
            name={name}
            className="sr-only"
            checked={value === option.value}
            onChange={() => onChange(option.value)}
          />
          {t(option.labelKey)}
        </label>
      ))}
    </fieldset>
  );
}
