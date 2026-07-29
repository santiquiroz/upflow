import { useTranslation } from "../../i18n/LocaleProvider";
import type { VoiceDelivery } from "../../lib/apiTypes";

interface DeliveryTargetPickerProps {
  deliveries: VoiceDelivery[];
  value: string | null;
  onChange: (id: string) => void;
}

export function DeliveryTargetPicker({
  deliveries,
  value,
  onChange,
}: DeliveryTargetPickerProps) {
  const { t } = useTranslation();

  return (
    <fieldset className="flex flex-col gap-1.5">
      <legend className="pb-1 text-xs font-medium text-text-dim">
        {t("voice.delivery.legend")}
      </legend>
      {deliveries.map((delivery) => {
        const active = value === delivery.id;
        return (
          <label
            key={delivery.id}
            className={`flex cursor-pointer gap-2.5 rounded border px-2.5 py-2 transition-[background-color,border-color] duration-fast motion-reduce:transition-none ${
              active ? "border-accent bg-surface" : "border-border/60 hover:border-border"
            }`}
          >
            <input
              type="radio"
              name="voice-delivery"
              value={delivery.id}
              checked={active}
              onChange={() => onChange(delivery.id)}
              className="mt-1 h-3 w-3 shrink-0 accent-accent"
            />
            <span className="flex min-w-0 flex-1 flex-col gap-1">
              <span className="flex flex-wrap items-baseline justify-between gap-2">
                <span className="text-xs font-medium text-text">
                  {t(delivery.labelKey)}
                </span>
                {/* Cifras tabulares: los numeros de las cuatro filas quedan
                    alineados en columna, que es como un productor los compara.
                    No van escondidos porque son la razon de elegir un destino. */}
                <span className="shrink-0 font-mono-tabular text-[10px] text-text-faint">
                  {delivery.lufs} LUFS · {delivery.truePeakDb} dBTP
                </span>
              </span>
              <span className="text-[11px] leading-relaxed text-text-faint">
                {t(delivery.descriptionKey)}
              </span>
            </span>
          </label>
        );
      })}
    </fieldset>
  );
}
