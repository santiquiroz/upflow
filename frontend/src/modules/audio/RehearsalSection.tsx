import { AccordionSection } from "../../components/AccordionSection";
import { useTranslation } from "../../i18n/LocaleProvider";
import type { SeparationModel } from "../../lib/apiTypes";
import { GUIDE_PERCENT_LEVELS, type RehearsalSelection } from "./useRehearsalSelection";

function guideButtonClassName(isActive: boolean): string {
  const base =
    "inline-flex items-center gap-2 rounded-sm border px-3 py-1.5 text-sm transition-[background-color,border-color,color] duration-fast focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";
  if (isActive) {
    return `${base} border-accent bg-accent text-bg`;
  }
  return `${base} border-border bg-surface text-text-dim hover:border-text-faint hover:text-text`;
}

function StemChips({
  model,
  selection,
}: {
  model: SeparationModel;
  selection: RehearsalSelection;
}) {
  const { t } = useTranslation();
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
        {t("audio.rehearsal.stemsLegend")}
      </legend>
      <div className="flex flex-wrap gap-2">
        {model.stems.map((stem) => (
          <label
            key={stem.id}
            className="flex items-center gap-1.5 rounded border border-border bg-surface-2 px-2 py-1 text-xs text-text"
          >
            <input
              type="checkbox"
              checked={selection.isEnabled(stem.id)}
              onChange={(event) => selection.toggleStem(stem.id, event.target.checked)}
              className="h-3.5 w-3.5 accent-accent"
            />
            {t(stem.labelKey)}
          </label>
        ))}
      </div>
    </fieldset>
  );
}

function GuideLevelControl({ selection }: { selection: RehearsalSelection }) {
  const { t } = useTranslation();
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
        {t("audio.rehearsal.guideLegend")}
      </legend>
      <div className="flex flex-wrap gap-2">
        {GUIDE_PERCENT_LEVELS.map((level) => (
          <button
            key={level}
            type="button"
            aria-pressed={selection.guidePercent === level}
            className={guideButtonClassName(selection.guidePercent === level)}
            onClick={() => selection.setGuidePercent(level)}
          >
            {level === 0 ? t("audio.rehearsal.guideNone") : `${level}%`}
          </button>
        ))}
      </div>
    </fieldset>
  );
}

/**
 * Pistas minus-one para ensayar: por cada stem elegido el backend hornea la
 * mezcla completa SIN ese instrumento, con guía opcional a bajo volumen.
 * Solo existe cuando el modelo elegido separa tres pistas o más — con dos, "la
 * mezcla sin una" es el otro stem y eso ya lo da el karaoke.
 */
export function RehearsalSection({
  model,
  selection,
}: {
  model: SeparationModel | undefined;
  selection: RehearsalSelection;
}) {
  const { t } = useTranslation();
  if (!model || !selection.available) {
    return null;
  }
  const count = selection.enabledStems.length;

  return (
    <AccordionSection
      title={t("audio.rehearsal.title")}
      summary={count > 0 ? t("audio.rehearsal.summary", { count }) : t("audio.mode.none")}
      tooltip={t("audio.rehearsal.tooltip")}
    >
      <div className="flex flex-col gap-3">
        <label className="flex items-center gap-2 text-sm text-text">
          <input
            type="checkbox"
            checked={selection.active}
            onChange={(event) => selection.setActive(event.target.checked)}
            className="h-4 w-4 accent-accent"
          />
          {t("audio.rehearsal.toggle")}
        </label>
        {selection.active && (
          <>
            <StemChips model={model} selection={selection} />
            <GuideLevelControl selection={selection} />
            {/* Lo imperfecto se dice ANTES de pedir el trabajo: un stem que
                separó débil deja un fantasma del instrumento en la minus-one. */}
            <p className="text-xs text-text-faint">{t("audio.rehearsal.help")}</p>
          </>
        )}
      </div>
    </AccordionSection>
  );
}
