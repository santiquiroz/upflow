import { CheckCircle2, TriangleAlert, X } from "lucide-react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { CctvBox } from "../../../services/cctv";

export interface OsdPanelProps {
  boxes: readonly CctvBox[];
  confirmed: boolean;
  confirmedFrame: number | null;
  noOsd: boolean;
  flagged: readonly number[];
  checking: boolean;
  checkError: string | null;
  onConfirm: () => void;
  onNoOsdChange: (noOsd: boolean) => void;
  onUseSuggested: () => void;
  onRemove: (index: number) => void;
}

const SECONDARY_BUTTON_CLASS =
  "rounded-sm border border-border bg-surface px-3 py-1.5 text-sm text-text transition-colors duration-fast hover:border-accent disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";

function BoxRow({ box, index, isFlagged, onRemove }: { box: CctvBox; index: number; isFlagged: boolean; onRemove: () => void }) {
  const { t } = useTranslation();
  return (
    <li className="flex flex-col gap-1">
      <div className="flex items-center gap-2 text-xs text-text">
        <span className="font-mono-tabular">
          {t("cctv.box.row", { index: index + 1, x: box[0], y: box[1], w: box[2], h: box[3] })}
        </span>
        <button
          type="button"
          onClick={onRemove}
          aria-label={t("cctv.box.remove", { index: index + 1 })}
          className="inline-flex h-5 w-5 items-center justify-center rounded-sm text-text-dim hover:text-danger focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          <X aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
        </button>
      </div>
      {isFlagged && (
        <p className="flex items-center gap-1.5 text-xs text-warn">
          <TriangleAlert aria-hidden="true" className="h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
          {t("cctv.osd.notText")}
        </p>
      )}
    </li>
  );
}

function ConfirmationStatus({ props }: { props: OsdPanelProps }) {
  const { t } = useTranslation();
  if (props.checkError) {
    return <p role="alert" className="text-xs text-danger">{props.checkError}</p>;
  }
  if (props.checking) {
    return <p role="status" className="text-xs text-text-dim">{t("cctv.osd.checking")}</p>;
  }
  if (!props.confirmed || props.confirmedFrame === null) {
    return null;
  }
  return (
    <p role="status" className="flex items-center gap-1.5 text-xs text-ok">
      <CheckCircle2 aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
      {t("cctv.osd.confirmed", { frame: props.confirmedFrame })}
    </p>
  );
}

function BoxList({ props }: { props: OsdPanelProps }) {
  const { t } = useTranslation();
  return (
    <>
      <p className="text-xs text-text-faint">{t("cctv.osd.suggested")}</p>
      <ol aria-label={t("cctv.osd.legend")} className="flex flex-col gap-1.5">
        {props.boxes.map((box, index) => (
          <BoxRow
            key={index}
            box={box}
            index={index}
            isFlagged={props.flagged.includes(index)}
            onRemove={() => props.onRemove(index)}
          />
        ))}
      </ol>
      <div className="flex flex-wrap gap-2">
        <button type="button" onClick={props.onConfirm} disabled={props.boxes.length === 0 || props.checking} className={SECONDARY_BUTTON_CLASS}>
          {t("cctv.osd.confirmAction")}
        </button>
        <button type="button" onClick={props.onUseSuggested} className={SECONDARY_BUTTON_CLASS}>
          {t("cctv.osd.useSuggested")}
        </button>
      </div>
      <ConfirmationStatus props={props} />
    </>
  );
}

export function CctvOsdPanel(props: OsdPanelProps) {
  const { t } = useTranslation();
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">{t("cctv.osd.legend")}</legend>
      <p className="text-xs text-text-dim">{t("cctv.osd.confirm")}</p>
      {!props.noOsd && <BoxList props={props} />}
      <label className="flex items-center gap-2 text-sm text-text">
        <input
          type="checkbox"
          checked={props.noOsd}
          onChange={(event) => props.onNoOsdChange(event.target.checked)}
          className="h-3.5 w-3.5 accent-accent"
        />
        {t("cctv.osd.none")}
      </label>
    </fieldset>
  );
}
