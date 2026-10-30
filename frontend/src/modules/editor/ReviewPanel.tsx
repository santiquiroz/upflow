import { AlertTriangle, CheckCircle2, ChevronDown, Loader2, RotateCcw } from "lucide-react";
import { useId, useState, type FormEvent } from "react";
import { useTranslation } from "../../i18n/LocaleProvider";
import type { EraseZone } from "../../lib/apiTypes";
import type { EditorPerson } from "../../services/editor";
import type { Choice } from "./peopleState";
import { cropStyle, engineLabel, zonePeople, zoneStatus, type RedoEngine } from "./zones";

interface ImageSize {
  width: number;
  height: number;
}

interface ReviewPanelProps {
  zones: EraseZone[];
  people: EditorPerson[];
  choices: Record<number, Choice>;
  hoveredZone: number | null;
  onHoverZone: (index: number | null) => void;
  onRedo: (zone: EraseZone, engine: RedoEngine, prompt?: string) => void;
  sdxlAvailable: boolean;
  eraserAvailable?: boolean;
  redoingIndex?: number | null;
  errors?: Record<number, string>;
  beforeUrl?: string | null;
  afterUrl?: string | null;
  imageSize?: ImageSize;
}

type RowProps = Omit<ReviewPanelProps, "zones" | "hoveredZone"> & { zone: EraseZone; hovered: boolean };

const ERASER_ENGINES: RedoEngine[] = ["lama", "osor", "lama+osor"];

function StatusLine({ zone }: { zone: EraseZone }) {
  const { t } = useTranslation();
  const status = zoneStatus(zone);
  const Icon = status === "clean" ? CheckCircle2 : AlertTriangle;
  const color = status === "clean" ? "text-ok" : status === "residual" ? "text-danger" : "text-accent";
  return (
    <p className="mt-1 flex items-center gap-1.5 text-xs text-text">
      <Icon aria-hidden="true" className={`h-3.5 w-3.5 shrink-0 ${color}`} />
      {t(`editor.review.status.${status}`)}
    </p>
  );
}

function BeforeAfter({ zone, beforeUrl, afterUrl, imageSize }: Pick<RowProps, "zone" | "beforeUrl" | "afterUrl" | "imageSize">) {
  const { t } = useTranslation();
  if (!zone.box || !beforeUrl || !afterUrl || !imageSize) return null;
  const crops = [
    { label: t("editor.review.before"), url: beforeUrl },
    { label: t("editor.review.after"), url: afterUrl },
  ];
  return (
    <div className="mt-2 flex gap-2" data-testid={`zone-compare-${zone.index}`}>
      {crops.map((crop) => (
        <figure key={crop.label} className="flex flex-col items-center gap-1">
          <div role="img" aria-label={crop.label} className="rounded border border-border" style={cropStyle(crop.url, zone.box!, imageSize, 120)} />
          <figcaption className="text-[11px] text-text-dim">{crop.label}</figcaption>
        </figure>
      ))}
    </div>
  );
}

function DescribeForm({ onSubmit }: { onSubmit: (text: string) => void }) {
  const { t } = useTranslation();
  const [text, setText] = useState("");
  const id = useId();
  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (text.trim()) onSubmit(text.trim());
  }
  return (
    <form onSubmit={handleSubmit} className="mt-2 flex flex-col gap-1.5">
      <label htmlFor={id} className="text-xs text-text-dim">{t("editor.review.describeLabel")}</label>
      <input
        id={id}
        autoFocus
        value={text}
        onChange={(event) => setText(event.target.value)}
        className="min-h-9 rounded border border-border bg-bg px-2 text-sm text-text"
      />
      <button
        type="submit"
        disabled={!text.trim()}
        className="min-h-9 rounded border border-border px-3 text-xs text-text hover:bg-bg disabled:opacity-50"
      >
        {t("editor.review.describeSubmit")}
      </button>
    </form>
  );
}

function MenuItem({ label, disabled, onSelect }: { label: string; disabled: boolean; onSelect: () => void }) {
  return (
    <button
      type="button"
      role="menuitem"
      disabled={disabled}
      onClick={onSelect}
      className="min-h-9 w-full rounded px-3 text-left text-xs text-text hover:bg-bg disabled:opacity-50"
    >
      {label}
    </button>
  );
}

function RedoOptions({ sdxlAvailable, eraserAvailable, onPick }: {
  sdxlAvailable: boolean;
  eraserAvailable: boolean;
  onPick: (engine: RedoEngine) => void;
}) {
  const { t } = useTranslation();
  return (
    <div role="menu" aria-label={t("editor.review.redoOptions")} className="mt-1 rounded border border-border bg-surface p-1">
      {ERASER_ENGINES.map((engine) => (
        <MenuItem key={engine} label={engineLabel(engine, t)} disabled={!eraserAvailable} onSelect={() => onPick(engine)} />
      ))}
      {!eraserAvailable && <p className="px-3 py-1 text-[11px] text-text-dim">{t("editor.review.needsEraser")}</p>}
      <MenuItem label={t("editor.review.describe")} disabled={!sdxlAvailable} onSelect={() => onPick("sdxl")} />
      {!sdxlAvailable && <p className="px-3 py-1 text-[11px] text-text-dim">{t("editor.replace.needsSdxl")}</p>}
    </div>
  );
}

function RedoControl({ zone, onRedo, sdxlAvailable, eraserAvailable = true, redoingIndex = null }: RowProps) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);
  const [describing, setDescribing] = useState(false);
  const redoing = redoingIndex === zone.index;
  const blocked = !zone.box || redoingIndex !== null;
  function pick(engine: RedoEngine) {
    setOpen(false);
    if (engine === "sdxl") setDescribing(true);
    else onRedo(zone, engine);
  }
  function describe(text: string) {
    setDescribing(false);
    onRedo(zone, "sdxl", text);
  }
  return (
    <div className="mt-2">
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        disabled={blocked}
        onClick={() => setOpen((current) => !current)}
        className="flex min-h-9 items-center gap-1.5 rounded border border-border px-3 text-xs text-text hover:bg-bg disabled:opacity-50"
      >
        {redoing
          ? <Loader2 aria-hidden="true" className="h-3.5 w-3.5 animate-spin" />
          : <RotateCcw aria-hidden="true" className="h-3.5 w-3.5" />}
        {redoing ? t("editor.review.redoing") : t("editor.review.redo")}
        {!redoing && <ChevronDown aria-hidden="true" className="h-3.5 w-3.5" />}
      </button>
      {!zone.box && <p className="mt-1 text-[11px] text-text-dim">{t("editor.review.noBox")}</p>}
      {open && !blocked && <RedoOptions sdxlAvailable={sdxlAvailable} eraserAvailable={eraserAvailable} onPick={pick} />}
      {describing && !blocked && <DescribeForm onSubmit={describe} />}
    </div>
  );
}

function ZoneRow(props: RowProps) {
  const { t } = useTranslation();
  const { zone, people, choices, hovered, onHoverZone, errors } = props;
  const number = zone.index + 1;
  const names = zonePeople(zone, people, choices, (value) => t("editor.people.person", { number: value }));
  const error = errors?.[zone.index];
  return (
    <li
      aria-label={t("editor.review.zone", { number })}
      data-hovered={hovered ? "true" : "false"}
      onMouseEnter={() => onHoverZone(zone.index)}
      onMouseLeave={() => onHoverZone(null)}
      className={`rounded-lg border bg-surface-2 p-3 ${hovered ? "border-accent" : "border-border"}`}
    >
      <div className="flex items-center gap-2">
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-bg text-xs font-semibold text-text">
          {number}
        </span>
        <span className="min-w-0 flex-1 truncate text-sm text-text">
          {names.length > 0 ? names.join(", ") : t("editor.review.zone", { number })}
        </span>
        <span className="text-[11px] text-text-dim">{engineLabel(zone.engine, t)}</span>
      </div>
      <StatusLine zone={zone} />
      {hovered && <BeforeAfter {...props} />}
      <RedoControl {...props} />
      {error && <p role="alert" className="mt-1 text-xs text-danger">{error}</p>}
    </li>
  );
}

export function ReviewPanel({ zones, hoveredZone, ...rest }: ReviewPanelProps) {
  const { t } = useTranslation();
  return (
    <section className="mb-3">
      <h2 className="text-sm font-semibold text-text">{t("editor.review.title")}</h2>
      <p className="mt-1 text-xs text-text-dim">{t("editor.review.hoverHint")}</p>
      <ul aria-label={t("editor.review.title")} className="mt-2 flex max-h-[50vh] flex-col gap-2 overflow-y-auto">
        {zones.map((zone) => (
          <ZoneRow key={zone.index} zone={zone} hovered={hoveredZone === zone.index} {...rest} />
        ))}
      </ul>
    </section>
  );
}
