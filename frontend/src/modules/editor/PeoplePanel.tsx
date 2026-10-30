import { Check, X } from "lucide-react";
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type MutableRefObject,
} from "react";
import { useTranslation } from "../../i18n/LocaleProvider";
import type { EditorPerson } from "../../services/editor";
import { countRemove, groupPeople, personNumbers, type Choice } from "./peopleState";
import type { SilhouetteProgress } from "./useSilhouetteBatch";

type PeopleStatus = "loading" | "ready" | "empty" | "error" | "unavailable";

interface PeoplePanelProps {
  people: EditorPerson[];
  choices: Record<number, Choice>;
  hoveredId: number | null;
  onHover: (id: number | null) => void;
  onToggle: (id: number) => void;
  onGroup: (role: EditorPerson["role"], choice: Choice) => void;
  protectKept: boolean;
  onProtectKeptChange: (checked: boolean) => void;
  onAccept: () => void;
  onRetry: () => void;
  status: PeopleStatus;
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  unavailableReason?: string;
  error?: string;
  progress?: SilhouetteProgress | null;
}

export function thumbnailStyle(
  person: EditorPerson,
  imageWidth: number,
  imageHeight: number,
  sizePx: number,
): Pick<CSSProperties, "backgroundPosition" | "backgroundSize"> {
  const [x1, y1, x2, y2] = person.face ?? person.box;
  const width = Math.max(1, x2 - x1);
  const height = Math.max(1, y2 - y1);
  const cropSize = person.face
    ? Math.max(width, height) * 1.6
    : Math.min(width, height * 0.25);
  const centerX = (x1 + x2) / 2;
  const centerY = person.face ? (y1 + y2) / 2 : y1 + cropSize / 2;
  const scale = sizePx / Math.max(1, cropSize);
  const offsetX = Math.max(0, Math.min(imageWidth - cropSize, centerX - cropSize / 2));
  const offsetY = Math.max(0, Math.min(imageHeight - cropSize, centerY - cropSize / 2));

  return {
    backgroundSize: `${imageWidth * scale}px ${imageHeight * scale}px`,
    backgroundPosition: `${-(offsetX * scale)}px ${-(offsetY * scale)}px`,
  };
}

function ChoiceSelector({
  person,
  personLabel,
  choice,
  onToggle,
}: {
  person: EditorPerson;
  personLabel: string;
  choice: Choice;
  onToggle: (id: number) => void;
}) {
  const { t } = useTranslation();
  return (
    <div
      role="radiogroup"
      aria-label={t("editor.people.choice", { person: personLabel })}
      className="flex shrink-0 gap-0.5 rounded-lg border border-border bg-bg p-0.5 text-[11px]"
    >
      <button
        type="button"
        role="radio"
        aria-checked={choice === "keep"}
        onClick={() => choice !== "keep" && onToggle(person.id)}
        className={`flex min-h-9 items-center gap-1 rounded-md px-2 ${
          choice === "keep" ? "bg-surface-2 font-medium text-ok" : "text-text-faint hover:text-text"
        }`}
      >
        <Check aria-hidden="true" className="h-3.5 w-3.5" />
        {t("editor.people.keep")}
      </button>
      <button
        type="button"
        role="radio"
        aria-checked={choice === "remove"}
        onClick={() => choice !== "remove" && onToggle(person.id)}
        className={`flex min-h-9 items-center gap-1 rounded-md px-2 ${
          choice === "remove" ? "bg-surface-2 font-medium text-danger" : "text-text-faint hover:text-text"
        }`}
      >
        <X aria-hidden="true" className="h-3.5 w-3.5" />
        {t("editor.people.remove")}
      </button>
    </div>
  );
}

function PersonRow({
  person,
  index,
  number,
  choices,
  imageUrl,
  imageWidth,
  imageHeight,
  hoveredId,
  onHover,
  onToggle,
  onFocus,
  tabIndex,
  rowRef,
  onKeyDown,
}: {
  person: EditorPerson;
  index: number;
  number: number;
  choices: Record<number, Choice>;
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  hoveredId: number | null;
  onHover: (id: number | null) => void;
  onToggle: (id: number) => void;
  onFocus: (index: number) => void;
  tabIndex: number;
  rowRef: (element: HTMLDivElement | null) => void;
  onKeyDown: (event: KeyboardEvent<HTMLDivElement>, index: number) => void;
}) {
  const { t } = useTranslation();
  const label = t("editor.people.person", { number });
  return (
    <div
      ref={rowRef}
      role="listitem"
      aria-label={label}
      tabIndex={tabIndex}
      onFocus={() => {
        onFocus(index);
        onHover(person.id);
      }}
      onBlur={() => onHover(null)}
      onMouseEnter={() => onHover(person.id)}
      onMouseLeave={() => onHover(null)}
      onKeyDown={(event) => onKeyDown(event, index)}
      className={`flex min-h-12 items-center gap-2 rounded-lg px-1.5 py-1 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent ${
        hoveredId === person.id ? "bg-surface-2" : ""
      }`}
    >
      <div
        aria-hidden="true"
        className="h-9 w-9 shrink-0 rounded-full border border-border bg-no-repeat"
        style={{
          backgroundImage: `url("${imageUrl}")`,
          ...thumbnailStyle(person, imageWidth, imageHeight, 36),
        }}
      />
      <span className="min-w-0 flex-1 truncate text-xs font-medium text-text" title={label}>{label}</span>
      <ChoiceSelector person={person} personLabel={label} choice={choices[person.id] ?? "remove"} onToggle={onToggle} />
    </div>
  );
}

function PersonGroup({
  role,
  group,
  title,
  indexOffset,
  numbers,
  choices,
  imageUrl,
  imageWidth,
  imageHeight,
  hoveredId,
  focusedIndex,
  rowRefs,
  onHover,
  onToggle,
  onGroup,
  onFocus,
  onKeyDown,
}: {
  role: EditorPerson["role"];
  group: EditorPerson[];
  title: string;
  indexOffset: number;
  numbers: Record<number, number>;
  choices: Record<number, Choice>;
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  hoveredId: number | null;
  focusedIndex: number;
  rowRefs: MutableRefObject<Array<HTMLDivElement | null>>;
  onHover: (id: number | null) => void;
  onToggle: (id: number) => void;
  onGroup: (role: EditorPerson["role"], choice: Choice) => void;
  onFocus: (index: number) => void;
  onKeyDown: (event: KeyboardEvent<HTMLDivElement>, index: number) => void;
}) {
  const { t } = useTranslation();
  if (!group.length) return null;
  const choice = group.every((person) => choices[person.id] === "keep") ? "remove" : "keep";
  const key = choice === "keep" ? "keepAll" : "removeAll";
  const label = t(`editor.people.${key}`);

  return (
    <section aria-label={title} className="mt-3">
      <div className="flex min-h-9 items-center gap-2">
        <h3 className="flex-1 text-[11px] font-semibold uppercase tracking-wide text-text-faint">
          {title} · {group.length}
        </h3>
        <button
          type="button"
          onClick={() => onGroup(role, choice)}
          className="min-h-9 rounded px-1.5 text-[11px] text-accent hover:bg-surface-2"
          aria-label={t(`editor.people.${key}Group`, {
            group: role === "subject" ? t("editor.people.frontLower") : t("editor.people.backLower"),
          })}
        >
          {label}
        </button>
      </div>
      <div role="list" className="flex flex-col gap-1">
        {group.map((person, groupIndex) => {
          const index = indexOffset + groupIndex;
          return (
            <PersonRow
              key={person.id}
              person={person}
              index={index}
              number={numbers[person.id]}
              choices={choices}
              imageUrl={imageUrl}
              imageWidth={imageWidth}
              imageHeight={imageHeight}
              hoveredId={hoveredId}
              onHover={onHover}
              onToggle={onToggle}
              onFocus={onFocus}
              tabIndex={focusedIndex === index ? 0 : -1}
              rowRef={(element) => { rowRefs.current[index] = element; }}
              onKeyDown={onKeyDown}
            />
          );
        })}
      </div>
    </section>
  );
}

export function PeoplePanel(props: PeoplePanelProps) {
  const {
    people, choices, hoveredId, onHover, onToggle, onGroup, protectKept,
    onProtectKeptChange, onAccept, onRetry, status, imageUrl, unavailableReason, error, progress,
  } = props;
  const { t } = useTranslation();
  const { front, back } = useMemo(() => groupPeople(people), [people]);
  const rows = useMemo(() => [...front, ...back], [back, front]);
  const numbers = useMemo(() => personNumbers(people), [people]);
  const rowRefs = useRef<Array<HTMLDivElement | null>>([]);
  const [focusedIndex, setFocusedIndex] = useState(0);
  const removeCount = countRemove(choices);

  useEffect(() => {
    setFocusedIndex((current) => rows.length === 0 ? 0 : Math.min(current, rows.length - 1));
  }, [rows]);

  function handleRowKeyDown(event: KeyboardEvent<HTMLDivElement>, index: number) {
    if (event.target !== event.currentTarget) return;
    if (event.key === " ") {
      event.preventDefault();
      onToggle(rows[index].id);
      return;
    }
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    event.preventDefault();
    const nextIndex = Math.max(0, Math.min(rows.length - 1, index + (event.key === "ArrowDown" ? 1 : -1)));
    setFocusedIndex(nextIndex);
    rowRefs.current[nextIndex]?.focus();
  }

  return (
    <div className="flex min-h-0 flex-col">
      <div className="mb-2">
        <h2 className="text-sm font-semibold text-text">{t("editor.people.title")}</h2>
        {status === "ready" && (
          <p className="mt-0.5 text-xs text-text-dim">
            {t("editor.people.summary", { count: people.length })}
          </p>
        )}
      </div>
      {status === "loading" && (
        <div aria-label={t("editor.people.loading")} className="flex min-h-0 flex-col gap-2 overflow-y-auto">
          {Array.from({ length: 5 }, (_, index) => (
            <div key={index} aria-hidden="true" className="h-12 animate-pulse rounded-lg bg-surface-2" />
          ))}
        </div>
      )}
      {status === "empty" && <p className="py-4 text-sm text-text-dim">{t("editor.people.empty")}</p>}
      {status === "unavailable" && (
        <p className="py-4 text-sm text-text-dim">{unavailableReason ?? t("editor.tool.peopleUnavailable")}</p>
      )}
      {status === "error" && (
        <div className="py-3 text-sm text-danger">
          <p>{error ?? t("editor.people.error")}</p>
          <button type="button" onClick={onRetry} className="mt-2 min-h-9 rounded border border-border px-3 text-text">
            {t("editor.people.retry")}
          </button>
        </div>
      )}
      {status === "ready" && (
        <>
          <div className="min-h-0 max-h-[calc(100dvh-26rem)] overflow-y-auto">
            <PersonGroup
              role="subject"
              group={front}
              title={t("editor.people.front")}
              indexOffset={0}
              numbers={numbers}
              choices={choices}
              imageUrl={imageUrl}
              imageWidth={props.imageWidth}
              imageHeight={props.imageHeight}
              hoveredId={hoveredId}
              focusedIndex={focusedIndex}
              rowRefs={rowRefs}
              onHover={onHover}
              onToggle={onToggle}
              onGroup={onGroup}
              onFocus={setFocusedIndex}
              onKeyDown={handleRowKeyDown}
            />
            <PersonGroup
              role="background"
              group={back}
              title={t("editor.people.back")}
              indexOffset={front.length}
              numbers={numbers}
              choices={choices}
              imageUrl={imageUrl}
              imageWidth={props.imageWidth}
              imageHeight={props.imageHeight}
              hoveredId={hoveredId}
              focusedIndex={focusedIndex}
              rowRefs={rowRefs}
              onHover={onHover}
              onToggle={onToggle}
              onGroup={onGroup}
              onFocus={setFocusedIndex}
              onKeyDown={handleRowKeyDown}
            />
          </div>
          <label className="mt-3 flex min-h-9 items-center gap-2 border-t border-border pt-3 text-xs text-text">
            <input
              type="checkbox"
              checked={protectKept}
              onChange={(event) => onProtectKeptChange(event.target.checked)}
              className="accent-accent"
            />
            {t("editor.people.protectKept")}
          </label>
          <button
            type="button"
            disabled={removeCount === 0 || Boolean(progress)}
            onClick={onAccept}
            className="mt-3 min-h-10 rounded-lg bg-accent px-3 text-sm font-semibold text-bg disabled:cursor-not-allowed disabled:opacity-40"
          >
            {progress
              ? t("editor.people.segmentingProgress", {
                current: Math.min(progress.done + 1, progress.total),
                total: progress.total,
              })
              : t("editor.people.accept", { count: removeCount })}
          </button>
        </>
      )}
    </div>
  );
}
