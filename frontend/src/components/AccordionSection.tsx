import { ChevronDown } from "lucide-react";
import { useId, useState, type KeyboardEvent, type ReactNode } from "react";
import { Tooltip } from "./Tooltip";

interface AccordionSectionProps {
  title: string;
  summary: ReactNode;
  tooltip: string;
  defaultOpen?: boolean;
  children: ReactNode;
}

const TOGGLE_KEYS = ["Enter", " "];

function chevronClassName(open: boolean): string {
  const base = "h-4 w-4 shrink-0 text-text-faint transition-transform duration-fast motion-reduce:transition-none";
  return open ? base : `${base} -rotate-90`;
}

function isToggleKey(key: string): boolean {
  return TOGGLE_KEYS.includes(key);
}

// The native `hidden` attribute only wins the browser's display cascade when
// no author stylesheet also sets `display` on the same element -- a static
// `flex` class here would keep the body visually visible despite `hidden`
// (Tailwind's author-origin `display:flex` beats the UA `[hidden]{display:none}`
// rule regardless of selector specificity). So `flex flex-col gap-4` is only
// applied while open.
function bodyClassName(open: boolean): string {
  const base = "border-t border-border p-3";
  return open ? `${base} flex flex-col gap-4` : base;
}

export function AccordionSection({ title, summary, tooltip, defaultOpen = false, children }: AccordionSectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  const bodyId = useId();

  function toggle(): void {
    setOpen((current) => !current);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLButtonElement>): void {
    if (isToggleKey(event.key)) {
      event.preventDefault();
      toggle();
    }
  }

  return (
    <div className="rounded border border-border bg-surface">
      <div className="flex items-center gap-2 px-3 py-2">
        <button
          type="button"
          aria-expanded={open}
          aria-controls={bodyId}
          onClick={toggle}
          onKeyDown={handleKeyDown}
          className="flex flex-1 items-center justify-between gap-3 text-left"
        >
          <span className="flex shrink-0 items-center gap-2">
            <ChevronDown aria-hidden="true" className={chevronClassName(open)} strokeWidth={1.75} />
            <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">{title}</span>
          </span>
          <span className="truncate pl-3 text-xs text-text-faint">{summary}</span>
        </button>
        <Tooltip label={`About ${title}`} content={tooltip} />
      </div>
      <div id={bodyId} hidden={!open} className={bodyClassName(open)}>
        {children}
      </div>
    </div>
  );
}
