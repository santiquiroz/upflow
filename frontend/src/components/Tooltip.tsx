import { Info } from "lucide-react";
import { useId, useState, type KeyboardEvent } from "react";

interface TooltipProps {
  label: string;
  content: string;
}

function handleEscapeKey(event: KeyboardEvent<HTMLButtonElement>, hide: () => void): void {
  if (event.key === "Escape") {
    event.stopPropagation();
    hide();
  }
}

export function Tooltip({ label, content }: TooltipProps) {
  const [visible, setVisible] = useState(false);
  const tooltipId = useId();

  function show(): void {
    setVisible(true);
  }

  function hide(): void {
    setVisible(false);
  }

  return (
    <span className="relative inline-flex shrink-0">
      <button
        type="button"
        aria-label={label}
        aria-describedby={visible ? tooltipId : undefined}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
        onKeyDown={(event) => handleEscapeKey(event, hide)}
        className="flex h-5 w-5 items-center justify-center rounded-sm text-text-faint transition-colors duration-fast hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        <Info aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
      </button>
      {visible && (
        <span
          role="tooltip"
          id={tooltipId}
          className="absolute right-0 top-full z-20 mt-1 w-56 rounded border border-border bg-surface-2 p-2 text-xs text-text shadow-lg"
        >
          {content}
        </span>
      )}
    </span>
  );
}
