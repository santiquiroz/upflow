import { useEffect, useRef, type KeyboardEvent, type ReactNode } from "react";

interface ModalProps {
  titleId: string;
  onClose: () => void;
  children: ReactNode;
  /**
   * Ancho maximo. El default sirve para un dialogo de confirmacion; un detalle
   * con dos columnas de datos necesita mas. Es una clase y no un numero para no
   * salirse de la escala de Tailwind que usa el resto de la app.
   */
  widthClassName?: string;
}

const FOCUSABLE_SELECTOR =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

function getFocusableElements(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR));
}

function focusInitialElement(dialog: HTMLElement): void {
  const focusables = getFocusableElements(dialog);
  (focusables[0] ?? dialog).focus();
}

function trapTabWithin(event: KeyboardEvent<HTMLDivElement>, dialog: HTMLElement): void {
  const focusables = getFocusableElements(dialog);
  if (focusables.length === 0) {
    event.preventDefault();
    return;
  }
  const first = focusables[0];
  const last = focusables[focusables.length - 1];
  const active = document.activeElement;
  if (event.shiftKey && (active === first || !dialog.contains(active))) {
    event.preventDefault();
    last.focus();
    return;
  }
  if (!event.shiftKey && (active === last || !dialog.contains(active))) {
    event.preventDefault();
    first.focus();
  }
}

export function Modal({ titleId, onClose, children, widthClassName = "max-w-sm" }: ModalProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    previouslyFocusedRef.current = document.activeElement as HTMLElement | null;
    if (dialogRef.current) {
      focusInitialElement(dialogRef.current);
    }
    return () => previouslyFocusedRef.current?.focus();
  }, []);

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>): void {
    if (event.key === "Escape") {
      event.stopPropagation();
      onClose();
      return;
    }
    if (event.key === "Tab" && dialogRef.current) {
      trapTabWithin(event, dialogRef.current);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4" onKeyDown={handleKeyDown}>
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        className={`flex max-h-[90vh] w-full flex-col gap-4 rounded border border-border bg-surface p-5 ${widthClassName}`}
      >
        {children}
      </div>
    </div>
  );
}
