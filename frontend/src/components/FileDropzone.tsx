import { UploadCloud } from "lucide-react";
import type { ChangeEvent, DragEvent } from "react";
import { useTranslation } from "../i18n/LocaleProvider";

interface FileDropzoneProps {
  inputId: string;
  accept: string;
  multiple: boolean;
  files: File[];
  emptyLabel: string;
  formatsHint: string;
  onFilesSelected: (files: File[]) => void;
}

// Un drop no respeta el atributo `multiple` del input: sin este recorte, soltar
// tres fotos en un panel de una sola foto las mandaria todas.
function limitSelection(files: File[], multiple: boolean): File[] {
  return multiple ? files : files.slice(0, 1);
}

function useSelectionLabel(files: File[], emptyLabel: string): string {
  const { t } = useTranslation();
  if (files.length === 0) {
    return emptyLabel;
  }
  if (files.length === 1) {
    return files[0].name;
  }
  return t("enhance.batch.selected", { count: files.length });
}

export function FileDropzone({
  inputId,
  accept,
  multiple,
  files,
  emptyLabel,
  formatsHint,
  onFilesSelected,
}: FileDropzoneProps) {
  const label = useSelectionLabel(files, emptyLabel);

  function report(selected: File[]) {
    if (selected.length > 0) {
      onFilesSelected(limitSelection(selected, multiple));
    }
  }

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    report(Array.from(event.dataTransfer.files));
  }

  function handleInputChange(event: ChangeEvent<HTMLInputElement>) {
    report(Array.from(event.target.files ?? []));
  }

  return (
    <label
      htmlFor={inputId}
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
      className="flex cursor-pointer flex-col items-center gap-2 rounded border border-dashed border-border bg-surface px-6 py-10 text-center transition-[border-color] duration-fast hover:border-accent"
    >
      <UploadCloud aria-hidden="true" className="h-6 w-6 text-text-faint" strokeWidth={1.5} />
      <span className="text-sm text-text">{label}</span>
      <span className="text-xs text-text-faint">{formatsHint}</span>
      <input
        id={inputId}
        type="file"
        accept={accept}
        multiple={multiple}
        className="sr-only"
        onChange={handleInputChange}
      />
    </label>
  );
}
