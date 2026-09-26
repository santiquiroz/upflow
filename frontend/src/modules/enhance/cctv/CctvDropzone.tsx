import { UploadCloud } from "lucide-react";
import type { ChangeEvent, DragEvent } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";

const INPUT_ID = "cctv-file-input";

// Sin `accept`: los exports de DVR se identifican por contenido y traen extensiones de todo tipo.
export function CctvDropzone({
  fileName,
  disabled,
  onFileSelected,
}: {
  fileName: string | null;
  disabled: boolean;
  onFileSelected: (file: File) => void;
}) {
  const { t } = useTranslation();

  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    const dropped = event.dataTransfer.files[0];
    if (dropped && !disabled) {
      onFileSelected(dropped);
    }
  }

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0];
    if (selected) {
      onFileSelected(selected);
    }
  }

  return (
    <label
      htmlFor={INPUT_ID}
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
      className="flex cursor-pointer flex-col items-center gap-2 rounded border border-dashed border-border bg-surface px-6 py-10 text-center transition-[border-color] duration-fast hover:border-accent"
    >
      <UploadCloud aria-hidden="true" className="h-6 w-6 text-text-faint" strokeWidth={1.5} />
      <span className="text-sm text-text">{fileName ?? t("cctv.dropzone")}</span>
      <span className="text-xs text-text-faint">{t("cctv.dropzone.formats")}</span>
      <input id={INPUT_ID} type="file" disabled={disabled} className="sr-only" onChange={handleChange} />
    </label>
  );
}
