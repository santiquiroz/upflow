import { ImageUp } from "lucide-react";
import { useTranslation } from "../../i18n/LocaleProvider";

export function EditorDropzone({ onFile }: { onFile: (file: File) => void }) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col items-center gap-3 rounded border border-dashed border-border bg-surface p-10 text-center">
      <ImageUp aria-hidden="true" className="h-8 w-8 text-text-dim" strokeWidth={1.5} />
      <p className="text-sm text-text-dim">{t("editor.dropzone.hint")}</p>
      <label
        htmlFor="editor-base-image-input"
        className="cursor-pointer rounded bg-accent px-4 py-2 text-sm font-medium text-surface"
      >
        {t("editor.dropzone.label")}
      </label>
      <input
        id="editor-base-image-input"
        type="file"
        accept="image/*"
        className="sr-only"
        aria-label={t("editor.dropzone.inputLabel")}
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) {
            onFile(file);
          }
          event.target.value = "";
        }}
      />
    </div>
  );
}
