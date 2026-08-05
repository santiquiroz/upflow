import { useState, type FormEvent } from "react";
import { useEditableSettings } from "../../hooks/useEditableSettings";
import { useTranslation } from "../../i18n/LocaleProvider";

const LIMIT_KEY = "max_video_upload_mb";

// El limite protege el DISCO, no la red: el pipeline extrae cada frame como
// imagen, asi que un video de 2 GB puede volverse cientos de GB de PNG. Pero es
// un numero de politica y no un techo tecnico, y hay maquinas con disco de
// sobra donde queda corto. La red de seguridad real vive aguas abajo:
// `video_upscaler._ensure_disk_room_for_png_path` calcula el pico real y aborta
// ANTES de escribir un solo frame.
export function VideoUploadLimitSection({ currentMb }: { currentMb: number }) {
  const { t } = useTranslation();
  const [value, setValue] = useState("");
  const { patchMutation } = useEditableSettings();

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (value.trim().length === 0) {
      return;
    }
    patchMutation.mutate({ key: LIMIT_KEY, value: value.trim() }, { onSuccess: () => setValue("") });
  }

  return (
    <div className="flex flex-col gap-3 rounded border border-border bg-surface p-4">
      <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
        {t("settings.videoLimit.title")}
      </h2>
      <p className="text-xs text-text-dim">{t("settings.videoLimit.explanation")}</p>
      <div className="flex items-center justify-between gap-4">
        <label htmlFor="video-upload-limit" className="text-sm text-text">
          {t("settings.videoLimit.label")}
        </label>
        <span className="font-mono-tabular text-xs text-text-dim">
          {t("settings.videoLimit.current", { mb: currentMb })}
        </span>
      </div>
      <form onSubmit={handleSubmit} className="flex items-center gap-2">
        <input
          id="video-upload-limit"
          type="number"
          min={1}
          value={value}
          onChange={(event) => {
            setValue(event.target.value);
            patchMutation.reset();
          }}
          placeholder={String(currentMb)}
          className="w-full rounded border border-border bg-surface px-3 py-2 text-sm text-text placeholder:text-text-faint focus:border-accent focus:outline-none disabled:opacity-60"
        />
        <button
          type="submit"
          disabled={value.trim().length === 0 || patchMutation.isPending}
          className="inline-flex shrink-0 items-center gap-2 rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg transition-[background-color] duration-fast hover:bg-accent-hover active:bg-accent-press disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          {patchMutation.isPending ? t("settings.videoLimit.saving") : t("settings.videoLimit.save")}
        </button>
      </form>
      {patchMutation.isSuccess && (
        <p role="status" className="text-xs text-ok">
          {t("settings.videoLimit.saved")}
        </p>
      )}
      {patchMutation.isError && (
        <p role="alert" className="text-xs text-danger">
          {patchMutation.error instanceof Error
            ? patchMutation.error.message
            : t("settings.videoLimit.saveError")}
        </p>
      )}
    </div>
  );
}
