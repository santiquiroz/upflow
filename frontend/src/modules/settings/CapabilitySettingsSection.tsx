import { useState, type FormEvent } from "react";
import { useEditableSettings } from "../../hooks/useEditableSettings";
import { useTranslation } from "../../i18n/LocaleProvider";
import type { EditableSettingStatus } from "../../lib/apiTypes";

const CAD_SERVER_KEY = "cad_llm_base_url";

// Flags que YA tienen su propia tarjeta en otra parte de esta pantalla. Se
// listan para no mostrarlos dos veces con dos textos distintos.
const FLAGS_OWNED_ELSEWHERE = new Set(["rebar_confirmed"]);

// Un flag es exactamente lo que el backend devuelve con valor: solo las claves
// no sensibles traen `value`, asi que el interruptor no se puede dibujar para
// un secreto ni por accidente.
function isFlag(setting: EditableSettingStatus): boolean {
  return setting.value !== null && !FLAGS_OWNED_ELSEWHERE.has(setting.key);
}

function FlagToggle({
  setting,
  onChange,
  isPending,
}: {
  setting: EditableSettingStatus;
  onChange: (key: string, checked: boolean) => void;
  isPending: boolean;
}) {
  const { t } = useTranslation();

  return (
    <div className="flex flex-col gap-1">
      <label className="flex items-center gap-2 text-sm text-text">
        <input
          type="checkbox"
          checked={setting.value === "true"}
          disabled={isPending}
          onChange={(event) => onChange(setting.key, event.target.checked)}
          className="h-3.5 w-3.5 accent-accent"
        />
        {t(`settings.flag.${setting.key}`)}
      </label>
      {setting.requiresRestart && (
        <p className="pl-6 text-xs text-warn">{t("settings.flag.restartNeeded")}</p>
      )}
    </div>
  );
}

function CadServerField({ setting }: { setting: EditableSettingStatus | undefined }) {
  const { t } = useTranslation();
  const [url, setUrl] = useState("");
  const { patchMutation } = useEditableSettings();

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (url.trim().length === 0) {
      return;
    }
    patchMutation.mutate({ key: CAD_SERVER_KEY, value: url.trim() }, { onSuccess: () => setUrl("") });
  }

  if (!setting) {
    return null;
  }

  return (
    <div className="flex flex-col gap-2 border-t border-border pt-3">
      <div className="flex items-center justify-between gap-4">
        <label htmlFor="cad-llm-base-url" className="text-sm text-text">
          {t("settings.cadServer.label")}
        </label>
        <span className={`text-xs ${setting.configured ? "text-ok" : "text-text-faint"}`}>
          {setting.configured ? t("settings.cadServer.configured") : t("settings.token.notConfigured")}
        </span>
      </div>
      <p className="text-xs text-text-faint">{t("settings.cadServer.explanation")}</p>
      <form onSubmit={handleSubmit} className="flex items-center gap-2">
        <input
          id="cad-llm-base-url"
          type="url"
          value={url}
          onChange={(event) => {
            setUrl(event.target.value);
            patchMutation.reset();
          }}
          placeholder={t("settings.cadServer.placeholder")}
          className="w-full rounded border border-border bg-surface px-3 py-2 text-sm text-text placeholder:text-text-faint focus:border-accent focus:outline-none"
        />
        <button
          type="submit"
          disabled={url.trim().length === 0 || patchMutation.isPending}
          className="inline-flex shrink-0 items-center gap-2 rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg transition-[background-color] duration-fast hover:bg-accent-hover active:bg-accent-press disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          {patchMutation.isPending ? t("settings.videoLimit.saving") : t("settings.videoLimit.save")}
        </button>
      </form>
      {setting.requiresRestart && (
        <p className="text-xs text-warn">{t("settings.cadServer.restartNeeded")}</p>
      )}
      {patchMutation.isError && (
        <p role="alert" className="text-xs text-danger">
          {patchMutation.error instanceof Error
            ? patchMutation.error.message
            : t("settings.flag.saveError")}
        </p>
      )}
    </div>
  );
}

export function CapabilitySettingsSection() {
  const { t } = useTranslation();
  const { settingsQuery, patchMutation } = useEditableSettings();
  const settings = settingsQuery.data?.settings ?? [];
  const flags = settings.filter(isFlag);

  function handleToggle(key: string, checked: boolean): void {
    patchMutation.mutate({ key, value: String(checked) });
  }

  return (
    <div className="flex flex-col gap-3 rounded border border-border bg-surface p-4">
      <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
        {t("settings.capabilityFlags.title")}
      </h2>
      <p className="text-xs text-text-dim">{t("settings.capabilityFlags.explanation")}</p>
      {settingsQuery.isLoading && (
        <p className="text-xs text-text-faint">{t("settings.capabilityFlags.loading")}</p>
      )}
      {settingsQuery.isError && (
        <p role="alert" className="text-xs text-danger">
          {t("settings.token.loadError")}
        </p>
      )}
      <div className="flex flex-col gap-2">
        {flags.map((setting) => (
          <FlagToggle
            key={setting.key}
            setting={setting}
            onChange={handleToggle}
            isPending={patchMutation.isPending}
          />
        ))}
      </div>
      {patchMutation.isError && (
        <p role="alert" className="text-xs text-danger">
          {patchMutation.error instanceof Error
            ? patchMutation.error.message
            : t("settings.flag.saveError")}
        </p>
      )}
      <CadServerField setting={settings.find((item) => item.key === CAD_SERVER_KEY)} />
    </div>
  );
}
