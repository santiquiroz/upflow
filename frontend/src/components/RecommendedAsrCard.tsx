import { useTranslation } from "../i18n/LocaleProvider";
import {
  useAsrModelInstall,
  useInstalledAsrModels,
} from "../hooks/useTranscribeJob";

// El mismo modelo que recomienda el instalador (model_packs.RECOMMENDED_ASR_REPO):
// el mejor de los instalables para letra cantada. Cambiarlo es cambiarlo en ambos.
export const RECOMMENDED_ASR_REPO_ID = "onnx-community/whisper-small_timestamped";

/**
 * Sin ningun modelo de voz instalado, Transcribir y Karaoke son pantallas
 * vacias con un buscador: esta tarjeta deja el default recomendado a UN click,
 * sin saber que buscar.
 */
export function RecommendedAsrCard() {
  const { t } = useTranslation();
  const modelsQuery = useInstalledAsrModels();
  const install = useAsrModelInstall();

  const models = modelsQuery.data ?? [];
  if (modelsQuery.isLoading || models.length > 0) {
    return null;
  }

  const busy = install.phase === "starting" || install.phase === "downloading";

  return (
    <div className="flex flex-col gap-2 rounded border border-accent/40 bg-surface-2 p-4">
      <span className="font-heading text-sm font-semibold text-text">
        {t("asrDefault.title")}
      </span>
      <p className="text-xs text-text-dim">{t("asrDefault.description")}</p>
      {install.phase === "installed" ? (
        <p role="status" className="text-sm text-ok">
          {t("asrDefault.done")}
        </p>
      ) : (
        <button
          type="button"
          onClick={() => install.install(RECOMMENDED_ASR_REPO_ID)}
          disabled={busy}
          className="self-start rounded bg-accent px-3 py-1.5 text-sm font-semibold text-bg disabled:opacity-50"
        >
          {busy
            ? t("asrDefault.installing", {
                pct: (install.progressPct ?? 0).toFixed(0),
              })
            : t("asrDefault.button")}
        </button>
      )}
      {install.errorMessage && (
        <p role="alert" className="text-xs text-danger">
          {install.errorMessage}
        </p>
      )}
    </div>
  );
}
