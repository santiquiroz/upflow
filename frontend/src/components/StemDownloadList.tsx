import { Download } from "lucide-react";
import { useTranslation } from "../i18n/LocaleProvider";
import type { AudioStemDownload } from "../lib/apiTypes";

/**
 * Las descargas de un job de separación, compartidas entre la tarjeta y la
 * cola: la lista `stems[]` del backend (que ya trae los derivados minus_<stem>
 * con su labelKey), o el par instrumental + voz de los jobs anteriores a
 * `stems`. Sin ninguna de las dos no dibuja nada: el link genérico es del
 * caller, que conoce su propio layout.
 */
export function StemDownloadList({
  stems,
  downloadUrl,
  vocalsUrl,
  linkClassName,
  iconClassName,
  containerClassName,
}: {
  stems: AudioStemDownload[] | null;
  downloadUrl: string;
  vocalsUrl: string | null;
  linkClassName: string;
  iconClassName: string;
  containerClassName: string;
}) {
  const { t } = useTranslation();
  if (stems) {
    // El backend ordena la lista: primero el stem que el usuario quiere.
    return (
      <div className={containerClassName}>
        {stems.map((stem) => (
          <a key={stem.id} href={stem.url} download className={linkClassName}>
            <Download aria-hidden="true" className={iconClassName} strokeWidth={1.75} />
            {t(stem.labelKey)}
          </a>
        ))}
      </div>
    );
  }
  if (vocalsUrl) {
    // Fallback karaoke pre-stems: instrumental + voz con labels fijos.
    return (
      <div className={containerClassName}>
        <a href={downloadUrl} download className={linkClassName}>
          <Download aria-hidden="true" className={iconClassName} strokeWidth={1.75} />
          {t("audio.karaoke.download.instrumental")}
        </a>
        <a href={vocalsUrl} download className={linkClassName}>
          <Download aria-hidden="true" className={iconClassName} strokeWidth={1.75} />
          {t("audio.karaoke.download.vocals")}
        </a>
      </div>
    );
  }
  return null;
}
