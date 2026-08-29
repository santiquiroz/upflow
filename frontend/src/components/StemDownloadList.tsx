import { Download } from "lucide-react";
import type { ReactNode } from "react";
import { useTranslation } from "../i18n/LocaleProvider";
import type { AudioStemDownload, TranscriptionDownload } from "../lib/apiTypes";

const TRANSCRIPTION_FORMAT_LABEL_KEYS: Record<string, string> = {
  midi: "audio.transcribe.format.midi",
  musicxml: "audio.transcribe.format.musicxml",
  tab: "audio.transcribe.format.tab",
};

// El backend no manda labelKey en las transcripciones -- solo un stemId
// crudo, el mismo id que ya tiene labelKey `audio.stem.<id>` en todo el resto
// de la app (ver `stems[]`).
function stemLabelKey(stemId: string): string {
  return `audio.stem.${stemId}`;
}

function groupTranscriptionsByStem(
  transcriptions: TranscriptionDownload[] | null | undefined,
): [string, TranscriptionDownload[]][] {
  if (!transcriptions || transcriptions.length === 0) {
    return [];
  }
  const order: string[] = [];
  const byStem = new Map<string, TranscriptionDownload[]>();
  for (const item of transcriptions) {
    if (!byStem.has(item.stemId)) {
      byStem.set(item.stemId, []);
      order.push(item.stemId);
    }
    byStem.get(item.stemId)?.push(item);
  }
  return order.map((stemId) => [stemId, byStem.get(stemId) ?? []]);
}

/** Un grupo de descargas (MIDI/MusicXML/tab) por stem transcripto. */
function TranscriptionDownloads({
  groups,
  linkClassName,
  iconClassName,
  containerClassName,
}: {
  groups: [string, TranscriptionDownload[]][];
  linkClassName: string;
  iconClassName: string;
  containerClassName: string;
}) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-2">
      {groups.map(([stemId, items]) => (
        <div key={stemId} className="flex flex-col gap-1">
          <span className="text-xs font-semibold uppercase tracking-wide text-text-dim">
            {t(stemLabelKey(stemId))}
          </span>
          <div className={containerClassName}>
            {items.map((item) => (
              <a
                key={`${item.stemId}-${item.format}`}
                href={item.url}
                download
                className={linkClassName}
              >
                <Download aria-hidden="true" className={iconClassName} strokeWidth={1.75} />
                {t(TRANSCRIPTION_FORMAT_LABEL_KEYS[item.format] ?? item.format)}
              </a>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function mainDownloads({
  stems,
  downloadUrl,
  vocalsUrl,
  linkClassName,
  iconClassName,
  containerClassName,
  t,
}: {
  stems: AudioStemDownload[] | null;
  downloadUrl: string;
  vocalsUrl: string | null;
  linkClassName: string;
  iconClassName: string;
  containerClassName: string;
  t: (key: string) => string;
}): ReactNode {
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

/**
 * Las descargas de un job de separación, compartidas entre la tarjeta y la
 * cola: la lista `stems[]` del backend (que ya trae los derivados minus_<stem>
 * con su labelKey), o el par instrumental + voz de los jobs anteriores a
 * `stems` -- más, si se pidió transcripción (F3a), un grupo de enlaces
 * MIDI/MusicXML/tab por stem transcripto. Sin nada de eso no dibuja nada: el
 * link genérico es del caller, que conoce su propio layout.
 */
export function StemDownloadList({
  stems,
  downloadUrl,
  vocalsUrl,
  transcriptions,
  linkClassName,
  iconClassName,
  containerClassName,
}: {
  stems: AudioStemDownload[] | null;
  downloadUrl: string;
  vocalsUrl: string | null;
  transcriptions?: TranscriptionDownload[] | null;
  linkClassName: string;
  iconClassName: string;
  containerClassName: string;
}) {
  const { t } = useTranslation();
  const groups = groupTranscriptionsByStem(transcriptions);
  const main = mainDownloads({
    stems,
    downloadUrl,
    vocalsUrl,
    linkClassName,
    iconClassName,
    containerClassName,
    t,
  });

  if (!main && groups.length === 0) {
    return null;
  }
  if (groups.length === 0) {
    // Sin transcripciones el layout queda idéntico al de antes de F3a.
    return main;
  }
  return (
    <div className="flex flex-col gap-3">
      {main}
      <TranscriptionDownloads
        groups={groups}
        linkClassName={linkClassName}
        iconClassName={iconClassName}
        containerClassName={containerClassName}
      />
    </div>
  );
}
