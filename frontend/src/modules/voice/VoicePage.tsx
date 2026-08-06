import { useQuery } from "@tanstack/react-query";
import { Mic, Download } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "../../i18n/LocaleProvider";
import { apiGet } from "../../lib/api";
import { PackDownload } from "../../components/PackDownload";
import { VoiceConvertPanel } from "./VoiceConvertPanel";

interface TtsCapabilities {
  available: boolean;
  voices: string[];
  reason: string | null;
  /** Que paquete falta. Es lo que le permite a la pantalla ofrecer el boton. */
  missingPack: string | null;
}

const MAX_CHARS = 2000;

type VoiceMode = "tts" | "convert";

export function VoicePage() {
  const { t } = useTranslation();
  const [mode, setMode] = useState<VoiceMode>("tts");
  const [text, setText] = useState("");
  const [voice, setVoice] = useState<string | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const capabilities = useQuery({
    queryKey: ["ttsCapabilities"],
    queryFn: () => apiGet<TtsCapabilities>("/tts/capabilities"),
  });

  const voices = capabilities.data?.voices ?? [];
  const selectedVoice = voice ?? voices[0] ?? null;

  async function handleSynthesize() {
    if (!text.trim() || !selectedVoice) {
      return;
    }
    setIsBusy(true);
    setError(null);
    try {
      const response = await fetch("/api/v1/tts/synthesize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, voice: selectedVoice, language: null }),
      });
      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(body?.detail ?? t("voice.tts.failed"));
      }
      const blob = new Blob([await response.arrayBuffer()], { type: "audio/wav" });
      const nextUrl = URL.createObjectURL(blob);
      // Se revoca el anterior: sin esto cada sintesis deja un blob colgado en
      // memoria hasta recargar la pagina.
      setAudioUrl((previous) => {
        if (previous) {
          URL.revokeObjectURL(previous);
        }
        return nextUrl;
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : t("voice.tts.failed"));
    } finally {
      setIsBusy(false);
    }
  }

  if (capabilities.data && !capabilities.data.available) {
    return (
      <div className="flex flex-col gap-4">
        <Header />
        {capabilities.data.missingPack ? (
          <PackDownload
            pack={capabilities.data.missingPack}
            reason={capabilities.data.reason}
            onDone={() => void capabilities.refetch()}
          />
        ) : (
          <div role="alert" className="rounded border border-border bg-surface p-4">
            <p className="text-sm text-text-dim">{capabilities.data.reason}</p>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <Header />
      <div role="group" aria-label={t("voice.tts.title")} className="flex gap-2">
        {(["tts", "convert"] as const).map((option) => (
          <button
            key={option}
            type="button"
            aria-pressed={mode === option}
            onClick={() => setMode(option)}
            className={`rounded-sm border px-3 py-1.5 text-sm transition-[background-color,border-color,color] duration-fast ${
              mode === option
                ? "border-accent bg-accent text-bg"
                : "border-border bg-surface text-text-dim hover:border-text-faint hover:text-text"
            }`}
          >
            {t(option === "tts" ? "voice.tab.tts" : "voice.tab.convert")}
          </button>
        ))}
      </div>

      {mode === "convert" && <VoiceConvertPanel />}

      {mode === "tts" && (
      <div className="flex flex-col gap-2">
        <label htmlFor="tts-text" className="text-xs font-medium text-text-dim">
          {t("voice.tts.text")}
        </label>
        <textarea
          id="tts-text"
          value={text}
          maxLength={MAX_CHARS}
          onChange={(event) => setText(event.target.value)}
          rows={5}
          className="rounded border border-border bg-surface p-2 text-sm text-text"
        />
        <span className="font-mono-tabular text-xs text-text-faint">
          {text.length} / {MAX_CHARS}
        </span>
      </div>
      )}

      {mode === "tts" && voices.length > 0 && (
        <div className="flex flex-col gap-2">
          <label htmlFor="tts-voice" className="text-xs font-medium text-text-dim">
            {t("voice.tts.voice")}
          </label>
          <select
            id="tts-voice"
            value={selectedVoice ?? ""}
            onChange={(event) => setVoice(event.target.value)}
            className="w-fit rounded border border-border bg-surface p-2 text-sm text-text"
          >
            {voices.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </div>
      )}

      {mode === "tts" && (
      <button
        type="button"
        onClick={() => void handleSynthesize()}
        disabled={text.trim().length === 0 || isBusy || !selectedVoice}
        className="inline-flex w-fit items-center gap-2 rounded bg-accent px-4 py-2 text-sm font-medium text-bg transition-[background-color,opacity] duration-fast hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
      >
        <Mic aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
        {isBusy ? t("voice.tts.working") : t("voice.tts.speak")}
      </button>
      )}

      {mode === "tts" && error && (
        <p role="alert" className="text-xs text-danger">
          {error}
        </p>
      )}

      {mode === "tts" && audioUrl && (
        <div className="flex flex-col gap-2 rounded border border-border bg-surface-2 p-4">
          {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
          <audio controls src={audioUrl} className="w-full" />
          <a
            href={audioUrl}
            download="voz.wav"
            className="inline-flex w-fit items-center gap-1.5 text-xs text-accent hover:text-accent-hover"
          >
            <Download aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
            {t("voice.tts.download")}
          </a>
        </div>
      )}
    </div>
  );
}

function Header() {
  const { t } = useTranslation();
  return (
    <div className="flex items-start gap-3">
      <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded border border-border bg-surface-2 text-accent">
        <Mic aria-hidden="true" className="h-5 w-5" strokeWidth={1.75} />
      </span>
      <div>
        <h1 className="font-heading text-2xl font-semibold text-text">{t("voice.tts.title")}</h1>
        <p className="mt-1 text-sm text-text-dim">{t("voice.tts.subtitle")}</p>
      </div>
    </div>
  );
}
