import { useQuery } from "@tanstack/react-query";
import { Download, Wand2 } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "../../i18n/LocaleProvider";
import { PackDownload } from "../../components/PackDownload";
import { apiGet } from "../../lib/api";

interface ConversionCapabilities {
  available: boolean;
  reason: string | null;
  maxSeconds: number;
  missingPack: string | null;
}

function FilePicker({
  id,
  label,
  file,
  onPick,
}: {
  id: string;
  label: string;
  file: File | null;
  onPick: (file: File) => void;
}) {
  return (
    <label htmlFor={id} className="flex cursor-pointer flex-col gap-1">
      <span className="text-xs font-medium text-text-dim">{label}</span>
      <span className="rounded border border-dashed border-border bg-surface px-3 py-3 text-sm text-text-dim transition-[border-color] duration-fast hover:border-accent">
        {file ? file.name : "—"}
      </span>
      <input
        id={id}
        type="file"
        accept="audio/*,video/*"
        className="sr-only"
        onChange={(event) => {
          const picked = event.target.files?.[0];
          if (picked) {
            onPick(picked);
          }
        }}
      />
    </label>
  );
}

export function VoiceConvertPanel() {
  const { t } = useTranslation();
  const [source, setSource] = useState<File | null>(null);
  const [reference, setReference] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const capabilities = useQuery({
    queryKey: ["voiceConversionCapabilities"],
    queryFn: () => apiGet<ConversionCapabilities>("/voice/conversion/capabilities"),
  });

  async function handleConvert() {
    if (!source || !reference) {
      return;
    }
    setIsBusy(true);
    setError(null);
    try {
      const form = new FormData();
      form.append("source", source);
      form.append("reference", reference);
      const response = await fetch("/api/v1/voice/conversion", { method: "POST", body: form });
      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(body?.detail ?? t("voice.convert.failed"));
      }
      const nextUrl = URL.createObjectURL(
        new Blob([await response.arrayBuffer()], { type: "audio/wav" }),
      );
      // Se revoca el anterior: si no, cada intento deja un blob en memoria.
      setAudioUrl((previous) => {
        if (previous) {
          URL.revokeObjectURL(previous);
        }
        return nextUrl;
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : t("voice.convert.failed"));
    } finally {
      setIsBusy(false);
    }
  }

  if (capabilities.data && !capabilities.data.available) {
    return (
      capabilities.data.missingPack ? (
        <PackDownload
          pack={capabilities.data.missingPack}
          reason={capabilities.data.reason}
          onDone={() => void capabilities.refetch()}
        />
      ) : (
        <div role="alert" className="rounded border border-border bg-surface p-4">
          <p className="text-sm text-text-dim">{capabilities.data.reason}</p>
        </div>
      )
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <p className="text-sm text-text-dim">{t("voice.convert.subtitle")}</p>
      <div className="grid grid-cols-2 gap-4 max-[700px]:grid-cols-1">
        <FilePicker
          id="vc-source"
          label={t("voice.convert.source")}
          file={source}
          onPick={setSource}
        />
        <FilePicker
          id="vc-reference"
          label={t("voice.convert.reference")}
          file={reference}
          onPick={setReference}
        />
      </div>
      {capabilities.data && (
        <span className="text-xs text-text-faint">
          {t("voice.convert.limit", { seconds: capabilities.data.maxSeconds })}
        </span>
      )}
      <button
        type="button"
        onClick={() => void handleConvert()}
        disabled={!source || !reference || isBusy}
        className="inline-flex w-fit items-center gap-2 rounded bg-accent px-4 py-2 text-sm font-medium text-bg transition-[background-color,opacity] duration-fast hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-40"
      >
        <Wand2 aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
        {isBusy ? t("voice.convert.working") : t("voice.convert.run")}
      </button>
      {error && (
        <p role="alert" className="text-xs text-danger">
          {error}
        </p>
      )}
      {audioUrl && (
        <div className="flex flex-col gap-2 rounded border border-border bg-surface-2 p-4">
          {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
          <audio controls src={audioUrl} className="w-full" />
          <a
            href={audioUrl}
            download="voz-convertida.wav"
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
