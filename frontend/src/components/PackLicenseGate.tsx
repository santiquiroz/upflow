import { useQuery } from "@tanstack/react-query";
import { Download, ScrollText } from "lucide-react";
import { useId, useState } from "react";
import { useTranslation } from "../i18n/LocaleProvider";
import { fetchPackLicense } from "../services/licenses";

export const LICENSE_REQUIRED_KEY = "pack.license.required";
export const LICENSE_UNAVAILABLE_KEY = "pack.license.unavailable";

export function isLicenseGateKey(key: string | null | undefined): boolean {
  return key === LICENSE_REQUIRED_KEY || key === LICENSE_UNAVAILABLE_KEY;
}

function LicenseUnavailable() {
  const { t } = useTranslation();
  return (
    <p role="alert" className="text-sm text-danger">
      {t("pack.license.unavailableText")}
    </p>
  );
}

function LicenseAcceptance({
  licenseText,
  busy,
  onAccept,
}: {
  licenseText: string;
  busy: boolean;
  onAccept: () => void;
}) {
  const { t } = useTranslation();
  const [accepted, setAccepted] = useState(false);
  const textId = useId();

  return (
    <>
      <div
        id={textId}
        role="region"
        aria-label={t("pack.license.textLabel")}
        tabIndex={0}
        className="max-h-64 overflow-auto rounded border border-border bg-surface p-2 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        <pre className="whitespace-pre-wrap break-words font-mono-tabular text-[11px] leading-relaxed text-text-dim">
          {licenseText}
        </pre>
      </div>

      <label className="flex cursor-pointer items-start gap-2 text-sm text-text">
        <input
          type="checkbox"
          checked={accepted}
          onChange={(event) => setAccepted(event.target.checked)}
          aria-describedby={textId}
          className="mt-0.5 h-3.5 w-3.5 shrink-0 cursor-pointer accent-accent"
        />
        {t("pack.license.accept")}
      </label>

      <button
        type="button"
        onClick={onAccept}
        disabled={!accepted || busy}
        className="inline-flex w-fit items-center gap-2 rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg hover:bg-accent-hover disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        <Download aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
        {t("pack.license.acceptAndDownload")}
      </button>
    </>
  );
}

/**
 * La licencia de un pack restrictivo, completa, y la casilla que la acepta.
 *
 * El backend es quien decide que un pack tiene compuerta (403 con clave); esto
 * solo muestra el texto que el backend sirve para ese mismo pack.
 */
export function PackLicenseGate({
  pack,
  busy,
  onAccept,
}: {
  pack: string;
  busy: boolean;
  onAccept: () => void;
}) {
  const { t } = useTranslation();
  const license = useQuery({
    queryKey: ["pack-license", pack],
    queryFn: () => fetchPackLicense(pack),
    staleTime: Infinity,
    retry: false,
  });
  const licenseText = license.data?.licenseText ?? null;

  return (
    <section
      aria-label={t("pack.license.title")}
      className="flex flex-col gap-2 border-l-2 border-warn pl-3"
    >
      <h3 className="inline-flex items-center gap-2 text-sm font-semibold text-text">
        <ScrollText aria-hidden="true" className="h-4 w-4 text-warn" strokeWidth={1.75} />
        {t("pack.license.title")}
      </h3>
      <p className="text-sm text-text-dim">{t("pack.license.intro")}</p>

      {license.isPending ? (
        <p role="status" className="text-xs text-text-faint">
          {t("pack.license.loading")}
        </p>
      ) : licenseText === null ? (
        <LicenseUnavailable />
      ) : (
        <LicenseAcceptance licenseText={licenseText} busy={busy} onAccept={onAccept} />
      )}
    </section>
  );
}
