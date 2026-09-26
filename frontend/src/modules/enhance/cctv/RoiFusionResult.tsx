import { TriangleAlert } from "lucide-react";
import type { ReactNode } from "react";
import { useTranslation } from "../../../i18n/LocaleProvider";
import type { CctvArtifactLink, CctvRoiSummary } from "../../../lib/apiTypes";
import { artifactUrl } from "./cctvResultFiles";
import { roiResultNotices } from "./cctvRoi";

interface RoiFusionResultProps {
  roi: CctvRoiSummary;
  artifacts: readonly CctvArtifactLink[];
}

function RoiFigure({ url, caption, children }: { url: string | null; caption: string; children?: ReactNode }) {
  if (!url) {
    return null;
  }
  return (
    <figure className="flex flex-col gap-1">
      {/* Sin suavizado: el navegador no debe "mejorar" la ampliacion al mostrarla. */}
      <img src={url} alt={caption} className="max-h-64 w-full rounded-sm border border-border bg-black object-contain [image-rendering:pixelated]" />
      <figcaption className="text-xs text-text-dim">{caption}</figcaption>
      {children}
    </figure>
  );
}

function FramesUsed({ roi }: { roi: CctvRoiSummary }) {
  const { t } = useTranslation();
  return (
    <p className="flex flex-wrap items-center gap-2 text-sm text-text">
      <span className="font-mono-tabular">
        {t("cctv.roi.framesUsed", { used: roi.framesUsed, total: roi.framesTotal, effective: roi.effectiveSamples })}
      </span>
      {roi.nearCopies && (
        <span className="rounded-sm border border-warn px-1.5 py-0.5 text-xs text-warn">{t("cctv.roi.littleToGain")}</span>
      )}
    </p>
  );
}

function Notices({ roi }: { roi: CctvRoiSummary }) {
  const { t } = useTranslation();
  const notices = roiResultNotices(roi);
  if (notices.length === 0) {
    return null;
  }
  return (
    <ul aria-label={t("cctv.result.warnings")} className="flex flex-col gap-1">
      {notices.map((notice) => (
        <li key={notice.key} className="flex items-start gap-1.5 text-xs text-warn">
          <TriangleAlert aria-hidden="true" className="mt-0.5 h-3.5 w-3.5 shrink-0" strokeWidth={1.75} />
          {t(notice.key, notice.params)}
        </li>
      ))}
    </ul>
  );
}

export function RoiFusionResult({ roi, artifacts }: RoiFusionResultProps) {
  const { t } = useTranslation();
  return (
    <section aria-label={t("cctv.roi.legend")} className="flex flex-col gap-3">
      <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">{t("cctv.roi.legend")}</span>
      <FramesUsed roi={roi} />
      <div className="grid grid-cols-2 gap-3 max-[640px]:grid-cols-1">
        <RoiFigure url={artifactUrl(artifacts, "roi:fused")} caption={t("cctv.roi.result.fused", { scale: roi.scale })} />
        <RoiFigure
          url={artifactUrl(artifacts, "roi:reference")}
          caption={t("cctv.roi.result.reference", { frame: roi.referenceFrame })}
        />
      </div>
      <RoiFigure url={artifactUrl(artifacts, "roi:agreement")} caption={t("cctv.roi.result.agreement")}>
        <p className="text-xs text-text-faint">{t("cctv.roi.agreement")}</p>
      </RoiFigure>
      <Notices roi={roi} />
    </section>
  );
}
