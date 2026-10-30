import { useTranslation } from "../../i18n/LocaleProvider";
import type { EraseZone } from "../../lib/apiTypes";
import { peopleOverlayLabelSize } from "./PeopleOverlay";

interface ReviewOverlayProps {
  zones: EraseZone[];
  resultUrl: string | null;
  hoveredZone: number | null;
  onHoverZone: (index: number | null) => void;
  imageWidth: number;
  imageHeight: number;
}

function ZoneMarker({ zone, hovered, onHoverZone, fontSize }: {
  zone: EraseZone;
  hovered: boolean;
  onHoverZone: (index: number | null) => void;
  fontSize: number;
}) {
  const [x1, y1, x2, y2] = zone.box!;
  const radius = fontSize;
  const color = hovered ? "#f5a524" : "#ffffff";
  return (
    <g
      data-testid={`zone-marker-${zone.index}`}
      className="pointer-events-auto cursor-default"
      onMouseEnter={() => onHoverZone(zone.index)}
      onMouseLeave={() => onHoverZone(null)}
    >
      <rect
        x={x1}
        y={y1}
        width={x2 - x1}
        height={y2 - y1}
        fill="none"
        stroke={color}
        strokeWidth={hovered ? 3 : 1.5}
        strokeDasharray={hovered ? undefined : "5 4"}
        vectorEffect="non-scaling-stroke"
      />
      <circle cx={x1 + radius} cy={y1 + radius} r={radius} fill={color} stroke="#111" strokeWidth={1} vectorEffect="non-scaling-stroke" />
      <text x={x1 + radius} y={y1 + radius + fontSize * 0.35} textAnchor="middle" fill="#111" fontSize={fontSize} fontWeight={700}>
        {zone.index + 1}
      </text>
    </g>
  );
}

export function ReviewOverlay({ zones, resultUrl, hoveredZone, onHoverZone, imageWidth, imageHeight }: ReviewOverlayProps) {
  const { t } = useTranslation();
  const { fontSize } = peopleOverlayLabelSize(imageWidth);
  return (
    <>
      {resultUrl && (
        <img src={resultUrl} alt={t("editor.review.result")} className="pointer-events-none absolute inset-0 h-full w-full" />
      )}
      <svg
        aria-label={t("editor.review.numbers")}
        className="pointer-events-none absolute inset-0 h-full w-full overflow-visible"
        viewBox={`0 0 ${imageWidth} ${imageHeight}`}
        preserveAspectRatio="none"
      >
        {zones.filter((zone) => zone.box).map((zone) => (
          <ZoneMarker
            key={zone.index}
            zone={zone}
            hovered={hoveredZone === zone.index}
            onHoverZone={onHoverZone}
            fontSize={fontSize}
          />
        ))}
      </svg>
    </>
  );
}
