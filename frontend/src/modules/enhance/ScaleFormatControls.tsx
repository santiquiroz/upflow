const OUTPUT_FORMATS = ["png", "jpg", "webp"] as const;

interface ScaleFormatControlsProps {
  allowedScales: number[];
  scale: number;
  onScaleChange: (scale: number) => void;
  format: string;
  onFormatChange: (format: string) => void;
}

function segmentButtonClassName(isActive: boolean): string {
  const base =
    "rounded-sm border px-3 py-1.5 text-sm transition-[background-color,border-color,color] duration-fast focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";
  if (isActive) {
    return `${base} border-accent bg-accent text-bg`;
  }
  return `${base} border-border bg-surface text-text-dim hover:border-text-faint hover:text-text`;
}

export function ScaleFormatControls({
  allowedScales,
  scale,
  onScaleChange,
  format,
  onFormatChange,
}: ScaleFormatControlsProps) {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-col gap-2">
        <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">Scale</span>
        <div role="group" aria-label="Scale" className="flex gap-2">
          {allowedScales.map((allowedScale) => (
            <button
              key={allowedScale}
              type="button"
              aria-pressed={allowedScale === scale}
              className={`font-mono-tabular ${segmentButtonClassName(allowedScale === scale)}`}
              onClick={() => onScaleChange(allowedScale)}
            >
              {allowedScale}x
            </button>
          ))}
        </div>
      </div>
      <div className="flex flex-col gap-2">
        <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">Format</span>
        <div role="group" aria-label="Output format" className="flex gap-2">
          {OUTPUT_FORMATS.map((outputFormat) => (
            <button
              key={outputFormat}
              type="button"
              aria-pressed={outputFormat === format}
              className={segmentButtonClassName(outputFormat === format)}
              onClick={() => onFormatChange(outputFormat)}
            >
              {outputFormat.toUpperCase()}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
