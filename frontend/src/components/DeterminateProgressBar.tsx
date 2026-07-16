interface DeterminateProgressBarProps {
  label: string;
  percent: number;
}

export function DeterminateProgressBar({ label, percent }: DeterminateProgressBarProps) {
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuenow={Math.round(percent)}
      aria-valuemin={0}
      aria-valuemax={100}
      className="h-1.5 w-full overflow-hidden rounded-sm bg-surface-2"
    >
      <div
        className="h-full rounded-sm bg-accent transition-[width] duration-normal"
        style={{ width: `${percent}%` }}
      />
    </div>
  );
}
