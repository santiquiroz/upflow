export function IndeterminateProgressBar({ label }: { label: string }) {
  return (
    <div role="progressbar" aria-label={label} aria-busy="true" className="h-1.5 w-full overflow-hidden rounded-sm bg-surface-2">
      <div className="job-progress-bar h-full w-1/3 rounded-sm bg-accent" />
    </div>
  );
}
