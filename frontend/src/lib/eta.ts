// Honest ETA: hidden (null) whenever the rate isn't trustworthy yet, rather
// than ever inventing a number. See app/services/progress.py for the
// backend-side progress/stage model these samples are derived from.

export interface EtaSample {
  progress: number; // 0..1, monotonic as displayed by the caller
  t: number; // ms, any monotonic clock (Date.now())
}

const RECENT_SAMPLE_WINDOW = 5;
const PROGRESS_COMPLETE = 1;

function recentWindow(samples: EtaSample[]): EtaSample[] {
  return samples.slice(-RECENT_SAMPLE_WINDOW);
}

function ratePerSecond(oldest: EtaSample, newest: EtaSample): number {
  const deltaSeconds = (newest.t - oldest.t) / 1000;
  if (deltaSeconds <= 0) {
    return 0;
  }
  return (newest.progress - oldest.progress) / deltaSeconds;
}

export function estimateEta(samples: EtaSample[]): number | null {
  if (samples.length < 2) {
    return null;
  }
  const window = recentWindow(samples);
  const newest = window[window.length - 1];
  if (newest.progress >= PROGRESS_COMPLETE) {
    return null;
  }
  const rate = ratePerSecond(window[0], newest);
  if (rate <= 0) {
    return null;
  }
  const etaSeconds = (PROGRESS_COMPLETE - newest.progress) / rate;
  return Number.isFinite(etaSeconds) ? etaSeconds : null;
}

function formatMinutesAndSeconds(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60);
  const remainderSeconds = totalSeconds % 60;
  return remainderSeconds === 0 ? `~${minutes} min` : `~${minutes} min ${remainderSeconds} s`;
}

export function formatEta(seconds: number): string {
  const rounded = Math.round(seconds);
  if (rounded < 1) {
    return "<1 min";
  }
  if (rounded < 60) {
    return `~${rounded} s`;
  }
  return formatMinutesAndSeconds(rounded);
}
