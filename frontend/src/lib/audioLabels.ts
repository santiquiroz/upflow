// Wire-value -> human label maps for the audio enhancement engines, shared by
// the Audio panel, the job queue card and the job detail modal so the same
// mode never shows two different names.

const DENOISE_LABELS: Record<string, string> = {
  deepfilter: "DeepFilterNet",
  rnnoise: "RNNoise",
};

const RESTORE_LABELS: Record<string, string> = {
  apollo: "Apollo",
  audiosr: "AudioSR",
};

export function denoiseLabel(mode: string | null): string {
  if (!mode) {
    return "None";
  }
  return DENOISE_LABELS[mode] ?? mode;
}

export function restoreLabel(mode: string | null): string {
  if (!mode) {
    return "None";
  }
  return RESTORE_LABELS[mode] ?? mode;
}
