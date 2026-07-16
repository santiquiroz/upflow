import { AudioPanel } from "./AudioPanel";

export function AudioPage() {
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="font-heading text-2xl font-semibold text-text">Audio</h1>
        <p className="mt-1 text-sm text-text-dim">Clean up noise and restore compression artefacts in an audio file.</p>
      </div>
      <AudioPanel />
    </div>
  );
}
