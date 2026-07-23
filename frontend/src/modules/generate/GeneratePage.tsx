import { GeneratePanel } from "./GeneratePanel";

export function GeneratePage() {
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="font-heading text-2xl font-semibold text-text">Generate</h1>
        <p className="mt-1 text-sm text-text-dim">Create an image from a text prompt, optionally upscaling it on completion.</p>
      </div>
      <GeneratePanel />
    </div>
  );
}
