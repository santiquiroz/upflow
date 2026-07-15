import { DeviceDefault } from "../modules/models/DeviceDefault";
import { HfSearch } from "../modules/models/HfSearch";
import { InstalledModels } from "../modules/models/InstalledModels";

export function ModelsPage() {
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="font-heading text-2xl font-semibold text-text">Models</h1>
        <p className="mt-1 text-sm text-text-dim">
          Search Hugging Face, install ONNX models, and manage what&apos;s on disk.
        </p>
      </div>
      <div className="grid grid-cols-[1fr_320px] gap-6 max-[900px]:grid-cols-1">
        <div className="flex flex-col gap-8">
          <HfSearch />
          <InstalledModels />
        </div>
        <DeviceDefault />
      </div>
    </div>
  );
}
