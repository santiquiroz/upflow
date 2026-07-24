import { Download, Trash2 } from "lucide-react";
import { useState } from "react";
import { useGenerationModelInstall } from "../../hooks/useGenerationJob";
import { DEFAULT_INSTALL_POLL_INTERVAL_MS, useDeleteModel, useInstalledModels } from "../../hooks/useModels";
import type { ModelResponse } from "../../lib/apiTypes";
import { formatModelSize } from "../../lib/sizeFormat";
import { DeleteConfirmDialog } from "./DeleteConfirmDialog";
import { InstallError, InstallProgress, isInstallInFlight } from "./installUi";

export const GENERATION_MODEL_REPO_PLACEHOLDER = "amd/stable-diffusion-1.5_io16_amdgpu";

interface GenerationModelsSectionProps {
  pollIntervalMs?: number;
}

function isDiffusionModel(model: ModelResponse): boolean {
  return model.kind === "diffusion-onnx";
}

function RepoIdForm({
  repoId,
  onRepoIdChange,
  onSubmit,
  disabled,
}: {
  repoId: string;
  onRepoIdChange: (value: string) => void;
  onSubmit: () => void;
  disabled: boolean;
}) {
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit();
      }}
      className="flex items-center gap-2"
    >
      <input
        type="text"
        value={repoId}
        onChange={(event) => onRepoIdChange(event.target.value)}
        placeholder={GENERATION_MODEL_REPO_PLACEHOLDER}
        disabled={disabled}
        className="w-full rounded border border-border bg-surface px-3 py-2 text-sm text-text placeholder:text-text-faint focus:border-accent focus:outline-none disabled:opacity-60"
      />
      <button
        type="submit"
        disabled={disabled || repoId.trim().length === 0}
        className="inline-flex shrink-0 items-center gap-2 rounded bg-accent px-3 py-1.5 text-sm font-medium text-bg transition-[background-color] duration-fast hover:bg-accent-hover active:bg-accent-press disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        <Download aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
        Install
      </button>
    </form>
  );
}

function DiffusionModelRow({
  model,
  onRequestDelete,
}: {
  model: ModelResponse;
  onRequestDelete: (model: ModelResponse) => void;
}) {
  return (
    <li className="flex items-center justify-between gap-4 rounded border border-border bg-surface px-4 py-3">
      <div className="flex flex-col gap-1">
        <span className="text-sm text-text">{model.name}</span>
        <span className="font-mono-tabular text-xs text-text-dim">{formatModelSize(model.sizeBytes)}</span>
      </div>
      <button
        type="button"
        aria-label={`Delete ${model.name}`}
        onClick={() => onRequestDelete(model)}
        className="shrink-0 rounded-sm border border-border bg-surface p-2 text-text-faint transition-[border-color,color] duration-fast hover:border-danger hover:text-danger focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        <Trash2 aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
      </button>
    </li>
  );
}

function DiffusionModelsEmptyState() {
  return <p className="text-sm text-text-faint">No generation models installed yet.</p>;
}

function DiffusionModelsList({
  models,
  onRequestDelete,
}: {
  models: ModelResponse[];
  onRequestDelete: (model: ModelResponse) => void;
}) {
  if (models.length === 0) {
    return <DiffusionModelsEmptyState />;
  }
  return (
    <ul className="flex flex-col gap-2">
      {models.map((model) => (
        <DiffusionModelRow key={model.id} model={model} onRequestDelete={onRequestDelete} />
      ))}
    </ul>
  );
}

function DeleteFailedNote({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : "Could not delete the model.";
  return (
    <p role="alert" className="text-sm text-danger">
      {message}
    </p>
  );
}

export function GenerationModelsSection({ pollIntervalMs = DEFAULT_INSTALL_POLL_INTERVAL_MS }: GenerationModelsSectionProps) {
  const [repoId, setRepoId] = useState("");
  const [pendingDelete, setPendingDelete] = useState<ModelResponse | null>(null);
  const { phase, progressPct, errorMessage, install, reset } = useGenerationModelInstall(pollIntervalMs);
  const modelsQuery = useInstalledModels();
  const deleteMutation = useDeleteModel();

  const diffusionModels = (modelsQuery.data?.models ?? []).filter(isDiffusionModel);
  const installInFlight = isInstallInFlight(phase);

  function handleSubmit() {
    const trimmedRepoId = repoId.trim();
    if (!trimmedRepoId) {
      return;
    }
    install(trimmedRepoId);
  }

  function handleConfirmDelete() {
    if (!pendingDelete) {
      return;
    }
    deleteMutation.mutate(pendingDelete.id, { onSuccess: () => setPendingDelete(null) });
  }

  return (
    <div className="flex flex-col gap-4 rounded border border-border bg-surface p-4">
      <h2 className="font-heading text-sm font-semibold text-text">Generation models (Stable Diffusion)</h2>
      <RepoIdForm repoId={repoId} onRepoIdChange={setRepoId} onSubmit={handleSubmit} disabled={installInFlight} />
      {installInFlight && <InstallProgress phase={phase} progressPct={progressPct} />}
      {phase === "error" && errorMessage && <InstallError message={errorMessage} onRetry={reset} />}
      <DiffusionModelsList models={diffusionModels} onRequestDelete={setPendingDelete} />
      {deleteMutation.isError && <DeleteFailedNote error={deleteMutation.error} />}
      {pendingDelete && (
        <DeleteConfirmDialog
          model={pendingDelete}
          onCancel={() => setPendingDelete(null)}
          onConfirm={handleConfirmDelete}
          isPending={deleteMutation.isPending}
        />
      )}
    </div>
  );
}
