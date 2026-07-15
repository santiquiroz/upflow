import { AlertTriangle, Lock, Trash2 } from "lucide-react";
import { useState, type ReactNode } from "react";
import { useDeleteModel, useInstalledModels } from "../../hooks/useModels";
import type { ModelResponse } from "../../lib/apiTypes";
import { formatModelSize } from "../../lib/sizeFormat";

function isBuiltinModel(model: ModelResponse): boolean {
  return model.kind === "builtin-ncnn";
}

function formatModelMeta(model: ModelResponse): string {
  const scale = model.scale ? `${model.scale}x` : "—";
  return `${scale} · ${model.arch ?? model.kind} · ${formatModelSize(model.sizeBytes)}`;
}

function statusTextClassName(status: string): string {
  if (status === "error") {
    return "text-danger";
  }
  return "text-warn";
}

function ModelStatusNote({ model }: { model: ModelResponse }) {
  if (model.status === "installed") {
    return null;
  }
  const label = model.status === "converting" ? "Converting…" : (model.error ?? model.status);
  return <span className={`text-xs ${statusTextClassName(model.status)}`}>{label}</span>;
}

function BuiltinBadge() {
  return (
    <span
      title="Built-in models cannot be removed"
      className="flex shrink-0 items-center gap-1.5 rounded-sm bg-surface-2 px-2 py-1 text-xs text-text-faint"
    >
      <Lock aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
      Built-in
    </span>
  );
}

function DeleteButton({ model, onRequestDelete }: { model: ModelResponse; onRequestDelete: (model: ModelResponse) => void }) {
  return (
    <button
      type="button"
      aria-label={`Delete ${model.name}`}
      onClick={() => onRequestDelete(model)}
      className="shrink-0 rounded-sm border border-border bg-surface p-2 text-text-faint transition-[border-color,color] duration-fast hover:border-danger hover:text-danger focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
    >
      <Trash2 aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
    </button>
  );
}

function ModelRow({
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
        <span className="font-mono-tabular text-xs text-text-dim">{formatModelMeta(model)}</span>
        <ModelStatusNote model={model} />
      </div>
      {isBuiltinModel(model) ? <BuiltinBadge /> : <DeleteButton model={model} onRequestDelete={onRequestDelete} />}
    </li>
  );
}

function OnnxEmptyState() {
  return <p className="text-sm text-text-faint">No custom ONNX models installed yet — search above to add one.</p>;
}

function ModelGroup({
  label,
  models,
  emptyState,
  onRequestDelete,
}: {
  label: string;
  models: ModelResponse[];
  emptyState?: ReactNode;
  onRequestDelete: (model: ModelResponse) => void;
}) {
  return (
    <div className="flex flex-col gap-2">
      <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">{label}</h2>
      {models.length === 0 && emptyState}
      {models.length > 0 && (
        <ul className="flex flex-col gap-2">
          {models.map((model) => (
            <ModelRow key={model.id} model={model} onRequestDelete={onRequestDelete} />
          ))}
        </ul>
      )}
    </div>
  );
}

function DeleteConfirmDialog({
  model,
  onCancel,
  onConfirm,
  isPending,
}: {
  model: ModelResponse;
  onCancel: () => void;
  onConfirm: () => void;
  isPending: boolean;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="delete-model-title"
        className="flex w-full max-w-sm flex-col gap-4 rounded border border-border bg-surface p-5"
      >
        <div className="flex items-start gap-3">
          <AlertTriangle aria-hidden="true" className="mt-0.5 h-5 w-5 shrink-0 text-danger" strokeWidth={1.75} />
          <div className="flex flex-col gap-1">
            <h2 id="delete-model-title" className="font-heading text-sm font-semibold text-text">
              Delete {model.name}?
            </h2>
            <p className="text-sm text-text-dim">This removes the model file from disk. This cannot be undone.</p>
          </div>
        </div>
        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-sm border border-border bg-surface px-3 py-1.5 text-sm text-text-dim transition-[border-color,color] duration-fast hover:border-text-faint hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={isPending}
            className="rounded-sm bg-danger px-3 py-1.5 text-sm font-medium text-bg transition-[background-color,opacity] duration-fast hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
          >
            Delete
          </button>
        </div>
      </div>
    </div>
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

export function InstalledModels() {
  const modelsQuery = useInstalledModels();
  const deleteMutation = useDeleteModel();
  const [pendingDelete, setPendingDelete] = useState<ModelResponse | null>(null);

  function handleConfirmDelete() {
    if (!pendingDelete) {
      return;
    }
    deleteMutation.mutate(pendingDelete.id, { onSuccess: () => setPendingDelete(null) });
  }

  if (modelsQuery.isLoading) {
    return <p className="text-sm text-text-dim">Loading installed models…</p>;
  }

  if (modelsQuery.isError) {
    return <p className="text-sm text-danger">Could not load installed models.</p>;
  }

  const models = modelsQuery.data?.models ?? [];
  const builtinModels = models.filter(isBuiltinModel);
  const onnxModels = models.filter((model) => !isBuiltinModel(model));

  return (
    <div className="flex flex-col gap-6">
      <ModelGroup label="Built-in" models={builtinModels} onRequestDelete={setPendingDelete} />
      <ModelGroup label="ONNX" models={onnxModels} emptyState={<OnnxEmptyState />} onRequestDelete={setPendingDelete} />
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
