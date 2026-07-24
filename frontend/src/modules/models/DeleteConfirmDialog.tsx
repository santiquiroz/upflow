import { AlertTriangle } from "lucide-react";
import { Modal } from "../../components/Modal";
import type { ModelResponse } from "../../lib/apiTypes";

interface DeleteConfirmDialogProps {
  model: ModelResponse;
  onCancel: () => void;
  onConfirm: () => void;
  isPending: boolean;
}

export function DeleteConfirmDialog({ model, onCancel, onConfirm, isPending }: DeleteConfirmDialogProps) {
  return (
    <Modal titleId="delete-model-title" onClose={onCancel}>
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
    </Modal>
  );
}
