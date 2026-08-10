import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Download, Trash2 } from "lucide-react";
import { useTranslation } from "../../i18n/LocaleProvider";
import { useState } from "react";
import { createInpaintVersion, optimizeGenerationModel } from "../../services/generation";
import { useGenerationModelInstall } from "../../hooks/useGenerationJob";
import { DEFAULT_INSTALL_POLL_INTERVAL_MS, useDeleteModel, useInstalledModels } from "../../hooks/useModels";
import type { ModelResponse } from "../../lib/apiTypes";
import { formatModelSize } from "../../lib/sizeFormat";
import { DeleteConfirmDialog } from "./DeleteConfirmDialog";
import { GenerationHfSearch } from "./GenerationHfSearch";
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

function isInpaintVariant(model: ModelResponse): boolean {
  return model.name.includes("(inpainting)");
}

function isOptimizedVariant(model: ModelResponse): boolean {
  return model.name.includes("(optimized)");
}

// La fusión de grafo re-exporta los pesos de origen: una variante de inpainting
// no los tiene (sus pesos son el resultado del merge) y la optimizada ya pasó por
// acá. El servidor rechaza los dos casos igual; esto evita ofrecer el botón.
function canOptimize(model: ModelResponse): boolean {
  return (
    model.status === "installed" && !isInpaintVariant(model) && !isOptimizedVariant(model)
  );
}

const SECONDARY_ACTION_CLASS =
  "rounded-sm border border-border bg-surface px-2 py-1.5 text-xs text-text-dim transition-[border-color,color] duration-fast hover:border-accent hover:text-text disabled:cursor-not-allowed disabled:opacity-40 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";

function DiffusionModelRow({
  model,
  onRequestDelete,
  onCreateInpaint,
  inpaintPending,
  onOptimize,
  optimizePending,
}: {
  model: ModelResponse;
  onRequestDelete: (model: ModelResponse) => void;
  onCreateInpaint: (model: ModelResponse) => void;
  inpaintPending: boolean;
  onOptimize: (model: ModelResponse) => void;
  optimizePending: boolean;
}) {
  const { t } = useTranslation();
  return (
    <li className="flex items-center justify-between gap-4 rounded border border-border bg-surface px-4 py-3">
      <div className="flex flex-col gap-1">
        <span className="text-sm text-text">{model.name}</span>
        <span className="font-mono-tabular text-xs text-text-dim">{formatModelSize(model.sizeBytes)}</span>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {canOptimize(model) && (
          <button
            type="button"
            title={t("models.optimize.hint")}
            disabled={optimizePending}
            onClick={() => onOptimize(model)}
            className={SECONDARY_ACTION_CLASS}
          >
            {t("models.optimize.button")}
          </button>
        )}
        {!isInpaintVariant(model) && !isOptimizedVariant(model) && (
          <button
            type="button"
            title={t("models.createInpaint.hint")}
            disabled={inpaintPending}
            onClick={() => onCreateInpaint(model)}
            className={SECONDARY_ACTION_CLASS}
          >
            {t("models.createInpaint.button")}
          </button>
        )}
        <button
          type="button"
          aria-label={t("models.delete.aria", { name: model.name })}
          onClick={() => onRequestDelete(model)}
          className="rounded-sm border border-border bg-surface p-2 text-text-faint transition-[border-color,color] duration-fast hover:border-danger hover:text-danger focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
        >
          <Trash2 aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
        </button>
      </div>
    </li>
  );
}

function DiffusionModelsEmptyState() {
  const { t } = useTranslation();
  return <p className="text-sm text-text-faint">{t("models.generation.empty")}</p>;
}

function DiffusionModelsList({
  models,
  onRequestDelete,
  onCreateInpaint,
  inpaintPending,
  onOptimize,
  optimizePending,
}: {
  models: ModelResponse[];
  onRequestDelete: (model: ModelResponse) => void;
  onCreateInpaint: (model: ModelResponse) => void;
  inpaintPending: boolean;
  onOptimize: (model: ModelResponse) => void;
  optimizePending: boolean;
}) {
  if (models.length === 0) {
    return <DiffusionModelsEmptyState />;
  }
  return (
    <ul className="flex flex-col gap-2">
      {models.map((model) => (
        <DiffusionModelRow
          key={model.id}
          model={model}
          onRequestDelete={onRequestDelete}
          onCreateInpaint={onCreateInpaint}
          inpaintPending={inpaintPending}
          onOptimize={onOptimize}
          optimizePending={optimizePending}
        />
      ))}
    </ul>
  );
}

function DeleteFailedNote({ error }: { error: unknown }) {
  const { t } = useTranslation();
  const message = error instanceof Error ? error.message : t("models.delete.failed");
  return (
    <p role="alert" className="text-sm text-danger">
      {message}
    </p>
  );
}

export function GenerationModelsSection({ pollIntervalMs = DEFAULT_INSTALL_POLL_INTERVAL_MS }: GenerationModelsSectionProps) {
  const { t } = useTranslation();
  const [repoId, setRepoId] = useState("");
  const [pendingDelete, setPendingDelete] = useState<ModelResponse | null>(null);
  const { phase, progressPct, stageLabel, errorMessage, install, cancelConversion, reset } =
    useGenerationModelInstall(pollIntervalMs);
  const modelsQuery = useInstalledModels();
  const deleteMutation = useDeleteModel();
  const queryClient = useQueryClient();
  const inpaintMutation = useMutation({
    mutationFn: (model: ModelResponse) => createInpaintVersion(model.id),
    onSuccess: () => {
      // El merge corre como conversión: la entrada "(inpainting)" aparece en
      // esta misma lista con su progreso apenas se refresca. Invalidar también
      // las conversiones activas engancha la barra de progreso AL INSTANTE —
      // sin esto solo aparecía tras recargar la página (visto real).
      queryClient.invalidateQueries({ queryKey: ["models"] });
      queryClient.invalidateQueries({ queryKey: ["generation-active-conversions"] });
    },
  });
  const optimizeMutation = useMutation({
    mutationFn: (model: ModelResponse) => optimizeGenerationModel(model.id),
    onSuccess: () => {
      // Misma razón que el merge: la variante aparece en esta lista con su
      // progreso apenas se refrescan modelos y conversiones activas.
      queryClient.invalidateQueries({ queryKey: ["models"] });
      queryClient.invalidateQueries({ queryKey: ["generation-active-conversions"] });
    },
  });

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
      <h2 className="font-heading text-sm font-semibold text-text">{t("models.generation.sectionTitle")}</h2>
      <GenerationHfSearch />
      <RepoIdForm repoId={repoId} onRepoIdChange={setRepoId} onSubmit={handleSubmit} disabled={installInFlight} />
      {installInFlight && (
        <InstallProgress
          phase={phase}
          progressPct={progressPct}
          stageLabel={stageLabel}
          onCancel={cancelConversion}
        />
      )}
      {phase === "error" && errorMessage && <InstallError message={errorMessage} onRetry={reset} />}
      <DiffusionModelsList
        models={diffusionModels}
        onRequestDelete={setPendingDelete}
        onCreateInpaint={(model) => inpaintMutation.mutate(model)}
        inpaintPending={inpaintMutation.isPending}
        onOptimize={(model) => optimizeMutation.mutate(model)}
        optimizePending={optimizeMutation.isPending}
      />
      {inpaintMutation.isError && (
        <p className="text-sm text-danger">
          {inpaintMutation.error instanceof Error
            ? inpaintMutation.error.message
            : t("models.createInpaint.failed")}
        </p>
      )}
      {optimizeMutation.isError && (
        <p role="alert" className="text-sm text-danger">
          {optimizeMutation.error instanceof Error
            ? optimizeMutation.error.message
            : t("models.optimize.failed")}
        </p>
      )}
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
