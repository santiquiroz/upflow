import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Bookmark, Trash2 } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "../../i18n/LocaleProvider";
import { apiDelete, apiGet, apiPostJson } from "../../lib/api";

export interface SavedPrompt {
  id: string;
  name: string;
  prompt: string;
  negativePrompt: string;
  mode: string;
}

// Lo que se guarda es DATO del usuario y no copia de la app: viaja y se muestra
// tal cual lo escribio, sin traducir.
export function SavedPrompts({
  mode,
  currentPrompt,
  currentNegativePrompt,
  onApply,
}: {
  mode: string;
  currentPrompt: string;
  currentNegativePrompt: string;
  onApply: (saved: SavedPrompt) => void;
}) {
  const { t } = useTranslation();
  const queryClient = useQueryClient();
  const [name, setName] = useState("");

  const savedQuery = useQuery({
    queryKey: ["savedPrompts"],
    queryFn: () => apiGet<{ prompts: SavedPrompt[] }>("/generation/saved-prompts"),
  });

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ["savedPrompts"] });

  const saveMutation = useMutation({
    mutationFn: () =>
      apiPostJson<SavedPrompt>("/generation/saved-prompts", {
        name,
        prompt: currentPrompt,
        negativePrompt: currentNegativePrompt,
        mode,
      }),
    onSuccess: () => {
      setName("");
      void invalidate();
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => apiDelete(`/generation/saved-prompts/${id}`),
    onSuccess: invalidate,
  });

  const mine = (savedQuery.data?.prompts ?? []).filter((saved) => saved.mode === mode);
  const canSave = name.trim().length > 0 && currentPrompt.trim().length > 0;

  return (
    <div className="flex flex-col gap-2">
      <span className="text-xs font-medium text-text-dim">{t("generate.saved.title")}</span>

      {mine.length === 0 ? (
        <span className="text-xs text-text-faint">{t("generate.saved.empty")}</span>
      ) : (
        <div className="flex flex-wrap gap-2">
          {mine.map((saved) => (
            <span
              key={saved.id}
              className="inline-flex items-center gap-1 rounded-sm border border-border bg-surface pl-3 pr-1 py-1 text-xs text-text-dim"
            >
              <button
                type="button"
                onClick={() => onApply(saved)}
                className="transition-colors duration-fast hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
              >
                {saved.name}
              </button>
              <button
                type="button"
                aria-label={t("generate.saved.delete", { name: saved.name })}
                onClick={() => deleteMutation.mutate(saved.id)}
                className="rounded-sm p-1 text-text-faint transition-colors duration-fast hover:text-danger focus-visible:outline focus-visible:outline-2 focus-visible:outline-danger"
              >
                <Trash2 aria-hidden="true" className="h-3 w-3" strokeWidth={1.75} />
              </button>
            </span>
          ))}
        </div>
      )}

      <div className="flex items-center gap-2">
        <input
          type="text"
          value={name}
          maxLength={80}
          onChange={(event) => setName(event.target.value)}
          placeholder={t("generate.saved.namePlaceholder")}
          className="w-48 rounded border border-border bg-surface px-2 py-1 text-xs text-text placeholder:text-text-faint focus:border-accent focus:outline-none"
        />
        <button
          type="button"
          onClick={() => saveMutation.mutate()}
          disabled={!canSave || saveMutation.isPending}
          className="inline-flex items-center gap-1.5 rounded-sm border border-border bg-surface px-3 py-1 text-xs text-text-dim transition-[border-color,color] duration-fast hover:border-accent hover:text-text disabled:cursor-not-allowed disabled:opacity-40"
        >
          <Bookmark aria-hidden="true" className="h-3 w-3" strokeWidth={1.75} />
          {t("generate.saved.save")}
        </button>
      </div>

      {saveMutation.isError && (
        <p role="alert" className="text-xs text-danger">
          {saveMutation.error instanceof Error
            ? saveMutation.error.message
            : t("generate.saved.failed")}
        </p>
      )}
    </div>
  );
}
