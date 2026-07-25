import { Search } from "lucide-react";

export const DEFAULT_SEARCH_DEBOUNCE_MS = 400;

export function SearchEmptyState({ message }: { message: string }) {
  return (
    <div className="flex flex-col items-center gap-2 rounded border border-dashed border-border bg-surface px-6 py-10 text-center">
      <Search aria-hidden="true" className="h-6 w-6 text-text-faint" strokeWidth={1.5} />
      <p className="text-sm text-text-faint">{message}</p>
    </div>
  );
}

export function NoResultsState({ query }: { query: string }) {
  return (
    <div className="flex flex-col items-center gap-2 rounded border border-border bg-surface px-6 py-10 text-center">
      <Search aria-hidden="true" className="h-6 w-6 text-text-faint" strokeWidth={1.5} />
      <p className="text-sm text-text-dim">No models found for &quot;{query}&quot;.</p>
    </div>
  );
}

export function SearchErrorState() {
  return (
    <p role="alert" className="rounded border border-danger bg-surface-2 px-3 py-2 text-sm text-danger">
      Hugging Face search failed. Try again.
    </p>
  );
}

export function SearchLoadingState() {
  return (
    <p role="status" className="text-sm text-text-dim">
      Searching…
    </p>
  );
}

export function SearchInput({ value, onChange }: { value: string; onChange: (value: string) => void }) {
  return (
    <label className="flex flex-col gap-1.5">
      <span className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
        Search Hugging Face
      </span>
      <div className="flex items-center gap-2 rounded border border-border bg-surface px-3 py-2 focus-within:border-accent">
        <Search aria-hidden="true" className="h-4 w-4 shrink-0 text-text-faint" strokeWidth={1.75} />
        <input
          type="search"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder="owner/model-name or keywords"
          className="w-full bg-transparent text-sm text-text placeholder:text-text-faint focus:outline-none"
        />
      </div>
    </label>
  );
}
