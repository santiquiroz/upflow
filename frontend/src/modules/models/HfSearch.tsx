import { Search } from "lucide-react";
import { useState } from "react";
import { useDebouncedValue } from "../../hooks/useDebouncedValue";
import { useHfSearchResults } from "../../hooks/useModels";
import { HfResultCard } from "./HfResultCard";

export const DEFAULT_SEARCH_DEBOUNCE_MS = 400;

interface HfSearchProps {
  debounceMs?: number;
}

function SearchEmptyState() {
  return (
    <div className="flex flex-col items-center gap-2 rounded border border-dashed border-border bg-surface px-6 py-10 text-center">
      <Search aria-hidden="true" className="h-6 w-6 text-text-faint" strokeWidth={1.5} />
      <p className="text-sm text-text-faint">Search Hugging Face for an ONNX upscaling model to install.</p>
    </div>
  );
}

function NoResultsState({ query }: { query: string }) {
  return (
    <div className="flex flex-col items-center gap-2 rounded border border-border bg-surface px-6 py-10 text-center">
      <Search aria-hidden="true" className="h-6 w-6 text-text-faint" strokeWidth={1.5} />
      <p className="text-sm text-text-dim">No models found for &quot;{query}&quot;.</p>
    </div>
  );
}

function SearchErrorState() {
  return (
    <p role="alert" className="rounded border border-danger bg-surface-2 px-3 py-2 text-sm text-danger">
      Hugging Face search failed. Try again.
    </p>
  );
}

function SearchLoadingState() {
  return (
    <p role="status" className="text-sm text-text-dim">
      Searching…
    </p>
  );
}

function SearchInput({ value, onChange }: { value: string; onChange: (value: string) => void }) {
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

function SearchResults({ query }: { query: string }) {
  const searchQuery = useHfSearchResults(query);

  if (searchQuery.isLoading) {
    return <SearchLoadingState />;
  }

  if (searchQuery.isError) {
    return <SearchErrorState />;
  }

  const results = searchQuery.data?.results ?? [];

  if (results.length === 0) {
    return <NoResultsState query={query} />;
  }

  return (
    <ul className="flex flex-col gap-3">
      {results.map((result) => (
        <li key={result.id}>
          <HfResultCard result={result} />
        </li>
      ))}
    </ul>
  );
}

export function HfSearch({ debounceMs = DEFAULT_SEARCH_DEBOUNCE_MS }: HfSearchProps) {
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query, debounceMs);
  const trimmedQuery = debouncedQuery.trim();

  return (
    <div className="flex flex-col gap-4">
      <SearchInput value={query} onChange={setQuery} />
      {trimmedQuery.length === 0 ? <SearchEmptyState /> : <SearchResults query={trimmedQuery} />}
    </div>
  );
}
