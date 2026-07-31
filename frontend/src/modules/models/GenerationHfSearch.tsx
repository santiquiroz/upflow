import { useState } from "react";
import { useGenerationHfSearchResults } from "../../hooks/useGenerationJob";
import { useDebouncedValue } from "../../hooks/useDebouncedValue";
import type { HfModelSearchResultResponse } from "../../lib/apiTypes";
import { GenerationModelCard } from "./GenerationModelCard";
import {
  DEFAULT_SEARCH_DEBOUNCE_MS,
  NoResultsState,
  SearchErrorState,
  SearchInput,
  SearchLoadingState,
} from "./hfSearchUi";

interface GenerationHfSearchProps {
  debounceMs?: number;
}

type CompatFilter = "all" | "ready" | "conversion";

const COMPAT_FILTERS: { value: CompatFilter; label: string }[] = [
  { value: "all", label: "Todos" },
  { value: "ready", label: "Listos para usar" },
  { value: "conversion", label: "Con conversión" },
];

// ready_onnx corre tal cual; needs_conversion y single_file pasan por la
// conversión (~30-45 min). El resto (gated, incompatible) solo aparece en
// "Todos", con su motivo en la tarjeta.
function matchesFilter(result: HfModelSearchResultResponse, filter: CompatFilter): boolean {
  if (filter === "all") {
    return true;
  }
  if (filter === "ready") {
    return result.compat === "ready_onnx";
  }
  return result.compat === "needs_conversion" || result.compat === "single_file";
}

function CompatFilterChips({
  value,
  onChange,
}: {
  value: CompatFilter;
  onChange: (next: CompatFilter) => void;
}) {
  return (
    <fieldset className="flex flex-wrap gap-2">
      <legend className="sr-only">Filtrar por compatibilidad</legend>
      {COMPAT_FILTERS.map((option) => (
        <label
          key={option.value}
          className={`cursor-pointer rounded border px-3 py-1.5 text-sm ${
            value === option.value
              ? "border-accent bg-surface-2 text-text"
              : "border-border bg-surface text-text-dim"
          }`}
        >
          <input
            type="radio"
            name="generation-compat-filter"
            className="sr-only"
            checked={value === option.value}
            onChange={() => onChange(option.value)}
          />
          {option.label}
        </label>
      ))}
    </fieldset>
  );
}

function SearchResults({ query, filter }: { query: string; filter: CompatFilter }) {
  const searchQuery = useGenerationHfSearchResults(query);

  if (searchQuery.isLoading) {
    return <SearchLoadingState />;
  }

  if (searchQuery.isError) {
    return <SearchErrorState />;
  }

  const results = (searchQuery.data?.results ?? []).filter((result) =>
    matchesFilter(result, filter),
  );

  if (results.length === 0) {
    return <NoResultsState query={query} />;
  }

  return (
    <ul className="flex flex-col gap-3">
      {results.map((result) => (
        <li key={result.id}>
          <GenerationModelCard result={result} />
        </li>
      ))}
    </ul>
  );
}

export function GenerationHfSearch({
  debounceMs = DEFAULT_SEARCH_DEBOUNCE_MS,
}: GenerationHfSearchProps) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<CompatFilter>("all");
  const debouncedQuery = useDebouncedValue(query, debounceMs);

  return (
    <div className="flex flex-col gap-4">
      <SearchInput value={query} onChange={setQuery} />
      <CompatFilterChips value={filter} onChange={setFilter} />
      <SearchResults query={debouncedQuery.trim()} filter={filter} />
    </div>
  );
}
