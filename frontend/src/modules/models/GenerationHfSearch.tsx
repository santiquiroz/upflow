import { useState } from "react";
import { useGenerationHfSearchResults } from "../../hooks/useGenerationJob";
import { useDebouncedValue } from "../../hooks/useDebouncedValue";
import { GenerationModelCard } from "./GenerationModelCard";
import { CompatFilterChips, matchesCompatFilter, type CompatFilter } from "./compatFilter";
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

function SearchResults({ query, filter }: { query: string; filter: CompatFilter }) {
  const searchQuery = useGenerationHfSearchResults(query);

  if (searchQuery.isLoading) {
    return <SearchLoadingState />;
  }

  if (searchQuery.isError) {
    return <SearchErrorState />;
  }

  const results = (searchQuery.data?.results ?? []).filter((result) =>
    matchesCompatFilter(result, filter),
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
      <CompatFilterChips value={filter} onChange={setFilter} name="generation-compat-filter" />
      <SearchResults query={debouncedQuery.trim()} filter={filter} />
    </div>
  );
}
