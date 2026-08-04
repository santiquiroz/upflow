import { useState } from "react";
import { useTranslation } from "../../i18n/LocaleProvider";
import { useDebouncedValue } from "../../hooks/useDebouncedValue";
import { useHfSearchResults } from "../../hooks/useModels";
import { CompatFilterChips, matchesCompatFilter, type CompatFilter } from "./compatFilter";
import { HfResultCard } from "./HfResultCard";
import {
  DEFAULT_SEARCH_DEBOUNCE_MS,
  NoResultsState,
  SearchEmptyState,
  SearchErrorState,
  SearchInput,
  SearchLoadingState,
} from "./hfSearchUi";

interface HfSearchProps {
  debounceMs?: number;
}

function SearchResults({ query, filter }: { query: string; filter: CompatFilter }) {
  const searchQuery = useHfSearchResults(query);

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
          <HfResultCard result={result} />
        </li>
      ))}
    </ul>
  );
}

export function HfSearch({ debounceMs = DEFAULT_SEARCH_DEBOUNCE_MS }: HfSearchProps) {
  const { t } = useTranslation();
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<CompatFilter>("all");
  const debouncedQuery = useDebouncedValue(query, debounceMs);
  const trimmedQuery = debouncedQuery.trim();

  return (
    <div className="flex flex-col gap-4">
      <SearchInput value={query} onChange={setQuery} />
      {trimmedQuery.length > 0 && (
        <CompatFilterChips value={filter} onChange={setFilter} name="upscaler-compat-filter" />
      )}
      {trimmedQuery.length === 0 ? (
        <SearchEmptyState message={t("models.hfSearch.empty")} />
      ) : (
        <SearchResults query={trimmedQuery} filter={filter} />
      )}
    </div>
  );
}
