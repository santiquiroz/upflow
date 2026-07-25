import { useState } from "react";
import {
  useGenerationHfSearchResults,
  useGenerationModelInstall,
} from "../../hooks/useGenerationJob";
import { useDebouncedValue } from "../../hooks/useDebouncedValue";
import type { HfModelSearchResultResponse } from "../../lib/apiTypes";
import { ResultCardLayout } from "./HfResultCard";
import {
  DEFAULT_SEARCH_DEBOUNCE_MS,
  NoResultsState,
  SearchEmptyState,
  SearchErrorState,
  SearchInput,
  SearchLoadingState,
} from "./hfSearchUi";

interface GenerationHfSearchProps {
  debounceMs?: number;
}

function GenerationResultCard({ result }: { result: HfModelSearchResultResponse }) {
  const { phase, progressPct, stageLabel, errorMessage, install, reset } =
    useGenerationModelInstall();

  return (
    <ResultCardLayout
      result={result}
      phase={phase}
      progressPct={progressPct}
      stageLabel={stageLabel}
      errorMessage={errorMessage}
      onInstall={() => install(result.id)}
      onReset={reset}
    />
  );
}

function SearchResults({ query }: { query: string }) {
  const searchQuery = useGenerationHfSearchResults(query);

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
          <GenerationResultCard result={result} />
        </li>
      ))}
    </ul>
  );
}

export function GenerationHfSearch({
  debounceMs = DEFAULT_SEARCH_DEBOUNCE_MS,
}: GenerationHfSearchProps) {
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebouncedValue(query, debounceMs);
  const trimmedQuery = debouncedQuery.trim();

  return (
    <div className="flex flex-col gap-4">
      <SearchInput value={query} onChange={setQuery} />
      {trimmedQuery.length === 0 ? (
        <SearchEmptyState message="Search Hugging Face for a Stable Diffusion (text-to-image) pipeline to install." />
      ) : (
        <SearchResults query={trimmedQuery} />
      )}
    </div>
  );
}
