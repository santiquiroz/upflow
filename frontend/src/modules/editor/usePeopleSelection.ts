import { useMemo, useState } from "react";
import type { InitImageResponse } from "../../lib/apiTypes";
import type { EditorPerson } from "../../services/editor";
import { initialChoices, setGroup, toggleChoice, type Choice } from "./peopleState";
import { usePeople } from "./usePeople";

export type PeopleSelectionStatus = "loading" | "ready" | "empty" | "error" | "unavailable";

export function usePeopleSelection(baseInfo: InitImageResponse | null, enabled: boolean) {
  const query = usePeople(baseInfo, enabled);
  const [choiceToken, setChoiceToken] = useState<string | null>(null);
  const [storedChoices, setStoredChoices] = useState<Record<number, Choice>>({});
  const [hoverState, setHoverState] = useState<{ token: string | null; id: number | null }>({
    token: null,
    id: null,
  });
  const [protectState, setProtectState] = useState<{ token: string | null; value: boolean }>({
    token: null,
    value: true,
  });
  const token = baseInfo?.initImageToken ?? null;
  const people = query.data ?? [];
  const hoveredId = hoverState.token === token ? hoverState.id : null;
  const protectKept = protectState.token === token ? protectState.value : true;
  const choices = useMemo(
    () => choiceToken === token ? storedChoices : initialChoices(people),
    [choiceToken, people, storedChoices, token],
  );

  const status: PeopleSelectionStatus = !enabled
    ? "unavailable"
    : query.isPending
      ? "loading"
      : query.isError
        ? "error"
        : people.length === 0
          ? "empty"
          : "ready";

  function toggle(id: number) {
    setStoredChoices(toggleChoice(choices, id));
    setChoiceToken(token);
  }

  function setGroupChoice(role: EditorPerson["role"], choice: Choice) {
    setStoredChoices(setGroup(choices, people, role, choice));
    setChoiceToken(token);
  }

  function setHoveredId(id: number | null) {
    setHoverState({ token, id });
  }

  function setProtectKept(value: boolean) {
    setProtectState({ token, value });
  }

  return {
    people,
    status,
    choices,
    hoveredId,
    setHoveredId,
    toggle,
    setGroupChoice,
    protectKept,
    setProtectKept,
    refetch: query.refetch,
    error: query.error instanceof Error ? query.error.message : undefined,
  };
}
