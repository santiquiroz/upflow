import type { EditorPerson } from "../../services/editor";

export type Choice = "keep" | "remove";

export function initialChoices(people: EditorPerson[]): Record<number, Choice> {
  return Object.fromEntries(
    people.map((person) => [person.id, person.role === "subject" ? "keep" : "remove"]),
  );
}

export function toggleChoice(choices: Record<number, Choice>, id: number): Record<number, Choice> {
  return { ...choices, [id]: choices[id] === "keep" ? "remove" : "keep" };
}

export function setGroup(
  choices: Record<number, Choice>,
  people: EditorPerson[],
  role: EditorPerson["role"],
  choice: Choice,
): Record<number, Choice> {
  const next = { ...choices };
  for (const person of people) {
    if (person.role === role) {
      next[person.id] = choice;
    }
  }
  return next;
}

export function countRemove(choices: Record<number, Choice>): number {
  return Object.values(choices).filter((choice) => choice === "remove").length;
}

export function groupPeople(people: EditorPerson[]): { front: EditorPerson[]; back: EditorPerson[] } {
  return {
    front: people.filter((person) => person.role === "subject"),
    back: people.filter((person) => person.role === "background"),
  };
}

export function personNumbers(people: EditorPerson[]): Record<number, number> {
  const { front, back } = groupPeople(people);
  return Object.fromEntries([...front, ...back].map((person, index) => [person.id, index + 1]));
}

export function anchorPoint(person: EditorPerson): { x: number; y: number } {
  const [x, y] = person.anchor;
  return { x, y };
}
