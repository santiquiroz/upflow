import { segmentEditorBox, type EditorPerson } from "../../services/editor";
import type { HistoryEntry } from "./editorMaskCanvas";
import { anchorPoint, type Choice } from "./peopleState";

type SilhouetteKind = "segment" | "protect";

export interface SilhouetteOptions {
  toMask: (blob: Blob) => Promise<HTMLCanvasElement>;
  onProgress?: (done: number, total: number) => void;
}

export interface SilhouetteResult {
  entry: HistoryEntry;
  failed: number;
}

interface Target {
  person: EditorPerson;
  kind: SilhouetteKind;
}

function kindFor(choice: Choice, protectKept: boolean): SilhouetteKind | null {
  if (choice === "remove") return "segment";
  return protectKept ? "protect" : null;
}

function targetsFor(people: EditorPerson[], choices: Record<number, Choice>, protectKept: boolean): Target[] {
  return people.flatMap((person) => {
    const kind = kindFor(choices[person.id] ?? "remove", protectKept);
    return kind ? [{ person, kind }] : [];
  });
}

export function silhouetteCount(people: EditorPerson[], choices: Record<number, Choice>, protectKept: boolean): number {
  return targetsFor(people, choices, protectKept).length;
}

async function silhouetteFor(
  imageToken: string,
  { person, kind }: Target,
  toMask: SilhouetteOptions["toMask"],
): Promise<HistoryEntry | null> {
  try {
    const mask = await toMask(await segmentEditorBox(imageToken, person.box));
    return { kind, mask, point: anchorPoint(person) };
  } catch {
    return null;
  }
}

export async function silhouettesFor(
  imageToken: string,
  people: EditorPerson[],
  choices: Record<number, Choice>,
  protectKept: boolean,
  { toMask, onProgress }: SilhouetteOptions,
): Promise<SilhouetteResult> {
  const targets = targetsFor(people, choices, protectKept);
  const entries: HistoryEntry[] = [];
  const personIds: number[] = [];
  for (const [index, target] of targets.entries()) {
    const entry = await silhouetteFor(imageToken, target, toMask);
    if (entry) entries.push(entry);
    if (entry?.kind === "segment") personIds.push(target.person.id);
    onProgress?.(index + 1, targets.length);
  }
  return { entry: { kind: "batch", entries, personIds }, failed: targets.length - entries.length };
}

export function lastPeopleBatch(history: HistoryEntry[]): HistoryEntry | null {
  return [...history].reverse().find((entry) => entry.kind === "batch" && Boolean(entry.personIds?.length)) ?? null;
}
