import { segmentEditorBox, type EditorPerson } from "../../services/editor";
import { createMaskCanvas, type CropMask, type HistoryEntry, type PixelRect } from "./editorMaskCanvas";
import { anchorPoint, type Choice } from "./peopleState";

type SilhouetteKind = "segment" | "protect";

const CROP_MARGIN_PX = 8;

type Box = [number, number, number, number];

export interface SilhouetteOptions {
  toMask: (blob: Blob, box: Box) => Promise<CropMask>;
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

interface PersonCrop {
  crop: CropMask;
  target: Target;
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

export function cropRect(box: Box, width: number, height: number): PixelRect {
  const left = Math.max(0, Math.floor(box[0] - CROP_MARGIN_PX));
  const top = Math.max(0, Math.floor(box[1] - CROP_MARGIN_PX));
  const right = Math.min(width, Math.ceil(box[2] + CROP_MARGIN_PX));
  const bottom = Math.min(height, Math.ceil(box[3] + CROP_MARGIN_PX));
  return { x: left, y: top, width: Math.max(0, right - left), height: Math.max(0, bottom - top) };
}

async function silhouetteFor(
  imageToken: string,
  { person }: Target,
  toMask: SilhouetteOptions["toMask"],
): Promise<CropMask | null> {
  try {
    return await toMask(await segmentEditorBox(imageToken, person.box), person.box);
  } catch {
    return null;
  }
}

// Une los recortes en un solo lienzo cuyo tamaño es la unión de sus cajas, no la foto entera.
export function mergeCrops(crops: CropMask[]): CropMask | null {
  if (crops.length === 0) return null;
  const left = Math.min(...crops.map((crop) => crop.origin.x));
  const top = Math.min(...crops.map((crop) => crop.origin.y));
  const right = Math.max(...crops.map((crop) => crop.origin.x + crop.mask.width));
  const bottom = Math.max(...crops.map((crop) => crop.origin.y + crop.mask.height));
  const mask = createMaskCanvas(right - left, bottom - top);
  const ctx = mask.getContext("2d");
  if (!ctx) throw new Error("canvas 2d context unavailable");
  for (const crop of crops) {
    ctx.drawImage(crop.mask, crop.origin.x - left, crop.origin.y - top);
  }
  return { mask, origin: { x: left, y: top } };
}

function batchEntries(segments: CropMask[], protections: PersonCrop[]): HistoryEntry[] {
  const segment = mergeCrops(segments);
  const protect = mergeCrops(protections.map((item) => item.crop));
  const entries: HistoryEntry[] = [];
  if (segment) entries.push({ kind: "segment", mask: segment.mask, origin: segment.origin });
  if (protect) {
    entries.push({
      kind: "protect",
      mask: protect.mask,
      origin: protect.origin,
      points: protections.map((item) => anchorPoint(item.target.person)),
    });
  }
  return entries;
}

export async function silhouettesFor(
  imageToken: string,
  people: EditorPerson[],
  choices: Record<number, Choice>,
  protectKept: boolean,
  { toMask, onProgress }: SilhouetteOptions,
): Promise<SilhouetteResult> {
  const targets = targetsFor(people, choices, protectKept);
  const segments: CropMask[] = [];
  const protections: PersonCrop[] = [];
  const personIds: number[] = [];
  for (const [index, target] of targets.entries()) {
    const crop = await silhouetteFor(imageToken, target, toMask);
    if (crop && target.kind === "segment") {
      segments.push(crop);
      personIds.push(target.person.id);
    } else if (crop) {
      protections.push({ crop, target });
    }
    onProgress?.(index + 1, targets.length);
  }
  const failed = targets.length - segments.length - protections.length;
  const entry: HistoryEntry = { kind: "batch", entries: batchEntries(segments, protections), personIds, failed };
  return { entry, failed };
}

export function lastPeopleBatch(history: HistoryEntry[]): HistoryEntry | null {
  return [...history].reverse().find((entry) => entry.kind === "batch" && hasPeopleNotice(entry)) ?? null;
}

function hasPeopleNotice(entry: HistoryEntry): boolean {
  return Boolean(entry.personIds?.length) || (entry.failed ?? 0) > 0;
}
