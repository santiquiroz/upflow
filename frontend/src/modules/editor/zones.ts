import type { CSSProperties } from "react";
import type { EraseZone, GenerationJob } from "../../lib/apiTypes";
import type { EditorPerson } from "../../services/editor";
import { personNumbers, type Choice } from "./peopleState";

export type Box = [number, number, number, number];
export type ZoneStatus = "clean" | "retried" | "residual";
export type RedoEngine = "lama" | "osor" | "lama+osor" | "sdxl";

type Translate = (key: string) => string;

const ENGINE_LABELS: Record<string, string> = {
  lama: "LaMa",
  osor: "OSOR",
  "lama+osor": "LaMa + OSOR",
  sdxl: "SDXL",
};

export function engineLabel(engine: string, t: Translate): string {
  if (engine === "hibrido") return t("editor.review.engine.hybrid");
  return ENGINE_LABELS[engine] ?? engine;
}

export function zoneStatus(zone: EraseZone): ZoneStatus {
  if (zone.residualPerson) return "residual";
  return zone.guard !== null ? "retried" : "clean";
}

function boxArea([x1, y1, x2, y2]: Box): number {
  return Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
}

function intersection(a: Box, b: Box): Box {
  return [Math.max(a[0], b[0]), Math.max(a[1], b[1]), Math.min(a[2], b[2]), Math.min(a[3], b[3])];
}

export function mostlyInside(person: Box, zone: Box): boolean {
  const area = boxArea(person);
  return area > 0 && boxArea(intersection(person, zone)) >= area / 2;
}

export function zonePeople(
  zone: EraseZone,
  people: EditorPerson[],
  choices: Record<number, Choice>,
  format: (number: number) => string = (number) => `Persona ${number}`,
): string[] {
  const box = zone.box;
  if (!box) return [];
  const numbers = personNumbers(people);
  return people
    .filter((person) => choices[person.id] === "remove" && mostlyInside(person.box, box))
    .map((person) => numbers[person.id])
    .sort((a, b) => a - b)
    .map(format);
}

function makeImageData(data: Uint8ClampedArray, width: number, height: number): ImageData {
  if (typeof ImageData === "function") return new ImageData(data as Uint8ClampedArray<ArrayBuffer>, width, height);
  return { data, width, height, colorSpace: "srgb" } as ImageData;
}

function insideBox(x: number, y: number, [x1, y1, x2, y2]: Box): boolean {
  return x >= Math.floor(x1) && x < Math.ceil(x2) && y >= Math.floor(y1) && y < Math.ceil(y2);
}

export function clipMaskToBox(mask: ImageData, box: Box): ImageData {
  const data = new Uint8ClampedArray(new ArrayBuffer(mask.data.length));
  data.set(mask.data);
  for (let index = 3; index < data.length; index += 4) {
    const pixel = (index - 3) / 4;
    if (!insideBox(pixel % mask.width, Math.floor(pixel / mask.width), box)) data[index] = 0;
  }
  return makeImageData(data, mask.width, mask.height);
}

export function redoneZone(zone: EraseZone, engine: RedoEngine, job: GenerationJob): EraseZone {
  if (engine === "sdxl") return { ...zone, engine: "sdxl", guard: null, residualPerson: false };
  const first = job.eraseZones?.[0];
  if (!first) return { ...zone, engine };
  return { ...zone, engine: first.engine, guard: first.guard, residualPerson: first.residualPerson };
}

export function replaceZone(zones: EraseZone[], next: EraseZone): EraseZone[] {
  return zones.map((zone) => (zone.index === next.index ? next : zone));
}

export function cropStyle(
  url: string,
  box: Box,
  image: { width: number; height: number },
  size: number,
): CSSProperties {
  const [x1, y1, x2, y2] = box;
  const scale = size / Math.max(x2 - x1, y2 - y1, 1);
  return {
    width: (x2 - x1) * scale,
    height: (y2 - y1) * scale,
    backgroundImage: `url("${url}")`,
    backgroundRepeat: "no-repeat",
    backgroundSize: `${image.width * scale}px ${image.height * scale}px`,
    backgroundPosition: `-${x1 * scale}px -${y1 * scale}px`,
  };
}
