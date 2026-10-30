import { describe, expect, it } from "vitest";
import type { EraseZone, GenerationJob } from "../../lib/apiTypes";
import type { EditorPerson } from "../../services/editor";
import {
  clipMaskToBox,
  cropStyle,
  engineLabel,
  mostlyInside,
  redoneZone,
  replaceZone,
  zonePeople,
  zoneStatus,
} from "./zones";

function zone(overrides: Partial<EraseZone> = {}): EraseZone {
  return { index: 0, box: [0, 0, 100, 100], engine: "lama", guard: null, residualPerson: false, ...overrides };
}

function person(id: number, box: [number, number, number, number], role: EditorPerson["role"] = "background"): EditorPerson {
  return { id, box, score: 0.9, role, face: null, anchor: [box[0], box[1]] };
}

function mask(width: number, height: number): ImageData {
  return { data: new Uint8ClampedArray(width * height * 4).fill(255), width, height, colorSpace: "srgb" } as ImageData;
}

function alphaAt(image: ImageData, x: number, y: number): number {
  return image.data[(y * image.width + x) * 4 + 3];
}

describe("engineLabel", () => {
  const t = (key: string) => (key === "editor.review.engine.hybrid" ? "Híbrido" : key);

  it.each([
    ["lama", "LaMa"],
    ["osor", "OSOR"],
    ["lama+osor", "LaMa + OSOR"],
    ["hibrido", "Híbrido"],
    ["sdxl", "SDXL"],
    ["mystery", "mystery"],
  ])("%s → %s", (engine, label) => {
    expect(engineLabel(engine, t)).toBe(label);
  });
});

describe("zoneStatus", () => {
  it("is clean without guard or residual person", () => {
    expect(zoneStatus(zone())).toBe("clean");
  });

  it("is retried when a guard fired", () => {
    expect(zoneStatus(zone({ guard: "person" }))).toBe("retried");
    expect(zoneStatus(zone({ guard: "face" }))).toBe("retried");
  });

  it("is residual when someone is still visible, even after a guard", () => {
    expect(zoneStatus(zone({ guard: "person", residualPerson: true }))).toBe("residual");
  });
});

describe("mostlyInside", () => {
  it("accepts exactly half of the person box", () => {
    expect(mostlyInside([50, 0, 150, 100], [0, 0, 100, 100])).toBe(true);
  });

  it("rejects less than half", () => {
    expect(mostlyInside([51, 0, 151, 100], [0, 0, 100, 100])).toBe(false);
  });

  it("rejects an empty person box", () => {
    expect(mostlyInside([10, 10, 10, 10], [0, 0, 100, 100])).toBe(false);
  });
});

describe("zonePeople", () => {
  const people = [
    person(1, [0, 0, 40, 40], "subject"),
    person(2, [10, 10, 60, 60]),
    person(3, [80, 0, 180, 100]),
    person(4, [20, 20, 50, 50]),
  ];

  const format = (number: number) => `Person ${number}`;

  it("lists the people to remove that fall at least half inside the zone, by person number", () => {
    const choices = { 1: "keep", 2: "remove", 3: "remove", 4: "remove" } as const;
    expect(zonePeople(zone(), people, choices, format)).toEqual(["Person 2", "Person 4"]);
  });

  it("formats each name with the given formatter", () => {
    const choices = { 2: "remove" } as const;
    expect(zonePeople(zone(), people, choices, (number) => `Persona ${number}`)).toEqual(["Persona 2"]);
  });

  it("returns nobody for a zone without a box", () => {
    expect(zonePeople(zone({ box: null }), people, { 2: "remove" }, format)).toEqual([]);
  });
});

describe("clipMaskToBox", () => {
  it("clears alpha outside the box and keeps it inside", () => {
    const clipped = clipMaskToBox(mask(4, 4), [1, 1, 3, 3]);
    expect(alphaAt(clipped, 0, 0)).toBe(0);
    expect(alphaAt(clipped, 3, 3)).toBe(0);
    expect(alphaAt(clipped, 1, 1)).toBe(255);
    expect(alphaAt(clipped, 2, 2)).toBe(255);
    expect(alphaAt(clipped, 1, 3)).toBe(0);
    expect(clipped.width).toBe(4);
    expect(clipped.height).toBe(4);
  });

  it("does not mutate the input", () => {
    const source = mask(4, 4);
    clipMaskToBox(source, [1, 1, 2, 2]);
    expect(Array.from(source.data).every((value) => value === 255)).toBe(true);
  });
});

describe("redoneZone", () => {
  const job = { eraseZones: [zone({ index: 0, engine: "osor", guard: "person", residualPerson: true })] } as GenerationJob;

  it("takes engine and guard from the first zone of the redo job", () => {
    expect(redoneZone(zone({ index: 2, box: [5, 5, 9, 9] }), "osor", job)).toEqual(
      zone({ index: 2, box: [5, 5, 9, 9], engine: "osor", guard: "person", residualPerson: true }),
    );
  });

  it("marks an SDXL redo as clean", () => {
    expect(redoneZone(zone({ guard: "person", residualPerson: true }), "sdxl", job)).toEqual(zone({ engine: "sdxl" }));
  });

  it("falls back to the requested engine without zones", () => {
    expect(redoneZone(zone(), "lama+osor", {} as GenerationJob).engine).toBe("lama+osor");
  });
});

describe("replaceZone", () => {
  it("replaces only the row with the same index", () => {
    const zones = [zone({ index: 0 }), zone({ index: 1, engine: "osor" })];
    expect(replaceZone(zones, zone({ index: 0, engine: "sdxl" }))).toEqual([
      zone({ index: 0, engine: "sdxl" }),
      zone({ index: 1, engine: "osor" }),
    ]);
  });
});

describe("cropStyle", () => {
  it("scales the image so the box fills the thumbnail", () => {
    expect(cropStyle("blob:x", [10, 20, 60, 45], { width: 200, height: 100 }, 100)).toEqual({
      width: 100,
      height: 50,
      backgroundImage: 'url("blob:x")',
      backgroundRepeat: "no-repeat",
      backgroundSize: "400px 200px",
      backgroundPosition: "-20px -40px",
    });
  });
});
