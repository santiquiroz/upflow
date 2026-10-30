import { describe, expect, it } from "vitest";

import type { EditorPerson } from "../../services/editor";
import {
  anchorPoint,
  countRemove,
  groupPeople,
  initialChoices,
  setGroup,
  toggleChoice,
  type Choice,
} from "./peopleState";

const PERSONS: EditorPerson[] = [
  { id: 1, box: [0, 0, 10, 20], score: 0.9, role: "background", face: null, anchor: [5, 4] },
  { id: 2, box: [10, 0, 20, 20], score: 0.8, role: "subject", face: [13, 2, 17, 7], anchor: [15, 4] },
  { id: 3, box: [20, 0, 30, 20], score: 0.7, role: "background", face: null, anchor: [25, 4] },
];

describe("initialChoices", () => {
  it("keeps subjects and removes background people without mutating its input", () => {
    const result = initialChoices(PERSONS);

    expect(result).toEqual({ 1: "remove", 2: "keep", 3: "remove" });
    expect(PERSONS[0].role).toBe("background");
  });
});

describe("toggleChoice", () => {
  it("returns a new choices object and toggles the selected person", () => {
    const original: Record<number, Choice> = { 1: "keep", 2: "remove" };

    const result = toggleChoice(original, 1);

    expect(result).toEqual({ 1: "remove", 2: "remove" });
    expect(result).not.toBe(original);
    expect(original[1]).toBe("keep");
  });
});

describe("setGroup", () => {
  it("sets only people of the requested role without mutating the inputs", () => {
    const original: Record<number, Choice> = { 1: "keep", 2: "remove", 3: "keep" };

    const result = setGroup(original, PERSONS, "background", "remove");

    expect(result).toEqual({ 1: "remove", 2: "remove", 3: "remove" });
    expect(result).not.toBe(original);
    expect(original).toEqual({ 1: "keep", 2: "remove", 3: "keep" });
  });
});

describe("countRemove", () => {
  it("counts choices marked for removal", () => {
    expect(countRemove({ 1: "keep", 2: "remove", 3: "remove" })).toBe(2);
  });
});

describe("groupPeople", () => {
  it("groups subjects and background people in their original relative order", () => {
    expect(groupPeople(PERSONS)).toEqual({
      front: [PERSONS[1]],
      back: [PERSONS[0], PERSONS[2]],
    });
  });
});

describe("anchorPoint", () => {
  it("returns the backend-provided anchor as x and y", () => {
    expect(anchorPoint(PERSONS[1])).toEqual({ x: 15, y: 4 });
  });
});
