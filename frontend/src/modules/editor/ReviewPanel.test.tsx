import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import type { EraseZone } from "../../lib/apiTypes";
import type { EditorPerson } from "../../services/editor";
import { ReviewPanel } from "./ReviewPanel";

const PEOPLE: EditorPerson[] = [
  { id: 7, box: [10, 10, 60, 90], score: 0.9, role: "background", face: null, anchor: [30, 50] },
];

function zone(overrides: Partial<EraseZone> = {}): EraseZone {
  return { index: 0, box: [0, 0, 100, 100], engine: "lama", guard: null, residualPerson: false, ...overrides };
}

function renderPanel(zones: EraseZone[], overrides: Partial<Parameters<typeof ReviewPanel>[0]> = {}) {
  const props = {
    zones,
    people: PEOPLE,
    choices: { 7: "remove" as const },
    hoveredZone: null,
    onHoverZone: vi.fn(),
    onRedo: vi.fn(),
    sdxlAvailable: true,
    ...overrides,
  };
  render(
    <LocaleProvider initialLocale="en">
      <ReviewPanel {...props} />
    </LocaleProvider>,
  );
  return props;
}

describe("ReviewPanel", () => {
  it("shows one row per zone with number, people, engine and a clean status", () => {
    renderPanel([zone(), zone({ index: 1, box: [200, 200, 300, 300], engine: "lama+osor" })]);

    const first = screen.getByRole("listitem", { name: "Zone 1" });
    expect(within(first).getByText("Person 1")).toBeInTheDocument();
    expect(within(first).getByText("LaMa")).toBeInTheDocument();
    expect(within(first).getByText("No people")).toBeInTheDocument();
    const second = screen.getByRole("listitem", { name: "Zone 2" });
    expect(within(second).getByText("Zone 2")).toBeInTheDocument();
    expect(within(second).getByText("LaMa + OSOR")).toBeInTheDocument();
  });

  it("explains a zone the guard redid on its own", () => {
    renderPanel([zone({ guard: "person", residualPerson: false })]);

    expect(screen.getByText("Saw a person: redid it on its own")).toBeInTheDocument();
  });

  it("warns when someone is still visible", () => {
    renderPanel([zone({ guard: "person", residualPerson: true })]);

    expect(screen.getByText("Someone is still visible")).toBeInTheDocument();
  });

  it("redoes a zone with LaMa", () => {
    const props = renderPanel([zone({ index: 3 })]);

    fireEvent.click(screen.getByRole("button", { name: "Redo" }));
    fireEvent.click(screen.getByRole("menuitem", { name: "LaMa" }));

    expect(props.onRedo).toHaveBeenCalledWith(expect.objectContaining({ index: 3 }), "lama");
  });

  it("describes the background with the typed text", () => {
    const props = renderPanel([zone()]);

    fireEvent.click(screen.getByRole("button", { name: "Redo" }));
    fireEvent.click(screen.getByRole("menuitem", { name: "Describe the background…" }));
    fireEvent.change(screen.getByLabelText("What should be there"), { target: { value: " a brick wall " } });
    fireEvent.click(screen.getByRole("button", { name: "Redo with this description" }));

    expect(props.onRedo).toHaveBeenCalledWith(expect.objectContaining({ index: 0 }), "sdxl", "a brick wall");
  });

  it("disables describing the background without SDXL and says why", () => {
    renderPanel([zone()], { sdxlAvailable: false });

    fireEvent.click(screen.getByRole("button", { name: "Redo" }));

    expect(screen.getByRole("menuitem", { name: "Describe the background…" })).toBeDisabled();
    expect(screen.getByText("Needs a fill model (SDXL)")).toBeVisible();
    expect(screen.getByRole("menuitem", { name: "LaMa" })).toBeEnabled();
  });

  it("disables the eraser engines without eraser-auto and says why", () => {
    renderPanel([zone()], { eraserAvailable: false });

    fireEvent.click(screen.getByRole("button", { name: "Redo" }));

    expect(screen.getByRole("menuitem", { name: "OSOR" })).toBeDisabled();
    expect(screen.getByRole("menuitem", { name: "LaMa + OSOR" })).toBeDisabled();
    expect(screen.getByText("Needs the automatic eraser (LaMa + OSOR)")).toBeVisible();
  });

  it("disables Redo for a zone without a location", () => {
    renderPanel([zone({ box: null })]);

    expect(screen.getByRole("button", { name: "Redo" })).toBeDisabled();
    expect(screen.getByText("Zone without a location")).toBeVisible();
  });

  it("shows progress on the redoing row and blocks the others", () => {
    renderPanel([zone(), zone({ index: 1 })], { redoingIndex: 1, errors: { 0: "boom" } });

    const first = screen.getByRole("listitem", { name: "Zone 1" });
    const second = screen.getByRole("listitem", { name: "Zone 2" });
    expect(within(first).getByRole("button", { name: "Redo" })).toBeDisabled();
    expect(within(first).getByRole("alert")).toHaveTextContent("boom");
    expect(within(second).getByRole("button", { name: "Redoing…" })).toBeDisabled();
  });

  it("reports hover and shows before and after for the hovered zone", () => {
    const props = renderPanel([zone()], {
      hoveredZone: 0,
      beforeUrl: "blob:before",
      afterUrl: "/result.png",
      imageSize: { width: 200, height: 200 },
    });

    const row = screen.getByRole("listitem", { name: "Zone 1" });
    fireEvent.mouseEnter(row);
    expect(props.onHoverZone).toHaveBeenCalledWith(0);
    expect(within(row).getByRole("img", { name: "Before" }).style.backgroundImage).toContain("blob:before");
    expect(within(row).getByRole("img", { name: "After" }).style.backgroundImage).toContain("/result.png");
  });
});
