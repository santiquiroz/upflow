import { fireEvent, render, screen, within } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import type { EditorPerson } from "../../services/editor";
import { initialChoices, personNumbers } from "./peopleState";
import { PeopleOverlay, peopleOverlayLabelSize } from "./PeopleOverlay";
import { PeoplePanel, thumbnailStyle } from "./PeoplePanel";

const PEOPLE: EditorPerson[] = Array.from({ length: 10 }, (_, index) => {
  const id = index + 1;
  const x = (id - 1) * 50;
  return {
    id,
    box: [x, 20, x + 40, 180],
    score: 0.9,
    role: id <= 3 ? "subject" : "background",
    face: [x + 5, 25, x + 20, 45],
    anchor: [x + 20, 100],
  };
});

function renderPeople() {
  function ControlledPeople() {
    const [choices, setChoices] = useState(() => initialChoices(PEOPLE));
    const [hoveredId, setHoveredId] = useState<number | null>(null);
    return (
      <div>
        <PeoplePanel
          people={PEOPLE}
          choices={choices}
          hoveredId={hoveredId}
          onHover={setHoveredId}
          onToggle={(id) => setChoices((current) => ({
            ...current,
            [id]: current[id] === "keep" ? "remove" : "keep",
          }))}
          onGroup={(role, choice) =>
            setChoices((current) => Object.fromEntries(
              Object.entries(current).map(([id, currentChoice]) => [
                id,
                PEOPLE.find((person) => person.id === Number(id))?.role === role
                  ? choice
                  : currentChoice,
              ]),
            ))
          }
          protectKept
          onProtectKeptChange={() => undefined}
          onAccept={() => undefined}
          onRetry={() => undefined}
          status="ready"
          imageUrl="blob:photo"
          imageWidth={640}
          imageHeight={480}
        />
        <PeopleOverlay
          people={PEOPLE}
          choices={choices}
          hoveredId={hoveredId}
          onHover={setHoveredId}
          onToggle={(id) => setChoices((current) => ({
            ...current,
            [id]: current[id] === "keep" ? "remove" : "keep",
          }))}
          imageWidth={640}
          imageHeight={480}
        />
      </div>
    );
  }

  return render(
    <LocaleProvider initialLocale="en">
      <ControlledPeople />
    </LocaleProvider>,
  );
}

describe("PeoplePanel", () => {
  it("shows front and back groups with the number of people marked for removal", () => {
    renderPeople();

    expect(screen.getByText("In front · 3")).toBeInTheDocument();
    expect(screen.getByText("In the back · 7")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add 7 silhouettes" })).toBeEnabled();
  });

  it("keeps the overlay in sync with a hovered row and toggles from the overlay", () => {
    renderPeople();

    const row = screen.getByRole("listitem", { name: "Person 4" });
    fireEvent.mouseEnter(row);
    expect(document.querySelector('g[data-person-id="4"]')).toHaveAttribute("data-hovered", "true");

    fireEvent.click(screen.getByTestId("person-outline-4"));
    expect(within(row).getByRole("radio", { name: /Keep/ })).toHaveAttribute("aria-checked", "true");
  });

  it("disables the accept action when Keep all is selected for the back group", () => {
    renderPeople();

    fireEvent.click(screen.getByRole("button", { name: "Keep all in the back" }));
    expect(screen.getByRole("button", { name: "Add 0 silhouettes" })).toBeDisabled();
  });

  it("toggles the focused row with Space", () => {
    renderPeople();
    const row = screen.getByRole("listitem", { name: "Person 1" });

    row.focus();
    fireEvent.keyDown(row, { key: " " });

    expect(within(row).getByRole("radio", { name: /Remove/ })).toHaveAttribute("aria-checked", "true");
  });

  it("moves row focus with the arrow keys", () => {
    renderPeople();
    const first = screen.getByRole("listitem", { name: "Person 1" });
    const second = screen.getByRole("listitem", { name: "Person 2" });
    first.focus();

    fireEvent.keyDown(first, { key: "ArrowDown" });
    expect(second).toHaveFocus();
    expect(second).toHaveAttribute("tabindex", "0");
    expect(first).toHaveAttribute("tabindex", "-1");
  });

  it("highlights a keyboard-focused row and clears the highlight on blur", () => {
    renderPeople();
    const row = screen.getByRole("listitem", { name: `Person ${personNumbers(PEOPLE)[4]}` });

    fireEvent.focus(row);
    expect(document.querySelector('g[data-person-id="4"]')).toHaveAttribute("data-hovered", "true");

    fireEvent.blur(row);
    expect(document.querySelector('g[data-person-id="4"]')).toHaveAttribute("data-hovered", "false");
  });

  it("shows empty and unavailable states", () => {
    const props = {
      people: [],
      choices: {},
      hoveredId: null,
      onHover: () => undefined,
      onToggle: () => undefined,
      onGroup: () => undefined,
      protectKept: true,
      onProtectKeptChange: () => undefined,
      onAccept: () => undefined,
      onRetry: () => undefined,
      imageUrl: "blob:photo",
      imageWidth: 640,
      imageHeight: 480,
    };
    const { rerender } = render(
      <LocaleProvider initialLocale="en">
        <PeoplePanel {...props} status="empty" />
      </LocaleProvider>,
    );

    expect(screen.getByText("I couldn't find people in this photo")).toBeInTheDocument();

    rerender(
      <LocaleProvider initialLocale="en">
        <PeoplePanel {...props} status="unavailable" unavailableReason="People detection is unavailable." />
      </LocaleProvider>,
    );
    expect(screen.getByText("People detection is unavailable.")).toBeInTheDocument();
  });

  it("shows skeletons while loading and retries after an error", () => {
    const onRetry = vi.fn();
    const props = {
      people: [],
      choices: {},
      hoveredId: null,
      onHover: () => undefined,
      onToggle: () => undefined,
      onGroup: () => undefined,
      protectKept: true,
      onProtectKeptChange: () => undefined,
      onAccept: () => undefined,
      onRetry,
      imageUrl: "blob:photo",
      imageWidth: 640,
      imageHeight: 480,
    };
    const { rerender } = render(
      <LocaleProvider initialLocale="en">
        <PeoplePanel {...props} status="loading" />
      </LocaleProvider>,
    );
    expect(screen.getByLabelText("Detecting people").querySelectorAll(".animate-pulse")).toHaveLength(5);

    rerender(
      <LocaleProvider initialLocale="en">
        <PeoplePanel {...props} status="error" error="Detection failed." />
      </LocaleProvider>,
    );
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(screen.getByText("Detection failed.")).toBeInTheDocument();
    expect(onRetry).toHaveBeenCalledOnce();
  });

  it("computes exact face and upper-quarter thumbnail crops", () => {
    expect(thumbnailStyle(PEOPLE[0], 640, 480, 36)).toEqual({
      backgroundPosition: "0px -21.375px",
      backgroundSize: "720px 540px",
    });
    const noFace = { ...PEOPLE[0], face: null };
    expect(thumbnailStyle(noFace, 640, 480, 36)).toEqual({
      backgroundPosition: "0px -18px",
      backgroundSize: "576px 432px",
    });
  });

  it("clamps the thumbnail crop when its face box touches the image edge", () => {
    const atEdge: EditorPerson = { ...PEOPLE[0], face: [630, 460, 640, 480] };
    expect(thumbnailStyle(atEdge, 640, 480, 36)).toEqual({
      backgroundPosition: "-684px -504px",
      backgroundSize: "720px 540px",
    });
  });

  it("scales the overlay label dimensions with the image width", () => {
    expect(peopleOverlayLabelSize(640)).toEqual({
      fontSize: 10,
      width: 74,
      height: 17,
      padding: 6,
      radius: 8,
    });
    expect(peopleOverlayLabelSize(4000)).toEqual({
      fontSize: 50,
      width: 370,
      height: 85,
      padding: 30,
      radius: 40,
    });
  });
});
