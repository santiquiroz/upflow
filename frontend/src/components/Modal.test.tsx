import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { Modal } from "./Modal";

function renderModal(onClose = vi.fn()) {
  render(
    <Modal titleId="modal-title" onClose={onClose}>
      <h2 id="modal-title">Title</h2>
      <button type="button">First</button>
      <button type="button">Second</button>
    </Modal>,
  );
  return onClose;
}

describe("Modal", () => {
  it("exposes a labelled modal dialog", () => {
    renderModal();

    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveAttribute("aria-labelledby", "modal-title");
  });

  it("moves initial focus onto the first control inside the dialog", () => {
    renderModal();

    expect(screen.getByRole("button", { name: "First" })).toHaveFocus();
  });

  it("closes when Escape is pressed", () => {
    const onClose = renderModal();

    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("wraps focus back to the first control when Tab is pressed on the last", () => {
    renderModal();
    const first = screen.getByRole("button", { name: "First" });
    const second = screen.getByRole("button", { name: "Second" });

    second.focus();
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Tab" });

    expect(first).toHaveFocus();
  });

  it("wraps focus to the last control on Shift+Tab from the first", () => {
    renderModal();
    const first = screen.getByRole("button", { name: "First" });
    const second = screen.getByRole("button", { name: "Second" });

    first.focus();
    fireEvent.keyDown(screen.getByRole("dialog"), { key: "Tab", shiftKey: true });

    expect(second).toHaveFocus();
  });

  it("restores focus to the triggering element when it unmounts", () => {
    const trigger = document.createElement("button");
    document.body.appendChild(trigger);
    trigger.focus();

    const { unmount } = render(
      <Modal titleId="modal-title" onClose={vi.fn()}>
        <h2 id="modal-title">Title</h2>
        <button type="button">First</button>
      </Modal>,
    );
    unmount();

    expect(trigger).toHaveFocus();
    trigger.remove();
  });
});
