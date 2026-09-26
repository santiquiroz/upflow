import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { en } from "../../../i18n/en";
import { CctvSuggestBanner } from "./CctvSuggestBanner";

describe("CctvSuggestBanner", () => {
  it("offers CCTV mode and switches on accept", () => {
    const onAccept = vi.fn();
    render(<CctvSuggestBanner onAccept={onAccept} />);

    expect(screen.getByRole("status")).toHaveTextContent(en["cctv.suggestMode"]);
    fireEvent.click(screen.getByRole("button", { name: en["cctv.suggest.accept"] }));

    expect(onAccept).toHaveBeenCalledTimes(1);
  });

  it("goes away when dismissed without switching", () => {
    const onAccept = vi.fn();
    render(<CctvSuggestBanner onAccept={onAccept} />);

    fireEvent.click(screen.getByRole("button", { name: en["cctv.suggest.dismiss"] }));

    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(onAccept).not.toHaveBeenCalled();
  });
});
