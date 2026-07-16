import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RealtimePage } from "./RealtimePage";

describe("RealtimePage", () => {
  it("renders without crashing", () => {
    render(<RealtimePage />);

    expect(screen.getByRole("heading", { level: 1, name: "Realtime" })).toBeInTheDocument();
  });

  it("is honest that the module is not functional yet", () => {
    render(<RealtimePage />);

    expect(screen.getByText(/próximamente/i)).toBeInTheDocument();
    expect(screen.getByText(/aún no lanza, configura ni controla ningún proceso/i)).toBeInTheDocument();
  });

  it("links to the module's design roadmap in a new tab", () => {
    render(<RealtimePage />);

    const link = screen.getByRole("link", { name: /visión completa del módulo/i });
    expect(link).toHaveAttribute("href", expect.stringContaining("REALTIME_MODULE.md"));
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", expect.stringContaining("noreferrer"));
  });

  it("summarizes the MVP scope and what is not viable yet", () => {
    render(<RealtimePage />);

    expect(screen.getByText(/MVP \(Fase 7\.1\)/i)).toBeInTheDocument();
    expect(screen.getByText(/No viable todavía/i)).toBeInTheDocument();
    expect(screen.getByText(/Fork\/vendor de Magpie/i)).toBeInTheDocument();
  });

  it("does not offer any launch, configure or stop controls", () => {
    render(<RealtimePage />);

    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });
});
