import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import { LocaleProvider } from "../../i18n/LocaleProvider";
import { LanguageSection } from "./LanguageSection";

function renderSection() {
  return render(
    <LocaleProvider>
      <LanguageSection />
    </LocaleProvider>,
  );
}

beforeEach(() => {
  // La suite corre con el locale fijado en inglés (ver vitest.setup.ts): cada
  // test parte de ahí para que el cambio sea observable.
  localStorage.setItem("upflow.locale", "en");
});

describe("LanguageSection", () => {
  it("names each language in its own language, not translated", () => {
    // Quien busca su idioma en una pantalla que no entiende reconoce
    // "Español", no "Spanish".
    renderSection();
    expect(screen.getByRole("radio", { name: "Español" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "English" })).toBeInTheDocument();
  });

  it("marks the active language", () => {
    renderSection();
    expect(screen.getByRole("radio", { name: "English" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "Español" })).not.toBeChecked();
  });

  it("switches the active language on click", () => {
    renderSection();
    fireEvent.click(screen.getByRole("radio", { name: "Español" }));
    expect(screen.getByRole("radio", { name: "Español" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "English" })).not.toBeChecked();
  });

  it("remembers the choice on this device", () => {
    // Sin persistencia el idioma se pierde al recargar, que es el caso normal
    // de uso: se elige una vez.
    renderSection();
    fireEvent.click(screen.getByRole("radio", { name: "Español" }));
    expect(localStorage.getItem("upflow.locale")).toBe("es");
  });

  it("translates the surrounding copy into the language just picked", () => {
    renderSection();
    const titleInEnglish = screen.getByRole("radiogroup").getAttribute("aria-label");

    fireEvent.click(screen.getByRole("radio", { name: "Español" }));

    expect(screen.getByRole("radiogroup").getAttribute("aria-label")).not.toBe(
      titleInEnglish,
    );
  });
});
