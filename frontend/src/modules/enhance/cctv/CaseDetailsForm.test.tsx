import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { en } from "../../../i18n/en";
import { CaseDetailsForm } from "./CaseDetailsForm";
import { EMPTY_CASE_DETAILS, LONG_TEXT_MAX, SHORT_TEXT_MAX, type CaseDetails } from "./cctvCase";

function Harness({ onChange }: { onChange: (next: CaseDetails) => void }) {
  const [details, setDetails] = useState<CaseDetails>(EMPTY_CASE_DETAILS);
  function change(next: CaseDetails): void {
    setDetails(next);
    onChange(next);
  }
  return <CaseDetailsForm value={details} onChange={change} />;
}

function renderForm() {
  const onChange = vi.fn();
  render(<Harness onChange={onChange} />);
  return onChange;
}

describe("CaseDetailsForm", () => {
  it("is folded by default and marked optional", () => {
    renderForm();

    const summary = screen.getByText(en["cctv.case.legend"]);
    expect(summary.closest("details")).not.toHaveAttribute("open");
    expect(screen.getByText(en["cctv.case.optional"])).toBeInTheDocument();
  });

  it("has every essential field from the spec", () => {
    renderForm();

    for (const key of [
      "cctv.case.caseLabel",
      "cctv.case.operatorName",
      "cctv.case.recorderMake",
      "cctv.case.recorderModel",
      "cctv.case.channel",
      "cctv.case.clockOffset",
      "cctv.case.clockOffsetMethod",
    ] as const) {
      expect(screen.getByLabelText(en[key])).toBeInTheDocument();
    }
  });

  it("suggests measuring the offset against the official time", () => {
    renderForm();

    expect(screen.getByText(en["cctv.case.clockOffsetMethod.hint"])).toBeInTheDocument();
  });

  it("caps the text at the report's limits", () => {
    renderForm();

    expect(screen.getByLabelText(en["cctv.case.caseLabel"])).toHaveAttribute("maxLength", String(SHORT_TEXT_MAX));
    expect(screen.getByLabelText(en["cctv.case.clockOffsetMethod"])).toHaveAttribute(
      "maxLength",
      String(LONG_TEXT_MAX),
    );
  });

  it("reports each edit as a new value", () => {
    const onChange = renderForm();

    fireEvent.change(screen.getByLabelText(en["cctv.case.operatorName"]), { target: { value: "S. Q." } });

    expect(onChange).toHaveBeenLastCalledWith({ ...EMPTY_CASE_DETAILS, operatorName: "S. Q." });
  });

  it("flags an offset that isn't a number of seconds", () => {
    renderForm();
    const offset = screen.getByLabelText(en["cctv.case.clockOffset"]);

    fireEvent.change(offset, { target: { value: "ten" } });

    expect(offset).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByRole("alert")).toHaveTextContent(en["cctv.case.offsetInvalid"]);
  });

  it("accepts a negative offset", () => {
    renderForm();
    const offset = screen.getByLabelText(en["cctv.case.clockOffset"]);

    fireEvent.change(offset, { target: { value: "-37.5" } });

    expect(offset).toHaveAttribute("aria-invalid", "false");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });
});
