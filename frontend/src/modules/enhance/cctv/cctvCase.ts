import type { CctvAcquisition, CctvCaseRequest } from "../../../services/cctv";

export interface CaseDetails {
  caseLabel: string;
  operatorName: string;
  recorderMake: string;
  recorderModel: string;
  recorderSerial: string;
  channel: string;
  clockOffset: string;
  clockOffsetMethod: string;
  exportMethod: string;
  exportDate: string;
}

export type CaseField = keyof CaseDetails;

// Los mismos topes que el informe (`ShortText` y `UserText` de cctv_report_model.py).
export const SHORT_TEXT_MAX = 200;
export const LONG_TEXT_MAX = 2000;

export const EMPTY_CASE_DETAILS: CaseDetails = {
  caseLabel: "",
  operatorName: "",
  recorderMake: "",
  recorderModel: "",
  recorderSerial: "",
  channel: "",
  clockOffset: "",
  clockOffsetMethod: "",
  exportMethod: "",
  exportDate: "",
};

const TEXT_ACQUISITION_FIELDS = [
  "recorderMake",
  "recorderModel",
  "recorderSerial",
  "channel",
  "clockOffsetMethod",
  "exportMethod",
  "exportDate",
] as const;
const DECIMAL_NUMBER = /^[+-]?(\d+\.?\d*|\.\d+)$/;

export function withCaseField(details: CaseDetails, field: CaseField, value: string): CaseDetails {
  return { ...details, [field]: value };
}

export function parseClockOffset(text: string): number | null {
  // Sin coma decimal: "1,500" se leeria como 1,5 o como 1500 segun quien lo escriba.
  const normalized = text.trim();
  if (!DECIMAL_NUMBER.test(normalized)) {
    return null;
  }
  const value = Number(normalized);
  return Number.isFinite(value) ? value : null;
}

export function isClockOffsetValid(text: string): boolean {
  return text.trim() === "" || parseClockOffset(text) !== null;
}

export function isCaseDetailsValid(details: CaseDetails): boolean {
  return isClockOffsetValid(details.clockOffset);
}

function presentText(value: string): string | null {
  const trimmed = value.trim();
  return trimmed === "" ? null : trimmed;
}

function textAcquisition(details: CaseDetails): CctvAcquisition {
  return Object.fromEntries(
    TEXT_ACQUISITION_FIELDS.flatMap((field) => {
      const value = presentText(details[field]);
      return value === null ? [] : [[field, value]];
    }),
  );
}

export function acquisitionOf(details: CaseDetails): CctvAcquisition {
  const offset = parseClockOffset(details.clockOffset);
  const text = textAcquisition(details);
  return offset === null ? text : { ...text, clockOffsetSeconds: offset };
}

// Lo que el usuario no llena no viaja: el pedido queda igual al de un job sin datos del caso.
export function caseRequestOf(details: CaseDetails): CctvCaseRequest {
  const caseLabel = presentText(details.caseLabel);
  const operatorName = presentText(details.operatorName);
  const acquisition = acquisitionOf(details);
  return {
    ...(caseLabel === null ? {} : { caseLabel }),
    ...(operatorName === null ? {} : { operatorName }),
    ...(Object.keys(acquisition).length === 0 ? {} : { acquisition }),
  };
}
