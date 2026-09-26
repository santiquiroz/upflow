import { useTranslation } from "../../../i18n/LocaleProvider";
import {
  isClockOffsetValid,
  LONG_TEXT_MAX,
  SHORT_TEXT_MAX,
  withCaseField,
  type CaseDetails,
  type CaseField,
} from "./cctvCase";

interface CaseDetailsFormProps {
  value: CaseDetails;
  onChange: (next: CaseDetails) => void;
}

interface FieldProps {
  field: CaseField;
  value: CaseDetails;
  onChange: (next: CaseDetails) => void;
}

const SHORT_FIELDS: readonly CaseField[] = ["caseLabel", "operatorName", "recorderMake", "recorderModel", "channel"];
const FIELD_CLASS =
  "w-full rounded-sm border border-border bg-surface px-2 py-1 text-sm text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent aria-[invalid=true]:border-danger";
const LABEL_CLASS = "text-xs text-text-dim";
const OFFSET_ERROR_ID = "cctv-case-clockOffset-error";
const OFFSET_HINT_ID = "cctv-case-clockOffset-hint";

function fieldId(field: CaseField): string {
  return `cctv-case-${field}`;
}

function ShortTextField({ field, value, onChange }: FieldProps) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={fieldId(field)} className={LABEL_CLASS}>
        {t(`cctv.case.${field}`)}
      </label>
      <input
        id={fieldId(field)}
        type="text"
        maxLength={SHORT_TEXT_MAX}
        value={value[field]}
        onChange={(event) => onChange(withCaseField(value, field, event.target.value))}
        className={FIELD_CLASS}
      />
    </div>
  );
}

function ClockOffsetField({ value, onChange }: Omit<FieldProps, "field">) {
  const { t } = useTranslation();
  const isValid = isClockOffsetValid(value.clockOffset);
  const describedBy = isValid ? OFFSET_HINT_ID : `${OFFSET_HINT_ID} ${OFFSET_ERROR_ID}`;
  return (
    <div className="flex flex-col gap-1">
      <label htmlFor={fieldId("clockOffset")} className={LABEL_CLASS}>
        {t("cctv.case.clockOffset")}
      </label>
      <input
        id={fieldId("clockOffset")}
        type="text"
        inputMode="decimal"
        maxLength={32}
        value={value.clockOffset}
        aria-invalid={!isValid}
        aria-describedby={describedBy}
        onChange={(event) => onChange(withCaseField(value, "clockOffset", event.target.value))}
        className={`font-mono-tabular ${FIELD_CLASS} max-w-40`}
      />
      <p id={OFFSET_HINT_ID} className="text-xs text-text-faint">
        {t("cctv.case.clockOffset.hint")}
      </p>
      {!isValid && (
        <p id={OFFSET_ERROR_ID} role="alert" className="text-xs text-danger">
          {t("cctv.case.offsetInvalid")}
        </p>
      )}
    </div>
  );
}

function OffsetMethodField({ value, onChange }: Omit<FieldProps, "field">) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-1 sm:col-span-2">
      <label htmlFor={fieldId("clockOffsetMethod")} className={LABEL_CLASS}>
        {t("cctv.case.clockOffsetMethod")}
      </label>
      <textarea
        id={fieldId("clockOffsetMethod")}
        rows={2}
        maxLength={LONG_TEXT_MAX}
        value={value.clockOffsetMethod}
        onChange={(event) => onChange(withCaseField(value, "clockOffsetMethod", event.target.value))}
        className={FIELD_CLASS}
      />
      <p className="text-xs text-text-faint">{t("cctv.case.clockOffsetMethod.hint")}</p>
    </div>
  );
}

export function CaseDetailsForm({ value, onChange }: CaseDetailsFormProps) {
  const { t } = useTranslation();
  return (
    <details className="rounded border border-border bg-surface-2 p-3">
      <summary className="cursor-pointer text-sm font-medium text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent">
        {t("cctv.case.legend")} <span className="text-xs font-normal text-text-dim">{t("cctv.case.optional")}</span>
      </summary>
      <div className="mt-3 flex flex-col gap-3">
        <p className="text-xs text-text-dim">{t("cctv.case.intro")}</p>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          {SHORT_FIELDS.map((field) => (
            <ShortTextField key={field} field={field} value={value} onChange={onChange} />
          ))}
          <ClockOffsetField value={value} onChange={onChange} />
          <OffsetMethodField value={value} onChange={onChange} />
        </div>
      </div>
    </details>
  );
}
