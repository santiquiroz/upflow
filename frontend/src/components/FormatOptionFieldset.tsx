export interface FormatOption<T extends string> {
  value: T;
  label: string;
  description: string;
}

function formatOptionClassName(isSelected: boolean): string {
  const base =
    "flex cursor-pointer flex-col gap-1 rounded border px-3 py-2 transition-[background-color,border-color] duration-fast focus-within:outline focus-within:outline-2 focus-within:outline-accent";
  if (isSelected) {
    return `${base} border-accent bg-surface-2`;
  }
  return `${base} border-border bg-surface hover:border-text-faint`;
}

// aria-label deliberately overrides the wrapping <label>'s implicit accessible
// name (which would otherwise include the description below): a description
// referencing another option's name (e.g. FLAC's "~50% smaller than WAV")
// would make that option's own accessible name ambiguously match a query for
// the other option's label.
// aria-describedby restores parity for assistive tech by programmatically
// associating the description with the radio input's accessible description.
function FormatOptionRow<T extends string>({
  option,
  name,
  isSelected,
  onChange,
}: {
  option: FormatOption<T>;
  name: string;
  isSelected: boolean;
  onChange: (value: T) => void;
}) {
  const descriptionId = `${name}-format-option-${option.value}-description`;

  return (
    <label className={formatOptionClassName(isSelected)}>
      <span className="flex items-center gap-2">
        <input
          type="radio"
          name={name}
          aria-label={option.label}
          aria-describedby={descriptionId}
          checked={isSelected}
          onChange={() => onChange(option.value)}
          className="h-3.5 w-3.5 accent-accent"
        />
        <span className="text-sm text-text">{option.label}</span>
      </span>
      <span id={descriptionId} className="pl-[26px] text-xs text-text-faint">{option.description}</span>
    </label>
  );
}

interface FormatOptionFieldsetProps<T extends string> {
  legend: string;
  name: string;
  options: readonly FormatOption<T>[];
  value: T;
  onChange: (value: T) => void;
}

export function FormatOptionFieldset<T extends string>({
  legend,
  name,
  options,
  value,
  onChange,
}: FormatOptionFieldsetProps<T>) {
  return (
    <fieldset className="flex flex-col gap-2">
      <legend className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">{legend}</legend>
      {options.map((option) => (
        <FormatOptionRow key={option.value} option={option} name={name} isSelected={option.value === value} onChange={onChange} />
      ))}
    </fieldset>
  );
}
