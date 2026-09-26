export interface CctvOption<T extends string | number> {
  value: T;
  label: string;
}

function optionClassName(isActive: boolean): string {
  const base =
    "rounded-sm border px-3 py-1.5 text-sm transition-[background-color,border-color,color] duration-fast focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent";
  return isActive ? `${base} border-accent bg-accent text-bg` : `${base} border-border bg-surface text-text-dim hover:border-text-faint hover:text-text`;
}

export function CctvOptionGroup<T extends string | number>({
  legend,
  options,
  value,
  onChange,
}: {
  legend: string;
  options: readonly CctvOption<T>[];
  value: T;
  onChange: (value: T) => void;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-xs text-text-dim">{legend}</span>
      <div role="radiogroup" aria-label={legend} className="flex flex-wrap gap-2">
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={option.value === value}
            onClick={() => onChange(option.value)}
            className={optionClassName(option.value === value)}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}
