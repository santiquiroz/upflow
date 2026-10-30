import type { EditorPerson } from "../../services/editor";
import { useTranslation } from "../../i18n/LocaleProvider";
import { groupPeople, personNumbers, type Choice } from "./peopleState";

interface PeopleOverlayProps {
  people: EditorPerson[];
  choices: Record<number, Choice>;
  hoveredId: number | null;
  onHover: (id: number | null) => void;
  onToggle: (id: number) => void;
  imageWidth: number;
  imageHeight: number;
}

export function peopleOverlayLabelSize(imageWidth: number) {
  const fontSize = Math.max(10, imageWidth / 80);
  return {
    fontSize,
    width: fontSize * 7.4,
    height: fontSize * 1.7,
    padding: fontSize * 0.6,
    radius: fontSize * 0.8,
  };
}

export function PeopleOverlay({
  people,
  choices,
  hoveredId,
  onHover,
  onToggle,
  imageWidth,
  imageHeight,
}: PeopleOverlayProps) {
  const { t } = useTranslation();
  const { front, back } = groupPeople(people);
  const rows = [...front, ...back];
  const numbers = personNumbers(people);
  return (
    <svg
      aria-label={t("editor.people.overlay")}
      className="pointer-events-none absolute inset-0 h-full w-full overflow-visible"
      viewBox={`0 0 ${imageWidth} ${imageHeight}`}
      preserveAspectRatio="none"
    >
      {rows.map((person) => {
        const hovered = hoveredId === person.id;
        const color = hovered ? "#f5a524" : choices[person.id] === "keep" ? "#22c55e" : "#ef4444";
        const [x1, y1, x2, y2] = person.box;
        const labelSize = peopleOverlayLabelSize(imageWidth);
        const labelY = Math.max(0, y1 - labelSize.height - labelSize.fontSize * 0.3);
        return (
          <g
            key={person.id}
            data-person-id={person.id}
            data-hovered={hovered ? "true" : "false"}
            className="pointer-events-auto cursor-pointer"
            onMouseEnter={() => onHover(person.id)}
            onMouseLeave={() => onHover(null)}
            onClick={() => onToggle(person.id)}
          >
            <rect
              data-testid={`person-outline-${person.id}`}
              x={x1}
              y={y1}
              width={x2 - x1}
              height={y2 - y1}
              rx={8}
              fill={color}
              fillOpacity={hovered ? 0.18 : 0.08}
              stroke={color}
              strokeWidth={hovered ? 3 : 1.5}
              strokeDasharray={hovered ? undefined : "5 4"}
              vectorEffect="non-scaling-stroke"
            />
            {hovered && (
              <g>
                <rect
                  x={x1}
                  y={labelY}
                  width={labelSize.width}
                  height={labelSize.height}
                  rx={labelSize.radius}
                  fill="#f5a524"
                />
                <text
                  x={x1 + labelSize.padding}
                  y={labelY + labelSize.fontSize * 1.2}
                  fill="#111"
                  fontSize={labelSize.fontSize}
                  fontWeight={600}
                >
                  {t("editor.people.person", { number: numbers[person.id] })}
                </text>
              </g>
            )}
          </g>
        );
      })}
    </svg>
  );
}
