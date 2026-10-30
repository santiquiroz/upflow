import type { EditorPerson } from "../../services/editor";
import { useTranslation } from "../../i18n/LocaleProvider";
import { groupPeople, type Choice } from "./peopleState";

interface PeopleOverlayProps {
  people: EditorPerson[];
  choices: Record<number, Choice>;
  hoveredId: number | null;
  onHover: (id: number | null) => void;
  onToggle: (id: number) => void;
  imageWidth: number;
  imageHeight: number;
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
  return (
    <svg
      aria-label={t("editor.people.overlay")}
      className="pointer-events-none absolute inset-0 h-full w-full overflow-visible"
      viewBox={`0 0 ${imageWidth} ${imageHeight}`}
      preserveAspectRatio="none"
    >
      {rows.map((person, index) => {
        const hovered = hoveredId === person.id;
        const color = hovered ? "#f5a524" : choices[person.id] === "keep" ? "#22c55e" : "#ef4444";
        const [x1, y1, x2, y2] = person.box;
        return (
          <g
            key={person.id}
            data-person-id={person.id}
            data-hovered={hovered ? "true" : "false"}
            className="pointer-events-auto cursor-pointer"
            onMouseEnter={() => onHover(person.id)}
            onMouseLeave={() => onHover(null)}
            onClick={(event) => {
              event.stopPropagation();
              onToggle(person.id);
            }}
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
                <rect x={x1} y={Math.max(0, y1 - 20)} width={74} height={17} rx={8} fill="#f5a524" />
                <text x={x1 + 6} y={Math.max(12, y1 - 8)} fill="#111" fontSize={10} fontWeight={600}>
                  Person {index + 1}
                </text>
              </g>
            )}
          </g>
        );
      })}
    </svg>
  );
}
