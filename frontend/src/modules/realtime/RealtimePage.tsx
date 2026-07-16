import { ArrowUpRight, CircleDashed, Zap } from "lucide-react";

const ROADMAP_URL = "https://github.com/santiquiroz/upflow/blob/master/docs/REALTIME_MODULE.md";

const MVP_HIGHLIGHTS: readonly string[] = [
  "Overlay de super-resolución en vivo sobre juegos o video, vía un proceso helper nativo separado (fork de Magpie).",
  "Reutiliza los mismos modelos ONNX instalados en Models — sin catálogo aparte.",
  "Shaders Anime4K y modelos compactos (RTMoSR, Compact-ESRGAN) pensados para un presupuesto de 8–16 ms por frame.",
];

const NOT_VIABLE_YET: readonly string[] = [
  "Frame generation open-source competitivo con Lossless Scaling en Windows.",
  "AFMF de AMD orquestado desde una app externa — es un toggle de driver sin API pública.",
  "FidelityFX Frame Interpolation sin motion vectors provistos por el motor del juego.",
];

interface RoadmapPhaseInfo {
  phase: string;
  label: string;
  description: string;
}

const ROADMAP_PHASES: readonly RoadmapPhaseInfo[] = [
  {
    phase: "7.1",
    label: "MVP overlay",
    description: "Fork/vendor de Magpie, overlay solo-shader/ONNX, control desde Upflow.",
  },
  {
    phase: "7.2",
    label: "Perfiles y hotkeys",
    description: "Perfiles por juego/ventana, hotkeys, selección de modelo y dispositivo.",
  },
  {
    phase: "7.3",
    label: "Frame generation (condicional)",
    description: "Solo si cambia el panorama open-source de Windows.",
  },
];

function RoadmapPhaseRow({ phase, label, description }: RoadmapPhaseInfo) {
  return (
    <li className="flex flex-col gap-1 rounded border border-border bg-surface-2 p-3">
      <div className="flex items-center gap-2">
        <span className="font-mono-tabular rounded-sm border border-border px-1.5 py-0.5 text-xs font-medium text-accent">
          {phase}
        </span>
        <span className="text-sm font-medium text-text">{label}</span>
      </div>
      <p className="text-xs text-text-dim">{description}</p>
    </li>
  );
}

function HighlightList({
  title,
  items,
  toneClassName,
}: {
  title: string;
  items: readonly string[];
  toneClassName: string;
}) {
  return (
    <div className="flex flex-col gap-2">
      <h2 className={`font-heading text-xs font-semibold uppercase tracking-wide ${toneClassName}`}>{title}</h2>
      <ul className="flex flex-col gap-2">
        {items.map((item) => (
          <li key={item} className="flex items-start gap-2 text-sm text-text-dim">
            <CircleDashed aria-hidden="true" className="mt-0.5 h-3.5 w-3.5 shrink-0 text-text-faint" strokeWidth={1.75} />
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export function RealtimePage() {
  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-start gap-3">
        <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded border border-border bg-surface-2 text-accent">
          <Zap aria-hidden="true" className="h-5 w-5" strokeWidth={1.75} />
        </span>
        <div>
          <h1 className="font-heading text-2xl font-semibold text-text">Realtime</h1>
          <p className="mt-1 text-sm text-text-dim">
            Overlay de reescalado en tiempo real para juegos y video — todavía en diseño.
          </p>
        </div>
      </div>

      <div role="status" className="rounded border border-border bg-surface p-4">
        <p className="text-sm text-warn">
          Próximamente. Este módulo aún no lanza, configura ni controla ningún proceso — lo que ves aquí es solo la
          visión de diseño.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 max-[900px]:grid-cols-1">
        <div className="flex flex-col gap-4 rounded border border-border bg-surface p-4">
          <HighlightList title="MVP (Fase 7.1)" items={MVP_HIGHLIGHTS} toneClassName="text-ok" />
        </div>
        <div className="flex flex-col gap-4 rounded border border-border bg-surface p-4">
          <HighlightList title="No viable todavía" items={NOT_VIABLE_YET} toneClassName="text-danger" />
        </div>
      </div>

      <div className="flex flex-col gap-3 rounded border border-border bg-surface p-4">
        <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">Fases</h2>
        <ul className="flex flex-col gap-2">
          {ROADMAP_PHASES.map((phaseInfo) => (
            <RoadmapPhaseRow key={phaseInfo.phase} {...phaseInfo} />
          ))}
        </ul>
      </div>

      <a
        href={ROADMAP_URL}
        target="_blank"
        rel="noreferrer"
        className="inline-flex w-fit items-center gap-1.5 text-sm text-accent transition-colors duration-fast hover:text-accent-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent"
      >
        Ver la visión completa del módulo
        <ArrowUpRight aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
      </a>
    </div>
  );
}
