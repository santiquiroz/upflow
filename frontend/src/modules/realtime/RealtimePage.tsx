import { ArrowUpRight, CircleDashed, Zap } from "lucide-react";

const ROADMAP_URL = "https://github.com/santiquiroz/upflow/blob/master/docs/REALTIME_MODULE.md";

const MVP_HIGHLIGHTS: readonly string[] = [
  "Live super-resolution overlay for games or video, via a separate native helper process (a fork of Magpie).",
  "Reuses the same ONNX models installed in Models — no separate catalog.",
  "Anime4K shaders and compact models (RTMoSR, Compact-ESRGAN) designed for an 8–16 ms per-frame budget.",
];

const NOT_VIABLE_YET: readonly string[] = [
  "An open-source frame generation option competitive with Lossless Scaling on Windows.",
  "AMD AFMF orchestrated from an external app — it's a driver-level toggle with no public API.",
  "FidelityFX Frame Interpolation without motion vectors supplied by the game engine.",
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
    description: "Fork/vendor Magpie, shader/ONNX-only overlay, controlled from Upflow.",
  },
  {
    phase: "7.2",
    label: "Profiles and hotkeys",
    description: "Per-game/window profiles, hotkeys, model and device selection.",
  },
  {
    phase: "7.3",
    label: "Frame generation (conditional)",
    description: "Only if the Windows open-source landscape changes.",
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
            Real-time upscaling overlay for games and video — still in design.
          </p>
        </div>
      </div>

      <div role="status" className="rounded border border-border bg-surface p-4">
        <p className="text-sm text-warn">
          Coming soon. This module doesn't launch, configure, or control any process yet — what you see here is
          only the design vision.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4 max-[900px]:grid-cols-1">
        <div className="flex flex-col gap-4 rounded border border-border bg-surface p-4">
          <HighlightList title="MVP (Phase 7.1)" items={MVP_HIGHLIGHTS} toneClassName="text-ok" />
        </div>
        <div className="flex flex-col gap-4 rounded border border-border bg-surface p-4">
          <HighlightList title="Not viable yet" items={NOT_VIABLE_YET} toneClassName="text-danger" />
        </div>
      </div>

      <div className="flex flex-col gap-3 rounded border border-border bg-surface p-4">
        <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">Phases</h2>
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
        View the module's full design vision
        <ArrowUpRight aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
      </a>
    </div>
  );
}
