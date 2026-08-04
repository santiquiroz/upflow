import { ArrowUpRight, Zap } from "lucide-react";
import { RealtimePanel } from "./RealtimePanel";

const ROADMAP_URL = "https://github.com/santiquiroz/upflow/blob/master/docs/REALTIME_MODULE.md";

// Lo que sigue sin ser viable, dicho sin rodeos. No es un "próximamente": son
// límites reales del ecosistema en Windows, no cosas que falte programar.
const NOT_VIABLE_YET: readonly string[] = [
  "Generar cuadros intermedios como hace Lossless Scaling: hoy no existe una opción de código abierto en Windows.",
  "AFMF de AMD no se puede manejar desde una app externa: es un interruptor del driver, sin API pública.",
  "FidelityFX Frame Interpolation necesita vectores de movimiento que solo puede dar el motor del juego.",
];

export function RealtimePage() {
  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-start gap-3">
        <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded border border-border bg-surface-2 text-accent">
          <Zap aria-hidden="true" className="h-5 w-5" strokeWidth={1.75} />
        </span>
        <div>
          <h1 className="font-heading text-2xl font-semibold text-text">Tiempo real</h1>
          <p className="mt-1 text-sm text-text-dim">
            Reescala en vivo lo que estés mirando o jugando, en una ventana superpuesta. El overlay
            lo presenta Magpie, un programa libre que corre aparte y que Upflow configura y abre por
            vos. No hace falta instalar ningún driver.
          </p>
        </div>
      </div>

      <RealtimePanel />

      <div className="flex flex-col gap-2 rounded border border-border bg-surface-2 p-4">
        <h2 className="font-heading text-xs font-semibold uppercase tracking-wide text-text-dim">
          Lo que todavía no se puede
        </h2>
        <ul className="flex flex-col gap-1.5">
          {NOT_VIABLE_YET.map((item) => (
            <li key={item} className="text-xs text-text-faint">
              {item}
            </li>
          ))}
        </ul>
        <a
          href={ROADMAP_URL}
          target="_blank"
          rel="noreferrer"
          className="inline-flex w-fit items-center gap-1.5 text-xs text-accent transition-colors duration-fast hover:text-accent-hover"
        >
          Ver el detalle técnico
          <ArrowUpRight aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
        </a>
      </div>
    </div>
  );
}
