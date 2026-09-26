import { Maximize, ZoomIn, ZoomOut } from "lucide-react";
import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type PointerEvent,
  type RefObject,
  type SyntheticEvent,
} from "react";
import { useTranslation } from "../i18n/LocaleProvider";
import {
  IDENTITY_VIEWPORT,
  isZoomed,
  panBy,
  transformStyle,
  type ContainerSize,
  type Viewport,
} from "../modules/editor/viewport";
import {
  COMPARE_ZOOM_STEP,
  CURTAIN_CENTER,
  actualSizeZoom,
  curtainAfterKey,
  curtainAtPointer,
  maxZoomFor,
  pixelPercent,
  zoomAroundCenter,
} from "./compareView";

export interface CompareSliderProps {
  beforeSrc: string;
  afterSrc: string;
  beforeAlt: string;
  afterAlt: string;
  // El artefacto "view" a resolucion completa: solo asi "100%" muestra pixeles reales.
  fullResolution: boolean;
}

interface NaturalSize {
  width: number;
  height: number;
}

type Drag = { kind: "curtain" } | { kind: "pan"; x: number; y: number };

const FALLBACK_ASPECT = "4 / 3";
const MAX_STAGE_HEIGHT_VH = 70;
const TOOL_BUTTON_CLASS =
  "inline-flex items-center gap-1 rounded border border-border px-2 py-1 text-xs text-text-dim transition hover:border-accent hover:text-text focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-50";

// Sin ampliar mas alla del tamano real ni pasar del 70% del alto de la ventana.
function stageStyle(natural: NaturalSize | null): CSSProperties {
  if (natural === null) {
    return { aspectRatio: FALLBACK_ASPECT };
  }
  const ratio = natural.width / natural.height;
  return {
    aspectRatio: `${natural.width} / ${natural.height}`,
    maxWidth: `min(${natural.width}px, calc(${MAX_STAGE_HEIGHT_VH}vh * ${ratio}))`,
  };
}

function containerOf(element: HTMLElement | null): ContainerSize {
  return { width: element?.clientWidth ?? 0, height: element?.clientHeight ?? 0 };
}

function useFittedWidth(stageRef: RefObject<HTMLDivElement | null>) {
  const [fittedWidth, setFittedWidth] = useState(0);
  const measure = () => setFittedWidth(stageRef.current?.clientWidth ?? 0);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver(() => setFittedWidth(stage.clientWidth));
    observer.observe(stage);
    return () => observer.disconnect();
  }, [stageRef]);

  return { fittedWidth, measure };
}

function ZoomReadout({ zoom, actualSize, fullResolution }: { zoom: number; actualSize: number | null; fullResolution: boolean }) {
  const { t } = useTranslation();
  const text = fullResolution
    ? actualSize === null
      ? ""
      : t("restore.control.percent", { value: pixelPercent(zoom, actualSize) })
    : t("restore.compare.previewZoom");
  return (
    <output
      aria-label={t("restore.compare.zoomLevel")}
      title={fullResolution ? undefined : t("restore.compare.previewHint")}
      className="min-w-14 text-center font-mono-tabular text-xs text-text"
    >
      {text}
    </output>
  );
}

interface CompareToolbarProps {
  viewport: Viewport;
  actualSize: number | null;
  fullResolution: boolean;
  onZoom: (zoom: number) => void;
  onFit: () => void;
}

function CompareToolbar({ viewport, actualSize, fullResolution, onZoom, onFit }: CompareToolbarProps) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-wrap items-center gap-2">
      <button type="button" className={TOOL_BUTTON_CLASS} onClick={() => onZoom(viewport.zoom / COMPARE_ZOOM_STEP)}>
        <ZoomOut aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
        {t("restore.compare.zoomOut")}
      </button>
      <ZoomReadout zoom={viewport.zoom} actualSize={actualSize} fullResolution={fullResolution} />
      <button type="button" className={TOOL_BUTTON_CLASS} onClick={() => onZoom(viewport.zoom * COMPARE_ZOOM_STEP)}>
        <ZoomIn aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
        {t("restore.compare.zoomIn")}
      </button>
      <button type="button" className={TOOL_BUTTON_CLASS} disabled={!isZoomed(viewport)} onClick={onFit}>
        <Maximize aria-hidden="true" className="h-3.5 w-3.5" strokeWidth={1.75} />
        {t("restore.compare.fit")}
      </button>
      {fullResolution && (
        <button
          type="button"
          className={TOOL_BUTTON_CLASS}
          disabled={actualSize === null}
          onClick={() => actualSize !== null && onZoom(actualSize)}
        >
          {t("restore.compare.actualSize")}
        </button>
      )}
    </div>
  );
}

function SideLabel({ text, side }: { text: string; side: "left" | "right" }) {
  const position = side === "left" ? "left-2" : "right-2";
  return (
    <span
      aria-hidden="true"
      className={`pointer-events-none absolute top-2 ${position} rounded bg-black/60 px-1.5 py-0.5 text-[11px] font-medium text-white`}
    >
      {text}
    </span>
  );
}

interface DividerProps {
  curtain: number;
  onKeyDown: (event: KeyboardEvent<HTMLDivElement>) => void;
  onPointerDown: (event: PointerEvent<HTMLDivElement>) => void;
}

function Divider({ curtain, onKeyDown, onPointerDown }: DividerProps) {
  const { t } = useTranslation();
  return (
    <div
      role="slider"
      tabIndex={0}
      aria-label={t("restore.compare.divider")}
      aria-orientation="horizontal"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(curtain)}
      onKeyDown={onKeyDown}
      onPointerDown={onPointerDown}
      className="group absolute inset-y-0 z-10 w-8 -translate-x-1/2 cursor-ew-resize focus-visible:outline-none"
      style={{ left: `${curtain}%` }}
    >
      <span className="absolute inset-y-0 left-1/2 w-0.5 -translate-x-1/2 bg-white shadow-[0_0_4px_rgb(0_0_0/0.6)]" />
      <span className="absolute left-1/2 top-1/2 h-7 w-7 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-black/50 shadow group-focus-visible:ring-2 group-focus-visible:ring-accent" />
    </div>
  );
}

export function CompareSlider({ beforeSrc, afterSrc, beforeAlt, afterAlt, fullResolution }: CompareSliderProps) {
  const { t } = useTranslation();
  const stageRef = useRef<HTMLDivElement>(null);
  const dragRef = useRef<Drag | null>(null);
  const [curtain, setCurtain] = useState(CURTAIN_CENTER);
  const [viewport, setViewport] = useState<Viewport>(IDENTITY_VIEWPORT);
  const [natural, setNatural] = useState<NaturalSize | null>(null);
  const { fittedWidth, measure } = useFittedWidth(stageRef);
  const actualSize = natural === null ? null : actualSizeZoom(natural.width, fittedWidth);
  const transform = transformStyle(viewport);

  function moveCurtainTo(clientX: number) {
    const rect = stageRef.current?.getBoundingClientRect();
    setCurtain((current) => curtainAtPointer(clientX, rect?.left ?? 0, rect?.width ?? 0, current));
  }

  function startDrag(event: PointerEvent<HTMLDivElement>, drag: Drag) {
    stageRef.current?.setPointerCapture?.(event.pointerId);
    dragRef.current = drag;
  }

  function handleDividerDown(event: PointerEvent<HTMLDivElement>) {
    event.stopPropagation();
    startDrag(event, { kind: "curtain" });
  }

  function handleStageDown(event: PointerEvent<HTMLDivElement>) {
    if (isZoomed(viewport)) {
      startDrag(event, { kind: "pan", x: event.clientX, y: event.clientY });
      return;
    }
    startDrag(event, { kind: "curtain" });
    moveCurtainTo(event.clientX);
  }

  function handleStageMove(event: PointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (drag?.kind === "curtain") {
      moveCurtainTo(event.clientX);
    }
    if (drag?.kind === "pan") {
      const { clientX, clientY } = event;
      dragRef.current = { kind: "pan", x: clientX, y: clientY };
      setViewport((current) => panBy(current, clientX - drag.x, clientY - drag.y, containerOf(stageRef.current)));
    }
  }

  function handleDividerKey(event: KeyboardEvent<HTMLDivElement>) {
    const next = curtainAfterKey(curtain, event.key);
    if (next !== null) {
      event.preventDefault();
      setCurtain(next);
    }
  }

  function handleAfterLoad(event: SyntheticEvent<HTMLImageElement>) {
    const { naturalWidth, naturalHeight } = event.currentTarget;
    setNatural(naturalWidth > 0 && naturalHeight > 0 ? { width: naturalWidth, height: naturalHeight } : null);
    measure();
  }

  function zoomTo(zoom: number) {
    setViewport((current) => zoomAroundCenter(current, zoom, containerOf(stageRef.current), maxZoomFor(actualSize)));
  }

  return (
    <figure className="flex flex-col gap-2" aria-label={t("restore.compare.region")}>
      <div
        ref={stageRef}
        data-testid="compare-stage"
        className="relative mx-auto w-full touch-none select-none overflow-hidden rounded border border-border bg-bg"
        style={stageStyle(natural)}
        onPointerDown={handleStageDown}
        onPointerMove={handleStageMove}
        onPointerUp={() => (dragRef.current = null)}
        onPointerCancel={() => (dragRef.current = null)}
      >
        <div className="absolute inset-0 origin-top-left" style={{ transform }}>
          <img src={beforeSrc} alt={beforeAlt} draggable={false} className="h-full w-full object-contain" />
        </div>
        <div className="absolute inset-0" style={{ clipPath: `inset(0 0 0 ${curtain}%)` }}>
          <div className="absolute inset-0 origin-top-left" style={{ transform }}>
            <img
              src={afterSrc}
              alt={afterAlt}
              draggable={false}
              onLoad={handleAfterLoad}
              className="h-full w-full object-contain"
            />
          </div>
        </div>
        <SideLabel text={t("restore.compare.before")} side="left" />
        <SideLabel text={t("restore.compare.after")} side="right" />
        <Divider curtain={curtain} onKeyDown={handleDividerKey} onPointerDown={handleDividerDown} />
      </div>
      <CompareToolbar
        viewport={viewport}
        actualSize={actualSize}
        fullResolution={fullResolution}
        onZoom={zoomTo}
        onFit={() => setViewport(IDENTITY_VIEWPORT)}
      />
    </figure>
  );
}
