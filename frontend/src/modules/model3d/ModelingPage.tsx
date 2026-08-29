import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  Info,
  Ruler,
  ScanLine,
  Shapes,
  UploadCloud,
} from "lucide-react";
import { useState, type ChangeEvent, type DragEvent } from "react";
import { useTranslation } from "../../i18n/LocaleProvider";
import {
  auditMesh,
  buildReferenceScene,
  fetchModel3dCapabilities,
  fetchProportions,
  renameViews,
  scoreFit,
  splitSheetViews,
  type FitScore,
  type MeshAudit,
  type MeshFit,
  type Model3dCapabilities,
  type ProportionsResponse,
  type ReferenceScene,
  type SheetView,
  type SheetViews,
  type ViewFit,
} from "../../services/model3d";

const MESH_FORMATS = ".stl,.obj,.ply,.glb,.gltf,.fbx";
const SHEET_FORMATS = ".png,.jpg,.jpeg,.webp";
const DEFAULT_HEIGHT_M = 1.7;
const VIEW_NAMES = ["front", "side", "back", "side_left"] as const;

function Dropzone({
  id,
  file,
  accept,
  hint,
  label,
  onFileSelected,
}: {
  id: string;
  file: File | null;
  accept: string;
  hint: string;
  label: string;
  onFileSelected: (file: File) => void;
}) {
  function handleDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    const dropped = event.dataTransfer.files[0];
    if (dropped) {
      onFileSelected(dropped);
    }
  }

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0];
    if (selected) {
      onFileSelected(selected);
    }
  }

  return (
    <label
      htmlFor={id}
      onDragOver={(event) => event.preventDefault()}
      onDrop={handleDrop}
      className="flex cursor-pointer flex-col items-center gap-2 rounded border border-dashed border-border bg-surface px-6 py-10 text-center transition-[border-color] duration-fast hover:border-accent"
    >
      <UploadCloud aria-hidden="true" className="h-6 w-6 text-text-faint" strokeWidth={1.5} />
      <span className="text-sm text-text">{file ? file.name : label}</span>
      <span className="text-xs text-text-faint">{hint}</span>
      <input
        id={id}
        type="file"
        accept={accept}
        aria-label={label}
        className="sr-only"
        onChange={handleChange}
      />
    </label>
  );
}

/** Sin Blender el carril esta apagado y se dice por que. No es un error. */
function BlenderStatus({ capabilities }: { capabilities: Model3dCapabilities }) {
  const { t } = useTranslation();
  const usable = capabilities.unlocked.length > 0;

  return (
    <div
      role="status"
      className={`flex items-start gap-2 rounded border p-3 ${
        usable ? "border-ok bg-surface-2" : "border-warn bg-surface-2"
      }`}
    >
      <Info
        aria-hidden="true"
        className={`mt-0.5 h-5 w-5 shrink-0 ${usable ? "text-ok" : "text-warn"}`}
        strokeWidth={1.75}
      />
      <div className="flex flex-col gap-1">
        <span className="font-heading text-sm font-semibold text-text">
          {usable
            ? t("modeling.blender.ready", { version: capabilities.blender.version ?? "" })
            : t("modeling.blender.missingTitle")}
        </span>
        {!usable && <span className="text-sm text-text-dim">{capabilities.missing}</span>}
      </div>
    </div>
  );
}

function Warnings({ items }: { items: string[] }) {
  if (items.length === 0) {
    return null;
  }
  return (
    <ul role="status" className="flex flex-col gap-1 rounded border border-warn bg-surface-2 p-3">
      {items.map((aviso) => (
        <li key={aviso} className="flex items-start gap-2 text-sm text-text">
          <AlertTriangle
            aria-hidden="true"
            className="mt-0.5 h-4 w-4 shrink-0 text-warn"
            strokeWidth={1.75}
          />
          {aviso}
        </li>
      ))}
    </ul>
  );
}

function ProportionsPanel({ proportions }: { proportions: ProportionsResponse }) {
  const { t } = useTranslation();

  return (
    <section
      aria-label={t("modeling.proportions.title")}
      className="flex flex-col gap-3 rounded border border-border bg-surface-2 p-3"
    >
      <h3 className="font-heading text-sm font-semibold text-text">
        {t("modeling.proportions.title")}
      </h3>

      <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
        <dt className="text-text-dim">{t("modeling.proportions.headsTall")}</dt>
        <dd className="font-mono-tabular text-text">{proportions.headsTall} ×</dd>
        <dt className="text-text-dim">{t("modeling.proportions.headHeight")}</dt>
        <dd className="font-mono-tabular text-text">{proportions.headMeters} m</dd>
      </dl>

      <div className="flex flex-col gap-2">
        <h4 className="text-xs font-semibold uppercase tracking-wide text-text-dim">
          {t("modeling.proportions.landmarks")}
        </h4>
        <ul className="flex flex-col gap-1">
          {proportions.landmarks.map((landmark) => (
            <li
              key={`${landmark.name}-${landmark.z}`}
              className={`grid grid-cols-[minmax(0,1fr)_auto] gap-x-3 gap-y-1 rounded border px-2 py-1.5 text-sm ${
                landmark.agrees ? "border-border bg-surface" : "border-warn bg-surface"
              }`}
            >
              <span className="min-w-0 text-text">{landmark.name}</span>
              <span className="font-mono-tabular text-text">{landmark.z} m</span>
              {!landmark.agrees && (
                <span className="col-span-2 flex flex-wrap items-center gap-2 text-xs text-warn">
                  <span className="rounded border border-warn px-1.5 py-0.5 font-semibold">
                    {t("modeling.proportions.unreliable")}
                  </span>
                  {t("modeling.proportions.disagreement", {
                    cm: landmark.disagreementCm,
                  })}
                </span>
              )}
            </li>
          ))}
        </ul>
      </div>

      <div className="flex flex-col gap-2">
        <h4 className="text-xs font-semibold uppercase tracking-wide text-text-dim">
          {t("modeling.proportions.widths")}
        </h4>
        <div className="max-h-48 overflow-y-auto rounded border border-border bg-surface">
          <table className="w-full border-collapse text-left text-xs">
            <thead className="sticky top-0 bg-surface-2 text-text-dim">
              <tr>
                <th className="px-2 py-1 font-medium">{t("modeling.proportions.z")}</th>
                <th className="px-2 py-1 font-medium">{t("modeling.proportions.front")}</th>
                <th className="px-2 py-1 font-medium">{t("modeling.proportions.side")}</th>
              </tr>
            </thead>
            <tbody>
              {proportions.widths.map((width) => (
                <tr key={width.z} className="border-t border-border">
                  <td className="px-2 py-1 font-mono-tabular text-text">{width.z}</td>
                  <td className="px-2 py-1 font-mono-tabular text-text">{width.frontCm}</td>
                  <td className="px-2 py-1 font-mono-tabular text-text">{width.sideCm}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function FitPanel({ token, views }: { token: string; views: SheetView[] }) {
  const { t } = useTranslation();
  const defaultScaleView =
    views.reduce<{ name: string; height: number } | null>((tallest, view) => {
      const height = view.inkBox[3] - view.inkBox[1];
      return !tallest || height > tallest.height ? { name: view.name, height } : tallest;
    }, null)?.name ?? "";
  const [mesh, setMesh] = useState<File | null>(null);
  const [heightMeters, setHeightMeters] = useState("1.70");
  const [scaleView, setScaleView] = useState(defaultScaleView);
  const [score, setScore] = useState<FitScore | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fit: MeshFit | null = score?.fit ?? null;

  async function handleScore() {
    if (!mesh) {
      return;
    }
    setIsBusy(true);
    setError(null);
    setScore(null);
    try {
      setScore(await scoreFit(token, mesh, Number(heightMeters), scaleView));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setIsBusy(false);
    }
  }

  return (
    <section
      aria-label={t("modeling.fit.title")}
      className="flex flex-col gap-4 rounded border border-border bg-surface-2 p-4"
    >
      <header className="flex flex-col gap-1">
        <h3 className="flex items-center gap-2 font-heading text-sm font-semibold text-text">
          <ScanLine aria-hidden="true" className="h-4 w-4 text-accent" strokeWidth={1.75} />
          {t("modeling.fit.title")}
        </h3>
        <p className="text-sm text-text-dim">{t("modeling.fit.subtitle")}</p>
      </header>

      <Dropzone
        id="modeling-fit-mesh-input"
        file={mesh}
        accept={MESH_FORMATS}
        label={t("modeling.fit.dropzone")}
        hint="STL · OBJ · PLY · GLB · GLTF · FBX"
        onFileSelected={(elegido) => {
          setMesh(elegido);
          setScore(null);
        }}
      />

      <div className="flex flex-wrap items-end gap-4">
        <label className="flex flex-col gap-1 text-sm text-text-dim">
          {t("modeling.fit.scaleView")}
          <select
            value={scaleView}
            onChange={(event) => setScaleView(event.target.value)}
            className="min-w-32 rounded border border-border bg-surface px-2 py-1 text-text"
          >
            {views.map((view) => (
              <option key={view.name} value={view.name}>
                {view.name}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-sm text-text-dim">
          {t("modeling.fit.height")}
          <input
            type="number"
            step="0.01"
            min="0.01"
            value={heightMeters}
            onChange={(event) => setHeightMeters(event.target.value)}
            className="w-28 rounded border border-border bg-surface px-2 py-1 text-text"
          />
        </label>
        <button
          type="button"
          disabled={!mesh || isBusy}
          onClick={handleScore}
          className="flex items-center gap-2 rounded bg-accent px-4 py-2 text-sm font-semibold text-on-accent disabled:opacity-50"
        >
          <ScanLine aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
          {t("modeling.fit.run")}
        </button>
      </div>
      <p className="text-xs text-text-faint">{t("modeling.fit.scaleViewHint")}</p>

      {fit && (
        <div className="flex flex-col gap-3">
          <dl className="grid gap-3 sm:grid-cols-2">
            <div className="rounded border border-ok bg-surface p-3">
              <dt className="text-xs font-semibold uppercase tracking-wide text-text-dim">
                {t("modeling.fit.average")}
              </dt>
              <dd className="font-mono-tabular text-lg font-semibold text-ok">
                {fit.average.toFixed(3)}
              </dd>
            </div>
            <div className="rounded border border-border bg-surface p-3">
              <dt className="text-xs font-semibold uppercase tracking-wide text-text-dim">
                {t("modeling.fit.worst")}
              </dt>
              <dd className="text-lg font-semibold text-text">{fit.worstView}</dd>
            </div>
          </dl>

          <div className="overflow-x-auto rounded border border-border bg-surface">
            <table className="w-full min-w-[48rem] border-collapse text-left text-xs">
              <thead className="bg-surface-2 text-text-dim">
                <tr>
                  <th className="px-2 py-2 font-medium">{t("modeling.fit.view")}</th>
                  <th className="px-2 py-2 font-medium">{t("modeling.fit.anchored")}</th>
                  <th className="px-2 py-2 font-medium">{t("modeling.fit.best")}</th>
                  <th className="px-2 py-2 font-medium">{t("modeling.fit.blameColumn")}</th>
                  <th className="px-2 py-2 font-medium">{t("modeling.fit.measured")}</th>
                </tr>
              </thead>
              <tbody>
                {fit.views.map((view: ViewFit) => (
                  <tr key={view.view} className="border-t border-border align-top">
                    <td className="px-2 py-2 font-semibold text-text">{view.view}</td>
                    <td className="px-2 py-2 font-mono-tabular text-text">
                      {view.anchored.toFixed(3)}
                    </td>
                    <td className="px-2 py-2 font-mono-tabular text-text">
                      {view.best.toFixed(3)}
                    </td>
                    <td className="min-w-64 px-2 py-2">
                      <span className="inline-flex rounded border border-accent bg-surface-2 px-2 py-1 font-semibold text-text">
                        {t("modeling.fit.blame." + view.blame)}
                      </span>
                    </td>
                    <td className="px-2 py-2">
                      <span className="block font-mono-tabular text-text">
                        ↔ {view.widthCm[0]} / {view.widthCm[1]} cm
                      </span>
                      <span className="block font-mono-tabular text-text">
                        ↕ {view.heightCm[0]} / {view.heightCm[1]} cm
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {error && <p className="text-sm text-danger">{error}</p>}
    </section>
  );
}

function ReferenceLane({ enabled }: { enabled: boolean }) {
  const { t } = useTranslation();
  const [file, setFile] = useState<File | null>(null);
  const [expectedViews, setExpectedViews] = useState("4");
  const [heightMeters, setHeightMeters] = useState(String(DEFAULT_HEIGHT_M));
  const [views, setViews] = useState<SheetViews | null>(null);
  const [selectedViewNames, setSelectedViewNames] = useState<string[]>([]);
  const [scene, setScene] = useState<ReferenceScene | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const parsedHeightMeters = Number(heightMeters);
  const heightIsValid = Number.isFinite(parsedHeightMeters) && parsedHeightMeters > 0;
  const hasDuplicateViewNames = new Set(selectedViewNames).size !== selectedViewNames.length;
  const proportionsQuery = useQuery({
    queryKey: ["model3d-proportions", views?.token, parsedHeightMeters],
    queryFn: () => fetchProportions(views!.token, parsedHeightMeters),
    enabled: Boolean(views && heightIsValid),
  });

  async function handleSplit() {
    if (!file) {
      return;
    }
    setIsBusy(true);
    setError(null);
    setScene(null);
    setViews(null);
    setSelectedViewNames([]);
    try {
      const splitViews = await splitSheetViews(file, Number(expectedViews) || 4);
      setViews(splitViews);
      setSelectedViewNames(splitViews.views.map((view) => view.name));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setIsBusy(false);
    }
  }

  async function handleRenameViews() {
    if (!views || hasDuplicateViewNames) {
      return;
    }
    const namesChanged = selectedViewNames.some(
      (name, index) => name !== views.views[index]?.name,
    );
    setIsBusy(true);
    setError(null);
    try {
      const renamedViews = await renameViews(views.token, selectedViewNames);
      setViews(renamedViews);
      setSelectedViewNames(renamedViews.views.map((view) => view.name));
      if (namesChanged) {
        setScene(null);
      }
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setIsBusy(false);
    }
  }

  async function handleBuild() {
    if (!views || !heightIsValid) {
      return;
    }
    setIsBusy(true);
    setError(null);
    try {
      setScene(await buildReferenceScene(views.token, parsedHeightMeters));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setIsBusy(false);
    }
  }

  return (
    <section className="flex flex-col gap-4">
      <p className="text-sm text-text-dim">{t("modeling.reference.subtitle")}</p>

      <Dropzone
        id="modeling-sheet-input"
        file={file}
        accept={SHEET_FORMATS}
        label={t("modeling.reference.dropzone")}
        hint={t("modeling.reference.dropzoneHint")}
        onFileSelected={(elegido) => {
          setFile(elegido);
          setViews(null);
          setSelectedViewNames([]);
          setScene(null);
        }}
      />

      <div className="flex flex-wrap items-end gap-4">
        <label className="flex flex-col gap-1 text-sm text-text-dim">
          {t("modeling.reference.expectedViews")}
          <input
            type="number"
            min={1}
            max={8}
            value={expectedViews}
            onChange={(event) => setExpectedViews(event.target.value)}
            className="w-24 rounded border border-border bg-surface px-2 py-1 text-text"
          />
        </label>
        <button
          type="button"
          disabled={!enabled || !file || isBusy}
          onClick={handleSplit}
          className="rounded bg-accent px-4 py-2 text-sm font-semibold text-on-accent disabled:opacity-50"
        >
          {t("modeling.reference.split")}
        </button>
      </div>

      {views && (
        <div className="flex flex-col gap-3">
          <Warnings items={views.warnings} />
          <p className="text-sm text-text-dim">
            {t("modeling.reference.namingConvention")}
          </p>
          {/* La vista recortada, no solo su nombre: es la unica forma de ver
              de un vistazo si la hoja se partio donde correspondia. */}
          <ul className="flex flex-wrap gap-3">
            {views.views.map((vista, index) => (
              <li
                key={vista.name}
                className="flex w-32 flex-col items-center gap-1 rounded border border-border bg-surface p-2"
              >
                <img
                  src={vista.image}
                  alt={vista.name}
                  loading="lazy"
                  className="h-32 w-full bg-white object-contain"
                />
                <span className="text-sm text-text">{vista.name}</span>
                <span className="font-mono-tabular text-xs text-text-faint">
                  {vista.widthPx} × {vista.heightPx} px
                </span>
                <label className="flex w-full flex-col gap-1 text-xs text-text-dim">
                  {t("modeling.reference.viewName")}
                  <select
                    aria-label={`${t("modeling.reference.viewName")} ${index + 1}`}
                    value={selectedViewNames[index] ?? vista.name}
                    onChange={(event) =>
                      setSelectedViewNames((current) =>
                        current.map((name, currentIndex) =>
                          currentIndex === index ? event.target.value : name,
                        ),
                      )
                    }
                    className="w-full rounded border border-border bg-surface px-2 py-1 text-sm text-text"
                  >
                    {VIEW_NAMES.map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </select>
                </label>
              </li>
            ))}
          </ul>
          <button
            type="button"
            disabled={isBusy || hasDuplicateViewNames}
            onClick={handleRenameViews}
            className="w-fit rounded bg-accent px-4 py-2 text-sm font-semibold text-on-accent disabled:opacity-50"
          >
            {t("modeling.reference.applyNames")}
          </button>
          {hasDuplicateViewNames && (
            <p role="alert" className="text-sm text-danger">
              {t("modeling.reference.duplicateNames")}
            </p>
          )}
          <div className="flex flex-wrap items-end gap-4">
            <label className="flex flex-col gap-1 text-sm text-text-dim">
              {t("modeling.reference.height")}
              <input
                type="number"
                step="0.01"
                min="0.01"
                value={heightMeters}
                onChange={(event) => setHeightMeters(event.target.value)}
                className="w-28 rounded border border-border bg-surface px-2 py-1 text-text"
              />
            </label>
            <button
              type="button"
              disabled={isBusy || !heightIsValid}
              onClick={handleBuild}
              className="rounded bg-accent px-4 py-2 text-sm font-semibold text-on-accent disabled:opacity-50"
            >
              {t("modeling.reference.build")}
            </button>
          </div>
          {!heightIsValid && (
            <p className="text-xs text-danger">
              {t("modeling.reference.height")}: &gt; 0
            </p>
          )}
          <p className="text-xs text-text-faint">{t("modeling.reference.heightHint")}</p>
          {proportionsQuery.data && <ProportionsPanel proportions={proportionsQuery.data} />}
          {proportionsQuery.isError && (
            <p className="text-sm text-danger">
              {proportionsQuery.error instanceof Error
                ? proportionsQuery.error.message
                : String(proportionsQuery.error)}
            </p>
          )}
        </div>
      )}

      {scene && (
        <div className="flex flex-wrap items-start gap-4">
          <a
            href={scene.downloadUrl}
            className="flex w-fit items-center gap-2 rounded border border-ok bg-surface-2 px-4 py-2 text-sm font-semibold text-text"
          >
            <Download aria-hidden="true" className="h-4 w-4 text-ok" strokeWidth={1.75} />
            {t("modeling.reference.download", { height: scene.heightMeters })}
          </a>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
            {scene.placed.map((vista) => (
              <div key={vista.view} className="contents">
                <dt className="text-text-dim">{vista.view}</dt>
                <dd className="font-mono-tabular text-text">
                  {vista.inkHeightMeters} · {vista.planeHeightMeters} × {vista.planeWidthMeters} ·{" "}
                  {t(vista.scaledByInk ? "common.yes" : "common.no")}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      )}

      {views && <FitPanel token={views.token} views={views.views} />}

      {error && <p className="text-sm text-danger">{error}</p>}
    </section>
  );
}

function AuditLane({ enabled }: { enabled: boolean }) {
  const { t } = useTranslation();
  const [file, setFile] = useState<File | null>(null);
  const [audit, setAudit] = useState<MeshAudit | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleAudit() {
    if (!file) {
      return;
    }
    setIsBusy(true);
    setError(null);
    setAudit(null);
    try {
      setAudit(await auditMesh(file));
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setIsBusy(false);
    }
  }

  const filas: [string, string][] = audit
    ? [
        [t("modeling.audit.faces"), audit.faces.toLocaleString()],
        [t("modeling.audit.quads"), `${audit.quads.toLocaleString()} / ${audit.tris.toLocaleString()}`],
        [t("modeling.audit.ngons"), audit.ngons.toLocaleString()],
        [t("modeling.audit.shells"), audit.shells.toLocaleString()],
        [
          t("modeling.audit.size"),
          audit.dims.map((valor) => valor.toFixed(3)).join(" × "),
        ],
        [t("modeling.audit.uvs"), audit.hasUvs ? t("common.yes") : t("common.no")],
      ]
    : [];

  return (
    <section className="flex flex-col gap-4">
      <p className="text-sm text-text-dim">{t("modeling.audit.subtitle")}</p>

      <Dropzone
        id="modeling-mesh-input"
        file={file}
        accept={MESH_FORMATS}
        label={t("modeling.audit.dropzone")}
        hint="STL · OBJ · PLY · GLB · GLTF · FBX"
        onFileSelected={(elegido) => {
          setFile(elegido);
          setAudit(null);
        }}
      />

      <button
        type="button"
        disabled={!enabled || !file || isBusy}
        onClick={handleAudit}
        className="w-fit rounded bg-accent px-4 py-2 text-sm font-semibold text-on-accent disabled:opacity-50"
      >
        {t("modeling.audit.run")}
      </button>

      {audit && (
        <div className="flex flex-col gap-3">
          <div
            role="status"
            className={`flex items-center gap-2 rounded border p-3 ${
              audit.ok ? "border-ok bg-surface-2" : "border-danger bg-surface-2"
            }`}
          >
            {audit.ok ? (
              <CheckCircle2 aria-hidden="true" className="h-5 w-5 text-ok" strokeWidth={1.75} />
            ) : (
              <AlertTriangle aria-hidden="true" className="h-5 w-5 text-danger" strokeWidth={1.75} />
            )}
            <span className="font-heading text-sm font-semibold text-text">
              {audit.ok ? t("modeling.audit.ok") : t("modeling.audit.blocked")}
            </span>
          </div>

          {audit.blockers.length > 0 && (
            <ul className="flex flex-col gap-1 rounded border border-danger bg-surface-2 p-3 text-sm text-text">
              {audit.blockers.map((bloqueo) => (
                <li key={bloqueo}>{bloqueo}</li>
              ))}
            </ul>
          )}
          <Warnings items={audit.warnings} />

          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
            {filas.map(([etiqueta, valor]) => (
              <div key={etiqueta} className="contents">
                <dt className="text-text-dim">{etiqueta}</dt>
                <dd className="font-mono-tabular text-text">{valor}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}

      {error && <p className="text-sm text-danger">{error}</p>}
    </section>
  );
}

export function ModelingPage() {
  const { t } = useTranslation();
  const [lane, setLane] = useState<"reference" | "audit">("reference");

  const capabilitiesQuery = useQuery({
    queryKey: ["model3d-capabilities"],
    queryFn: fetchModel3dCapabilities,
  });
  const capabilities = capabilitiesQuery.data;
  const enabled = (capabilities?.unlocked.length ?? 0) > 0;

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="flex items-center gap-2 font-heading text-xl font-semibold text-text">
          <Shapes aria-hidden="true" className="h-5 w-5 text-accent" strokeWidth={1.75} />
          {t("modeling.title")}
        </h1>
        <p className="text-sm text-text-dim">{t("modeling.subtitle")}</p>
      </header>

      {capabilitiesQuery.isLoading ? (
        <p role="status" className="text-sm text-text-dim">
          {t("capability.tree.loading")}
        </p>
      ) : capabilitiesQuery.isError || !capabilities ? (
        <p role="alert" className="text-sm text-danger">
          {t("capability.tree.loadFailed")}
        </p>
      ) : (
        <BlenderStatus capabilities={capabilities} />
      )}

      <div role="tablist" className="flex gap-2 border-b border-border">
        {(["reference", "audit"] as const).map((id) => (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={lane === id}
            onClick={() => setLane(id)}
            className={`flex items-center gap-2 px-3 py-2 text-sm ${
              lane === id
                ? "border-b-2 border-accent font-semibold text-text"
                : "text-text-dim hover:text-text"
            }`}
          >
            {id === "reference" ? (
              <Ruler aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
            ) : (
              <Shapes aria-hidden="true" className="h-4 w-4" strokeWidth={1.75} />
            )}
            {t(`modeling.lane.${id}`)}
          </button>
        ))}
      </div>

      {lane === "reference" ? <ReferenceLane enabled={enabled} /> : <AuditLane enabled={enabled} />}
    </div>
  );
}
