import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, CheckCircle2, Download, Info, Ruler, Shapes, UploadCloud } from "lucide-react";
import { useState, type ChangeEvent, type DragEvent } from "react";
import { useTranslation } from "../../i18n/LocaleProvider";
import {
  auditMesh,
  buildReferenceScene,
  fetchModel3dCapabilities,
  splitSheetViews,
  type MeshAudit,
  type Model3dCapabilities,
  type ReferenceScene,
  type SheetViews,
} from "../../services/model3d";

const MESH_FORMATS = ".stl,.obj,.ply,.glb,.gltf,.fbx";
const SHEET_FORMATS = ".png,.jpg,.jpeg,.webp";
const DEFAULT_HEIGHT_M = 1.7;

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

function ReferenceLane({ enabled }: { enabled: boolean }) {
  const { t } = useTranslation();
  const [file, setFile] = useState<File | null>(null);
  const [expectedViews, setExpectedViews] = useState("4");
  const [heightMeters, setHeightMeters] = useState(String(DEFAULT_HEIGHT_M));
  const [views, setViews] = useState<SheetViews | null>(null);
  const [scene, setScene] = useState<ReferenceScene | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const parsedHeightMeters = Number(heightMeters);
  const heightIsValid = Number.isFinite(parsedHeightMeters) && parsedHeightMeters > 0;

  async function handleSplit() {
    if (!file) {
      return;
    }
    setIsBusy(true);
    setError(null);
    setScene(null);
    setViews(null);
    try {
      setViews(await splitSheetViews(file, Number(expectedViews) || 4));
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
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
            {views.views.map((vista) => (
              <div key={vista.name} className="contents">
                <dt className="text-text-dim">{vista.name}</dt>
                <dd className="font-mono-tabular text-text">
                  {vista.widthPx} × {vista.heightPx} px
                </dd>
              </div>
            ))}
          </dl>
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
