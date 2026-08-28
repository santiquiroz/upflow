import { apiGet, apiPostForm, apiPostJson } from "../lib/api";
import type { UploadOptions } from "../lib/uploadRequest";

export interface BlenderBuild {
  found: boolean;
  path: string | null;
  version: string | null;
  meetsMinimum: boolean;
}

export interface Model3dCapabilities {
  blender: BlenderBuild;
  /** Qué se puede hacer HOY en esta máquina. Vacío = carril apagado, no error. */
  unlocked: string[];
  missing: string | null;
}

export interface MeshAudit {
  vertices: number;
  faces: number;
  tris: number;
  quads: number;
  ngons: number;
  nonManifoldEdges: number;
  boundaryEdges: number;
  looseVerts: number;
  shells: number;
  dims: [number, number, number];
  thinnestAxisRatio: number;
  hasUvs: boolean;
  /** Lo que impide seguir. */
  blockers: string[];
  /** Lo que saldría mejor de otra forma. */
  warnings: string[];
  ok: boolean;
}

export interface SheetView {
  name: string;
  image: string;
  widthPx: number;
  heightPx: number;
  inkBox: [number, number, number, number];
}

export interface SheetViews {
  token: string;
  views: SheetView[];
  /** Vacío = la hoja se entendió entera. */
  warnings: string[];
}

export interface PlacedView {
  view: string;
  image: string;
  inkHeightMeters: number;
  planeHeightMeters: number;
  planeWidthMeters: number;
  scaledByInk: boolean;
}

export interface ReferenceScene {
  token: string;
  downloadUrl: string;
  heightMeters: number;
  placed: PlacedView[];
}

export function fetchModel3dCapabilities(): Promise<Model3dCapabilities> {
  return apiGet<Model3dCapabilities>("/model3d/capabilities");
}

export function auditMesh(file: File, options: UploadOptions = {}): Promise<MeshAudit> {
  const formData = new FormData();
  formData.append("file", file);
  return apiPostForm<MeshAudit>("/model3d/audit", formData, options);
}

export function splitSheetViews(
  file: File,
  expectedViews: number,
  options: UploadOptions = {},
): Promise<SheetViews> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("expectedViews", String(expectedViews));
  return apiPostForm<SheetViews>("/model3d/sheet/views", formData, options);
}

export function buildReferenceScene(
  token: string,
  heightMeters: number,
): Promise<ReferenceScene> {
  return apiPostJson<ReferenceScene>("/model3d/reference-scene", { token, heightMeters });
}
