import { apiGet, apiPostForm, apiPostJson } from "../lib/api";
import type { UploadOptions } from "../lib/uploadRequest";

export interface BlenderBuild {
  found: boolean;
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
  hasUvs: boolean;
  /** Lo que impide seguir. */
  blockers: string[];
  /** Lo que saldría mejor de otra forma. */
  warnings: string[];
  ok: boolean;
}

export interface ViewFit {
  view: string;
  /** IoU alineando por el centro de la caja de tinta. */
  anchored: number;
  /** El IoU más alto dejando correr la silueta. */
  best: number;
  gainFromMoving: number;
  offsetCm: [number, number];
  /**
   * "escala" | "partes" | "forma" — dónde está el problema. "partes" NO
   * significa mover la malla entera: una traslación global no cambia el
   * número porque la comparación centra las dos siluetas.
   */
  blame: string;
  /** (modelo, dibujo) en cm, para que el número se pueda discutir. */
  widthCm: [number, number];
  heightCm: [number, number];
}

export interface MeshFit {
  /** La vista cuya altura real fija la escala de TODA la hoja. */
  scaleView: string;
  scaleViewHeightMeters: number;
  metersPerPixelModel: number;
  metersPerPixelSheet: number;
  average: number;
  worstView: string;
  views: ViewFit[];
}

export interface FitScore {
  /** Viajan juntas: una malla puede calzar la silueta y estar rota. */
  audit: MeshAudit;
  fit: MeshFit;
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

export interface LandmarkResponse {
  name: string;
  z: number;
  front: number;
  side: number;
  agrees: boolean;
  disagreementCm: number;
}

export interface WidthBandResponse {
  z: number;
  frontCm: number;
  sideCm: number;
}

export interface ProportionsResponse {
  heightMeters: number;
  headMeters: number;
  headsTall: number;
  landmarks: LandmarkResponse[];
  uncertain: string[];
  widths: WidthBandResponse[];
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

export function scoreFit(
  token: string,
  file: File,
  heightMeters: number,
  scaleView?: string,
  options: UploadOptions = {},
): Promise<FitScore> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("heightMeters", String(heightMeters));
  if (scaleView) {
    formData.append("scaleView", scaleView);
  }
  return apiPostForm<FitScore>(`/model3d/fit/${token}`, formData, options);
}

export function renameViews(token: string, names: string[]): Promise<SheetViews> {
  return apiPostJson<SheetViews>(`/model3d/sheet/${token}/names`, { names });
}

export function fetchProportions(
  token: string,
  heightMeters: number,
): Promise<ProportionsResponse> {
  return apiGet<ProportionsResponse>(
    `/model3d/proportions/${token}?heightMeters=${encodeURIComponent(String(heightMeters))}`,
  );
}

export function buildReferenceScene(
  token: string,
  heightMeters: number,
): Promise<ReferenceScene> {
  return apiPostJson<ReferenceScene>("/model3d/reference-scene", { token, heightMeters });
}
