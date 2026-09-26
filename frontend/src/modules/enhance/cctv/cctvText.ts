import { ApiError } from "../../../lib/api";
import type { TranslationParams } from "../../../i18n";

export type Translate = (key: string, params?: TranslationParams) => string;

export interface CctvErrorInfo {
  key: string | null;
  message: string;
}

// Los textos del catalogo del backend (pasos, filtros) traen su version en ingles como respaldo.
export function translateOr(t: Translate, key: string | null, fallback: string, params?: TranslationParams): string {
  if (key === null) {
    return fallback;
  }
  const translated = t(key, params);
  return translated === key ? fallback : translated;
}

export function errorInfoOf(error: unknown): CctvErrorInfo | null {
  if (error instanceof ApiError) {
    return { key: error.key, message: error.message };
  }
  if (error instanceof Error) {
    return { key: null, message: error.message };
  }
  return null;
}

export function errorText(t: Translate, error: CctvErrorInfo): string {
  return translateOr(t, error.key, error.message);
}
