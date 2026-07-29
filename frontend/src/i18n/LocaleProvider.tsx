import { createContext, useCallback, useContext, useMemo, useState } from "react";
import type { ReactNode } from "react";
import {
  type Locale,
  type TranslationParams,
  detectLocale,
  persistLocale,
  translate,
} from ".";

interface LocaleContextValue {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: string, params?: TranslationParams) => string;
}

const LocaleContext = createContext<LocaleContextValue | null>(null);

export function LocaleProvider({
  children,
  initialLocale,
}: {
  children: ReactNode;
  initialLocale?: Locale;
}) {
  const [locale, setLocaleState] = useState<Locale>(
    () => initialLocale ?? detectLocale(),
  );

  const setLocale = useCallback((next: Locale) => {
    setLocaleState(next);
    persistLocale(next);
  }, []);

  const value = useMemo<LocaleContextValue>(
    () => ({
      locale,
      setLocale,
      t: (key, params) => translate(locale, key, params),
    }),
    [locale, setLocale],
  );

  return <LocaleContext.Provider value={value}>{children}</LocaleContext.Provider>;
}

export function useTranslation(): LocaleContextValue {
  const context = useContext(LocaleContext);
  if (context === null) {
    // Sin provider se degrada al locale detectado en vez de reventar: un
    // componente renderizado en aislamiento (un test suelto) sigue mostrando
    // texto en vez de tirar la pantalla entera.
    const fallback = detectLocale();
    return {
      locale: fallback,
      setLocale: () => undefined,
      t: (key, params) => translate(fallback, key, params),
    };
  }
  return context;
}
