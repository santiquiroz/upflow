import { useEffect, useSyncExternalStore } from "react";
import { useTranslation } from "../../i18n/LocaleProvider";
import { editorHandoffStore, type EditorHandoff, type EditorHandoffStore } from "../../lib/editorHandoffStore";

export type FetchBlob = (url: string) => Promise<Blob>;

export interface EditorHandoffDeps {
  store: EditorHandoffStore;
  fetchBlob: FetchBlob;
}

const FALLBACK_TYPE = "image/png";

async function fetchBlobOrThrow(url: string): Promise<Blob> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.blob();
}

const DEFAULT_DEPS: EditorHandoffDeps = { store: editorHandoffStore, fetchBlob: fetchBlobOrThrow };

export function handoffFile(blob: Blob, handoff: EditorHandoff): File {
  return new File([blob], handoff.fileName, { type: blob.type || FALLBACK_TYPE });
}

export function useEditorHandoff(
  onFile: (file: File) => void,
  onError: (message: string) => void,
  deps: EditorHandoffDeps = DEFAULT_DEPS,
): void {
  const { t } = useTranslation();
  const { store, fetchBlob } = deps;
  const pending = useSyncExternalStore(store.subscribe, store.getSnapshot);

  useEffect(() => {
    if (pending === null) {
      return;
    }
    const handoff = store.take();
    if (handoff === null) {
      return;
    }
    fetchBlob(handoff.url)
      .then((blob) => onFile(handoffFile(blob, handoff)))
      .catch(() => onError(t("editor.handoff.failed", { name: handoff.fileName })));
  }, [pending, store, fetchBlob, onFile, onError, t]);
}
