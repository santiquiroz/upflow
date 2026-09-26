export interface EditorHandoff {
  url: string;
  fileName: string;
}

export interface EditorHandoffStore {
  subscribe: (listener: () => void) => () => void;
  getSnapshot: () => EditorHandoff | null;
  offer: (handoff: EditorHandoff) => void;
  take: () => EditorHandoff | null;
}

// Mismo patron que jobQueueStore: el Editor vive montado aparte (KeepMounted) y
// otra pestana le deja una imagen para abrir; el Editor la toma una sola vez.
export function createEditorHandoffStore(): EditorHandoffStore {
  let pending: EditorHandoff | null = null;
  const listeners = new Set<() => void>();

  function emitChange(): void {
    listeners.forEach((listener) => listener());
  }

  function subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  }

  function getSnapshot(): EditorHandoff | null {
    return pending;
  }

  function offer(handoff: EditorHandoff): void {
    pending = handoff;
    emitChange();
  }

  function take(): EditorHandoff | null {
    const taken = pending;
    if (taken !== null) {
      pending = null;
      emitChange();
    }
    return taken;
  }

  return { subscribe, getSnapshot, offer, take };
}

export const editorHandoffStore: EditorHandoffStore = createEditorHandoffStore();
