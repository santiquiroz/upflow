export type TrackedJobKind = "image" | "video" | "audio" | "generation";

export interface TrackedJob {
  id: string;
  kind: TrackedJobKind;
  fileName: string;
  createdAt: number;
}

export interface JobQueueStore {
  subscribe: (listener: () => void) => () => void;
  getSnapshot: () => TrackedJob[];
  addTrackedJob: (job: TrackedJob) => void;
  removeTrackedJob: (id: string) => void;
}

// A plain module-level store (subscribe/getSnapshot) instead of a React
// Context: ImagePanel and VideoPanel submit jobs from inside hooks that run
// outside any single component subtree, and JobQueue lives in AppShell far
// away from both -- a context provider would have to wrap the whole app for
// no benefit over a singleton, and createJobQueueStore() still lets tests
// build an isolated instance instead of sharing global state.
export function createJobQueueStore(): JobQueueStore {
  let jobs: TrackedJob[] = [];
  const listeners = new Set<() => void>();

  function emitChange(): void {
    listeners.forEach((listener) => listener());
  }

  function subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  }

  function getSnapshot(): TrackedJob[] {
    return jobs;
  }

  function addTrackedJob(job: TrackedJob): void {
    jobs = [job, ...jobs];
    emitChange();
  }

  function removeTrackedJob(id: string): void {
    jobs = jobs.filter((tracked) => tracked.id !== id);
    emitChange();
  }

  return { subscribe, getSnapshot, addTrackedJob, removeTrackedJob };
}

export const jobQueueStore: JobQueueStore = createJobQueueStore();
