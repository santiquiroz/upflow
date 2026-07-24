import type { AudioJob, GenerationJob, JobResponse, VideoJobResponse } from "./apiTypes";

export type AnyJobResponse = JobResponse | VideoJobResponse | AudioJob | GenerationJob;

export function isGenerationJob(job: AnyJobResponse): job is GenerationJob {
  return "prompt" in job;
}
