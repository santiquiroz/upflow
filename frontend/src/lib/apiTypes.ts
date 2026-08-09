// Mirrors app/schemas.py response shapes. Field names follow the camelCase
// serialization_alias each Pydantic model uses on the wire, not the Python
// snake_case attribute names.

export type JobStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export type StageStatus = "pending" | "active" | "done";

// Upscale runtime selected per video job. Mirrors the backend UPSCALE_BACKEND
// contract: "auto" lets the server pick the fastest available backend, "ncnn"
// forces Real-ESRGAN NCNN Vulkan, "onnx" forces ONNX Runtime (DirectML/CUDA/CPU).
export type UpscaleBackend = "auto" | "ncnn" | "onnx";

// Video encoder selected per video job. Mirrors the backend video_encoder
// contract: "software" uses libx264/libx265 on the CPU (default), "auto" picks a
// hardware encoder (AMF/NVENC/QSV) by the job's GPU and falls back to software.
export type VideoEncoder = "software" | "auto";

// Mirrors app/services/progress.py::Stage (asdict()'d into job.metadata.stages).
export interface JobStage {
  key: string;
  label: string;
  weight: number;
  status: StageStatus;
}

// Known job.metadata keys written by app/services/progress.py and
// video_upscaler.py. The index signature keeps this permissive for the
// remaining ad-hoc metadata (duration, hasAudio, outputWidth, ...) callers
// read defensively rather than relying on a fully-typed dictionary.
export interface JobMetadata {
  stage?: string;
  stages?: JobStage[];
  stageStartedAt?: string;
  framesDone?: number;
  framesTotal?: number | null;
  interpFramesTotal?: number | null;
  outputFps?: string;
  /** Por que este trabajo corrio rapido o lento. Medido a 1080p->4x en una
   *  RX 7800 XT: onnx fp16 1.58 fps, ncnn 0.72, onnx fp32 0.32. */
  upscaleBackend?: string;
  upscalePrecision?: string;
  upscaleTiled?: boolean;
  [key: string]: unknown;
}

export interface CreateJobResponse {
  jobId: string;
  status: JobStatus;
  statusUrl: string;
  downloadUrl: string | null;
}

export interface JobResponse {
  jobId: string;
  status: JobStatus;
  originalFilename: string;
  modelName: string;
  scale: number;
  outputFormat: string;
  modelId: string | null;
  device: string | null;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  error: string | null;
  ownerId: string | null;
  metadata: JobMetadata;
  progressPct: number | null;
  downloadUrl: string | null;
}

export interface VideoJobResponse {
  jobId: string;
  status: JobStatus;
  originalFilename: string;
  modelName: string;
  scale: number;
  // Present on current responses; optional keeps older persisted/mock jobs
  // readable while exposing the target-resolution contract to new callers.
  targetHeight?: number | null;
  outputContainer: string;
  videoCodec: string;
  videoPreset: string;
  crf: number;
  keepAudio: boolean;
  fpsMultiplier: number;
  targetFps: string | null;
  audioEnhance: string | null;
  audioRestore: string | null;
  // Frame-interpolation engine used for this job: "rife" (default, always) or
  // "gmfss" (opt-in, much higher quality, 10x or more slower). Mirrors app.config.INTERP_ENGINES.
  interpEngine: string;
  // Runtime that ran the upscale. Serialized as `backend` (single word, no
  // camelCase transform); null until a job explicitly selects one, in which
  // case the server behaves as "auto".
  backend?: UpscaleBackend | null;
  // Video encoder for the final re-encode. Serialized as `videoEncoder`.
  videoEncoder?: VideoEncoder | null;
  modelId: string | null;
  device: string | null;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  error: string | null;
  ownerId: string | null;
  metadata: JobMetadata;
  progressPct: number | null;
  downloadUrl: string | null;
}

// Mirrors app/schemas.py::AudioTrackResponse / SubtitleTrackResponse — one
// entry per audio/subtitle stream ffprobe reports on the uploaded video,
// keyed by the stream's absolute ffprobe index (not a per-kind relative one).
export interface AudioTrackInfo {
  index: number;
  codec: string;
  channels: number;
  isDefault: boolean;
  language: string | null;
}

export interface SubtitleTrackInfo {
  index: number;
  codec: string;
  language: string | null;
}

// Mirrors app/schemas.py::AnalyzeVideoResponse (POST /video/analyze). The
// uploadToken identifies the staged upload server-side so a subsequent
// POST /video/jobs can reuse it instead of re-uploading the file.
export interface AnalyzeVideoResponse {
  uploadToken: string;
  audioTracks: AudioTrackInfo[];
  subtitleTracks: SubtitleTrackInfo[];
}

// Mirrors app/schemas.py::AudioJobResponse. Unlike the image/video responses
// this one keys the id as `id` (not `jobId`), carries no `metadata`, and puts
// `stages` at the top level -- consumers must narrow before reading them.
export interface AudioJob {
  id: string;
  status: JobStatus;
  originalFilename: string;
  denoise: string | null;
  restore: string | null;
  device: string | null;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  progressPct: number | null;
  stages: JobStage[] | null;
  error: string | null;
  ownerId: string | null;
  downloadUrl: string | null;
  // Modo karaoke. Opcionales para no obligar a cada fixture a declararlos:
  // el backend los manda siempre.
  separate?: boolean;
  separationModel?: string | null;
  /** Solo en modo karaoke: downloadUrl baja la instrumental y esto la voz. */
  vocalsDownloadUrl?: string | null;
}

// Mirrors app/schemas.py::MasteringPresetResponse. La copia viaja como clave
// de traduccion: el backend no decide en que idioma se lee la pantalla.
export interface MasteringPreset {
  id: string;
  labelKey: string;
  descriptionKey: string;
  targetLufs: number;
}

// Mirrors app/schemas.py::SeparationModelResponse. `name` es nombre propio del
// modelo: se muestra tal cual, no se traduce.
export interface SeparationModel {
  id: string;
  name: string;
  installed: boolean;
  primaryStem: string;
}

export interface AudioCapabilities {
  /** Acabado profesional (EBU R128). Siempre presente: lo hace ffmpeg. */
  masteringPresets?: MasteringPreset[];
  denoiseModes: string[];
  restoreAvailable: boolean;
  restoreModes: string[];
  /** Catálogo de modelos de separación karaoke (instalados o no). */
  separationModels?: SeparationModel[];
}

// Mirrors app/schemas.py::VoiceStepResponse. La copia viaja como clave de
// traduccion: el backend es dueño de la ESTRUCTURA (que pasos hay y en que
// orden, que tiene causalidad documentada) y el frontend de la COPIA.
export interface VoiceStep {
  id: string;
  labelKey: string;
  descriptionKey: string;
  // "filter" corre dentro del filtergraph de ffmpeg; "model" corta la cadena en
  // dos pasadas porque ffmpeg no sabe invocar un modelo a mitad de camino.
  kind: string;
  defaultEnabled: boolean;
}

export interface VoiceDelivery {
  id: string;
  labelKey: string;
  descriptionKey: string;
  lufs: number;
  truePeakDb: number;
}

export interface VoiceCatalog {
  steps: VoiceStep[];
  deliveries: VoiceDelivery[];
}

// Mirrors app/schemas.py::VideoCapabilitiesResponse. Only lists engines the
// server actually has ready (ENABLE_GMFSS + models installed for "gmfss"),
// same filtering convention as AudioCapabilities.restoreModes.
export interface VideoCapabilities {
  interpEngines: string[];
}

export interface SupportedModelResponse {
  key: string;
  label: string;
  category: string;
  description: string;
  scales: number[];
}

export interface VideoProfileResponse {
  key: string;
  label: string;
  category: string;
  description: string;
  modelKey: string;
  scale: number;
  videoCodec: string;
  videoPreset: string;
  crf: number;
  keepAudio: boolean;
}

export interface EngineInfoResponse {
  engine: string;
  configuredBinary: string;
  configuredModelsDir: string;
  available: boolean;
  defaultModel: string;
  allowedScales: number[];
  supportedModels: SupportedModelResponse[];
  // Publicados por el backend para poder avisar ANTES de subir; una copia
  // escrita a mano en el frontend se desincroniza con el .env del servidor.
  maxUploadMb: number;
  maxVideoUploadMb: number;
  videoProfiles: VideoProfileResponse[];
  ffmpegAvailable: boolean;
}

export interface HealthResponse {
  status: "ok";
  engine: string;
  gpuConcurrency: number;
  queueDepth: number;
  videoQueueDepth: number;
}

// "auto" is never returned by GET /devices (real hardware only) -- it's a
// frontend-only sentinel for the synthetic "Auto" DevicePicker entry, mirrored
// server-side by app.services.devices_service.AUTO_DEVICE_ID.
export type DeviceKind = "cpu" | "gpu" | "npu" | "auto";
export type DeviceBackend = "cpu" | "directml" | "winml" | "auto";

export type DeviceEpState =
  | ""
  | "native"
  | "ready"
  | "baseline"
  | "preparing"
  | "error"
  | "cpu_fallback";

export interface DeviceInfoResponse {
  id: string;
  kind: DeviceKind;
  name: string;
  backend: DeviceBackend;
  // EP activo por dispositivo (Fase 1b): opcionales para tolerar backends
  // anteriores que no los envían.
  activeEp?: string;
  epLabel?: string;
  epState?: DeviceEpState;
  epDetail?: string;
}

export interface DevicesResponse {
  devices: DeviceInfoResponse[];
  defaultDeviceId: string;
}

export interface ModelResponse {
  id: string;
  name: string;
  kind: string;
  source: string;
  scale: number | null;
  arch: string | null;
  sizeBytes: number;
  status: string;
  error: string | null;
}

export interface ModelsResponse {
  models: ModelResponse[];
}

export type Precision = "fp16" | "fp32";
export type CompatVerdict =
  | "ready_onnx"
  | "needs_conversion"
  | "single_file"
  | "gated"
  | "incompatible";

export interface PrecisionCost {
  precision: Precision;
  downloadBytes: number;
  estimatedPeakBytes: number;
}

export interface CheckpointCandidate {
  path: string;
  sizeBytes: number;
  architecture: string | null;
  installable: boolean | null;
  reasonKey: string;
  reasonParams: Record<string, string>;
}

export interface DeviceCapacity {
  id: string;
  name: string;
  kind: string;
  freeVramBytes: number | null;
}

export interface PreflightResponse {
  repoId: string;
  compat: CompatVerdict | null;
  compatReasonKey: string | null;
  compatReasonParams: Record<string, string>;
  degraded: boolean;
  referenceWidth: number;
  referenceHeight: number;
  precisions: PrecisionCost[];
  devices: DeviceCapacity[];
  disk: { targetPath: string; freeBytes: number } | null;
  checkpoints: CheckpointCandidate[];
  freeRamBytes: number | null;
}

export interface UpscalerPreflightResponse {
  repoId: string;
  compat: CompatVerdict | null;
  compatReasonKey: string | null;
  compatReasonParams: Record<string, string>;
  degraded: boolean;
  downloadBytes: number | null;
  devices: DeviceCapacity[];
  disk: { targetPath: string; freeBytes: number } | null;
  freeRamBytes: number | null;
}

export interface HfModelSearchResultResponse {
  id: string;
  author: string | null;
  pipelineTag: string | null;
  downloads: number;
  likes: number;
  tags: string[];
  compat: CompatVerdict | null;
  compatReasonKey: string | null;
  compatReasonParams: Record<string, string>;
  availablePrecisions: Precision[];
}

export interface ModelSearchResponse {
  results: HfModelSearchResultResponse[];
}

export interface CreateInstallResponse {
  installId: string;
  statusUrl: string;
}

// Mirrors app/schemas.py::UpdateCheckResponse. Backend always answers 200:
// on any failure (offline, rate-limit, bad JSON) it returns updateAvailable
// false with a non-null error instead of throwing.
export interface UpdateCheck {
  currentVersion: string;
  latestVersion: string | null;
  updateAvailable: boolean;
  releaseUrl: string | null;
  publishedAt: string | null;
  checkedAt: string;
  error: string | null;
}

export interface InstallStatusResponse {
  installId: string;
  repoId: string;
  status: string;
  progressPct: number | null;
  modelId: string | null;
  error: string | null;
  conversionId?: string | null;
}

export interface ConversionStatusResponse {
  conversionId: string;
  repoId: string;
  status: JobStatus;
  progressPct: number | null;
  stage: string | null;
  stages: Array<{ key: string; label: string; weight: number; status: string }> | null;
  modelId: string | null;
  error: string | null;
}

export type LeverStatus = "ok" | "unavailable" | "not_applicable" | "needs_admin";

export interface LeverResponse {
  id: string;
  label: string;
  status: LeverStatus;
  detail: string;
  fixable: boolean;
}

export interface CapabilitiesResponse {
  levers: LeverResponse[];
}

export interface FixLeverResponse {
  lever: LeverResponse;
}

export interface CpuFallbackReportResponse {
  modelId: string;
  deviceId: string;
  hotOps: string[];
  clean: boolean;
}

export interface OnnxDiagnosticEntryResponse {
  modelId: string;
  deviceId: string;
  report: CpuFallbackReportResponse | null;
}

export interface OnnxDiagnosticsResponse {
  entries: OnnxDiagnosticEntryResponse[];
}

export interface ScanOnnxDiagnosticResponse {
  report: CpuFallbackReportResponse;
}

export type CapabilityDomainId = "video" | "image" | "audio" | "generate" | "print";
export type CapabilityStatus = "available" | "needs_setup" | "not_implemented";
/** `builtin` anda sin bajar nada; `none` significa NO implementada todavía. */
export type CapabilityProvisioning = "registry" | "vendored_pack" | "builtin" | "none";
export type CapabilityStrategy = "dsp" | "model";

export interface CapabilityResponse {
  id: string;
  domain: CapabilityDomainId;
  labelKey: string;
  status: CapabilityStatus;
  provisioning: CapabilityProvisioning;
  jobKind: string | null;
  strategies: CapabilityStrategy[];
  missingPacks: string[];
  unavailableReasonKey: string | null;
  setupReasonKey: string | null;
}

export interface CapabilityDomainResponse {
  domain: CapabilityDomainId;
  labelKey: string;
  capabilities: CapabilityResponse[];
  roadmap: CapabilityResponse[];
}

export interface CapabilityTreeResponse {
  domains: CapabilityDomainResponse[];
}

// Mirrors app/schemas.py::ProvisionJobResponse. Mismo shape que un install job
// a proposito, para que la UI pueda tratarlos igual.
export type ProvisionStatus = "queued" | "running" | "done" | "error";

export interface ProvisionJob {
  jobId: string;
  pack: string;
  status: ProvisionStatus;
  error: string | null;
  statusUrl: string;
}

export interface InitImageResponse {
  initImageToken: string;
  originalFilename: string;
  width: number;
  height: number;
}

export interface GenerationJob {
  id: string;
  status: JobStatus;
  prompt: string;
  negativePrompt: string | null;
  modelId: string;
  steps: number;
  guidance: number;
  width: number;
  height: number;
  seed: number | null;
  // Opcionales: backends anteriores no los envían.
  seedWasRandom?: boolean;
  device: string | null;
  executionProvider?: string | null;
  strength?: number | null;
  scheduler?: string | null;
  speedClass?: string | null;
  precision?: string | null;
  autoUpscale: boolean;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  progressPct: number | null;
  stages: JobStage[] | null;
  error: string | null;
  ownerId: string | null;
  downloadUrl: string | null;
  /** La URL de descarga no lleva extension: esto decide imagen vs reproductor. */
  isVideo?: boolean;
  /** El job termina COMPLETO aunque el agrandado falle: la imagen generada
   *  sirve igual, pero salio del tamaño generado y no del pedido. */
  upscaleError?: string | null;
}

export interface GenerationModelSummary {
  id: string;
  name: string;
  /** "installed" o "converting" — el dropdown muestra las conversiones en curso. */
  status: string;
  /** Soporte real de inpainting (chequeado contra el mapeo de clases del backend). */
  supportsInpaint?: boolean;
  /** Motor que solo sabe quitar (rellena continuando el entorno, sin prompt). */
  eraseOnly?: boolean;
  /** Clase de velocidad (turbo/lightning/lcm): el server ancla sus parámetros
   *  y el editor esconde los turbo (512px, sin negative prompt). */
  speed?: string | null;
  /** Checkpoint de inpainting dedicado (unet 9ch): solo edita con máscara. */
  inpaintOnly?: boolean;
}

export interface GenerationCapabilities {
  available: boolean;
  reason: string | null;
  models: GenerationModelSummary[];
  devices: string[];
  cpuOnly: boolean;
}

// Respuesta de POST /editor/insert-object: el composite viene inline en base64
// y, si se pidió armonizar, jobId es un job de generación normal en la cola.
export interface InsertObjectResponse {
  compositeToken: string;
  compositePngBase64: string;
  jobId: string | null;
}

export interface VideoModelSummary {
  id: string;
  name: string;
  /** Destilado: 4 pasos en vez de 20. La diferencia entre 2 minutos y 10. */
  fast: boolean;
  defaultSteps: number;
  defaultGuidance: number;
}

export interface VideoGenerationCapabilities {
  available: boolean;
  models: VideoModelSummary[];
  defaultFrames: number;
  defaultFps: number;
  maxFrames: number;
}

// Mirrors app/schemas.py::RealtimePresetResponse.
export interface RealtimePreset {
  id: string;
  labelKey: string;
  descriptionKey: string;
}

export interface RealtimeCapabilities {
  available: boolean;
  presets: RealtimePreset[];
  reason: string | null;
}

export interface RealtimeStarted {
  pid: number;
  preset: string;
}

export interface QuotaStatus {
  maxConcurrent: number;
  maxQueued: number;
  maxJobsPerDay: number;
  maxGpuSecondsPerDay: number;
  usedJobsToday: number;
  usedGpuSecondsToday: number;
}

export interface MeResponse {
  userId: string | null;
  username: string;
  role: string;
  permissions: string[];
  mustChangePassword: boolean;
  authMode: "off" | "multi";
  quota: QuotaStatus;
}

export interface UserSummary {
  id: string;
  username: string;
  role: string;
  disabled: boolean;
  mustChangePassword: boolean;
  quotaOverrides: Record<string, number>;
  createdAt: string;
  usedJobsToday: number;
  usedGpuSecondsToday: number;
}

export interface UsersListResponse {
  users: UserSummary[];
}

export interface CreateUserResponse {
  user: UserSummary;
  temporaryPassword: string;
}

export interface UpdateUserResponse {
  user: UserSummary;
  temporaryPassword: string | null;
}

export interface OwnedJobSummary {
  id: string;
  kind: string;
  status: JobStatus;
  originalFilename: string | null;
  createdAt: string;
  finishedAt: string | null;
}

export interface UserJobsResponse {
  jobs: OwnedJobSummary[];
}

export interface EditableSettingStatus {
  key: string;
  configured: boolean;
}

export interface EditableSettingsResponse {
  settings: EditableSettingStatus[];
}

// Mirrors app/schemas.py::TranscribeJobResponse. The transcription text is
// part of the polled job response; downloadUrl is only the optional .txt copy.
export interface TranscribeJob {
  id: string;
  status: JobStatus;
  originalFilename: string;
  modelId: string;
  language: string | null;
  device: string | null;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  progressPct: number | null;
  text: string | null;
  error: string | null;
  ownerId: string | null;
  downloadUrl: string | null;
  // Solo cuando el job pidio el video con subtitulos y ffmpeg ya lo dejo listo.
  videoUrl?: string | null;
  /** Cuantas lineas del doblaje no entraron en su hueco ni al maximo de
   *  velocidad. Se avisa en vez de entregar un doblaje corrido en silencio. */
  dubOverflowSegments?: number | null;
}

export type AsrInstallStatus = "queued" | "downloading" | "installed" | "error";

export interface AsrInstallStatusResponse {
  installId: string;
  repoId: string;
  status: AsrInstallStatus;
  progressPct: number | null;
  modelId: string | null;
  error: string | null;
}

/** Lo que hay en una URL, antes de comprometerse a descargarlo. */
export interface MediaProbe {
  title: string;
  durationSeconds: number | null;
  uploader: string | null;
  extractor: string;
  isPlaylist: boolean;
  entryCount: number;
  availableHeights: number[];
  /** La miniatura que vuelve reconocible un video antes de bajarlo. */
  thumbnailUrl: string | null;
}

export interface DownloadJob {
  id: string;
  status: JobStatus;
  url: string;
  maxHeight: number;
  audioOnly: boolean;
  audioFormat: string;
  audioBitrateKbps: number | null;
  videoContainer: string;
  mediaTitle: string | null;
  mediaUploader: string | null;
  extractor: string | null;
  createdAt: string;
  startedAt: string | null;
  finishedAt: string | null;
  progressPct: number | null;
  downloadedBytes: number;
  totalBytes: number | null;
  /** Solo nombres, uno por archivo. */
  outputFiles: string[];
  /** Donde quedaron. Vacio mientras no haya archivos. */
  outputDirectory: string;
  error: string | null;
  ownerId: string | null;
}
