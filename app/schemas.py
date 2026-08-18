from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models import JobStatus
from app.services.capability_probe import LeverStatus
from app.services.generation_variants import Precision
from app.services.object_transfer import DEFAULT_HARMONIZE_BLEND


class CreateJobResponse(BaseModel):
    job_id: str = Field(serialization_alias="jobId")
    status: JobStatus
    status_url: str = Field(serialization_alias="statusUrl")
    download_url: str | None = Field(default=None, serialization_alias="downloadUrl")


class JobResponse(BaseModel):
    job_id: str = Field(serialization_alias="jobId")
    status: JobStatus
    original_filename: str = Field(serialization_alias="originalFilename")
    model_name: str = Field(serialization_alias="modelName")
    scale: int
    output_format: str = Field(serialization_alias="outputFormat")
    model_id: str | None = Field(default=None, serialization_alias="modelId")
    device: str | None = None
    created_at: datetime = Field(serialization_alias="createdAt")
    started_at: datetime | None = Field(default=None, serialization_alias="startedAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")
    error: str | None = None
    owner_id: str | None = Field(default=None, serialization_alias="ownerId")
    metadata: dict[str, Any] = Field(default_factory=dict)
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    download_url: str | None = Field(default=None, serialization_alias="downloadUrl")


class VideoJobResponse(BaseModel):
    job_id: str = Field(serialization_alias="jobId")
    status: JobStatus
    original_filename: str = Field(serialization_alias="originalFilename")
    model_name: str = Field(serialization_alias="modelName")
    scale: int
    output_container: str = Field(serialization_alias="outputContainer")
    video_codec: str = Field(serialization_alias="videoCodec")
    video_preset: str = Field(serialization_alias="videoPreset")
    crf: int
    keep_audio: bool = Field(serialization_alias="keepAudio")
    target_height: int | None = Field(default=None, serialization_alias="targetHeight")
    fps_multiplier: int = Field(serialization_alias="fpsMultiplier")
    target_fps: str | None = Field(default=None, serialization_alias="targetFps")
    audio_enhance: str | None = Field(default=None, serialization_alias="audioEnhance")
    audio_restore: str | None = Field(default=None, serialization_alias="audioRestore")
    audio_track_indices: list[int] | None = Field(default=None, serialization_alias="audioTrackIndices")
    keep_subtitles: bool = Field(default=False, serialization_alias="keepSubtitles")
    audio_output_format: str = Field(default="auto", serialization_alias="audioOutputFormat")
    interp_engine: str = Field(default="rife", serialization_alias="interpEngine")
    model_id: str | None = Field(default=None, serialization_alias="modelId")
    device: str | None = None
    backend: str | None = None
    video_encoder: str = Field(default="auto", serialization_alias="videoEncoder")
    created_at: datetime = Field(serialization_alias="createdAt")
    started_at: datetime | None = Field(default=None, serialization_alias="startedAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")
    error: str | None = None
    owner_id: str | None = Field(default=None, serialization_alias="ownerId")
    metadata: dict[str, Any] = Field(default_factory=dict)
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    download_url: str | None = Field(default=None, serialization_alias="downloadUrl")


class AudioStemDownloadResponse(BaseModel):
    """Una descarga por stem de un job de separacion completado. `id` es el
    valor de download?stem=; la copia viaja como clave de traduccion."""

    id: str
    label_key: str = Field(serialization_alias="labelKey")
    url: str


class AudioJobResponse(BaseModel):
    id: str
    status: JobStatus
    original_filename: str = Field(serialization_alias="originalFilename")
    denoise: str | None = None
    restore: str | None = None
    device: str | None = None
    output_format: str = Field(default="flac", serialization_alias="outputFormat")
    # Escalon de calidad de los destinos con perdida. Viaja siempre para que el
    # detalle pueda decir con que bitrate se escribio un mp3/m4a.
    lossy_quality: str = Field(default="maximum", serialization_alias="lossyQuality")
    # Acabado profesional elegido (id del catalogo de mastering). Sin esto el
    # detalle del trabajo no puede decir a que sonoridad se normalizo.
    master: str | None = None
    # Cadena de voz REALMENTE pedida (ids del catalogo) + su destino de entrega:
    # son parametros que el usuario eligio y que no viajaban a ningun lado.
    voice_steps: list[str] = Field(default_factory=list, serialization_alias="voiceSteps")
    voice_delivery: str | None = Field(default=None, serialization_alias="voiceDelivery")
    voice_presence_db: float | None = Field(
        default=None, serialization_alias="voicePresenceDb"
    )
    # Cadena de limpieza YA normalizada (orden del catalogo, sin redundancias):
    # es la cadena que realmente se va a correr, no la lista que llego.
    cleanup_steps: list[str] = Field(
        default_factory=list, serialization_alias="cleanupSteps"
    )
    separate: bool = False
    separation_model: str | None = Field(default=None, serialization_alias="separationModel")
    created_at: datetime = Field(serialization_alias="createdAt")
    started_at: datetime | None = Field(default=None, serialization_alias="startedAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    stages: list[dict[str, Any]] | None = None
    # Las decisiones que el pipeline tomo solo (masteringSkipped,
    # voiceLoudnessSkipped) y lo que midio (loudnessBefore/loudnessTarget) vivian
    # en la metadata del job y no salian por ningun lado: el aviso de
    # "se salto la masterizacion" ya existia en la UI y nunca podia dispararse
    # para audio. Viaja entera, igual que en image/video.
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    owner_id: str | None = Field(default=None, serialization_alias="ownerId")
    download_url: str | None = Field(default=None, serialization_alias="downloadUrl")
    # Solo en jobs de separacion completados: UNA descarga por stem con el
    # label del catalogo, ORDENADAS (la primera es la que el usuario quiere y
    # coincide con downloadUrl). Son dos en karaoke y limpieza y cuatro en los
    # multi-stem: el cliente recorre la lista, no la desestructura.
    stems: list[AudioStemDownloadResponse] | None = None
    # Compat v0.59 (karaoke 2 stems): downloadUrl baja la instrumental y esto
    # la voz. Solo se llena cuando el modelo del job es exactamente ese par;
    # para reverb_hq y para los multi-stem usar `stems`.
    vocals_download_url: str | None = Field(
        default=None, serialization_alias="vocalsDownloadUrl"
    )


class TranslationPairResponse(BaseModel):
    source: str
    target: str


class TranslationPairsResponse(BaseModel):
    pairs: list[TranslationPairResponse]
    # Los que se PUEDEN bajar. Sin esto, con cero pares instalados la pantalla
    # escondia la traduccion entera y el usuario no tenia como conseguirla.
    installable: list[str] = Field(default_factory=list)


class VoiceConversionCapabilitiesResponse(BaseModel):
    available: bool
    reason: str | None = None
    max_seconds: int = Field(serialization_alias="maxSeconds")
    missing_pack: str | None = Field(default=None, serialization_alias="missingPack")


class PrinterResponse(BaseModel):
    id: str
    bed_mm: tuple[float, float, float] = Field(serialization_alias="bedMm")


class PrintersResponse(BaseModel):
    printers: list[PrinterResponse]


class PrintCheckResponse(BaseModel):
    can_print: bool = Field(serialization_alias="canPrint")
    size_mm: tuple[float, float, float] = Field(serialization_alias="sizeMm")
    triangle_count: int = Field(serialization_alias="triangleCount")
    watertight: bool
    manifold: bool
    volume_mm3: float | None = Field(default=None, serialization_alias="volumeMm3")
    overhang_ratio: float = Field(serialization_alias="overhangRatio")
    # Lo que impide imprimir. Vacio = se puede.
    blockers: list[str] = Field(default_factory=list)
    # Lo que se imprime igual pero saldria mejor de otra forma.
    advice: list[str] = Field(default_factory=list)


class MeshRepairResponse(BaseModel):
    can_print: bool = Field(serialization_alias="canPrint")
    watertight: bool
    manifold: bool
    triangle_count: int = Field(serialization_alias="triangleCount")
    volume_mm3: float | None = Field(default=None, serialization_alias="volumeMm3")
    blockers: list[str] = Field(default_factory=list)
    # La malla reparada, para bajarla. Se entrega igual cuando NO quedo cerrada:
    # el usuario decide si le sirve, y el reporte le dice la verdad.
    download_url: str = Field(serialization_alias="downloadUrl")


class PartParamResponse(BaseModel):
    name: str
    label_key: str = Field(serialization_alias="labelKey")
    default: float
    minimum: float


class PartKindResponse(BaseModel):
    id: str
    label_key: str = Field(serialization_alias="labelKey")
    description_key: str = Field(serialization_alias="descriptionKey")
    params: list[PartParamResponse]


class PartKindsResponse(BaseModel):
    kinds: list[PartKindResponse]


class GeneratePartRequest(BaseModel):
    kind: str
    params: dict[str, float]
    printer: str = "ender-3"


class GeneratedPartResponse(BaseModel):
    can_print: bool = Field(serialization_alias="canPrint")
    size_mm: tuple[float, float, float] = Field(serialization_alias="sizeMm")
    volume_mm3: float | None = Field(default=None, serialization_alias="volumeMm3")
    triangle_count: int = Field(serialization_alias="triangleCount")
    overhang_ratio: float = Field(serialization_alias="overhangRatio")
    blockers: list[str] = Field(default_factory=list)
    advice: list[str] = Field(default_factory=list)
    # El archivo se entrega SIEMPRE: si no entra en esa cama, la pieza igual esta
    # bien hecha y el usuario capaz tiene otra impresora.
    download_url: str = Field(serialization_alias="downloadUrl")


class Shape3dJobRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # Vacio solo en "photo": ahi la entrada es la foto, no una descripcion.
    prompt: str = ""
    printer: str = "ender-3"
    # "mesh" = forma sin cotas (Shap-E). "photo" = interpretacion de una foto
    # (Shap-E img2img), tampoco con cotas. "cad" = codigo OpenSCAD con cotas.
    source: str = "mesh"
    # Solo en "photo": token de una foto ya subida con POST /generation/init-image.
    # Se sube aparte para no volver multipart este contrato JSON, igual que el
    # flujo de generacion de imagenes.
    image_token: str | None = Field(default=None, alias="imageToken")
    # En "mesh" y "photo": a cuanto escalar el lado mas largo.
    target_mm: float | None = Field(default=None, serialization_alias="targetMm")
    # De donde salio `target_mm`. Lo declara el cliente porque es el unico que
    # sabe si el usuario escribio la medida o acepto la sugerida: mirando el
    # numero no hay forma de distinguirlas. Solo "user" o "estimate" — el
    # "default" lo pone el servidor cuando no llega medida.
    target_mm_source: str | None = Field(default=None, alias="targetMmSource")
    # Contra que objeto comparo el modelo al estimar. Se guarda solo si la medida
    # vino de una estimacion.
    target_mm_reference: str | None = Field(default=None, alias="targetMmReference")
    # Solo en "cad": lo que la pieza TIENE que medir. Si no coincide, el error
    # vuelve al modelo en vez de entregar algo que no entra.
    expected_size: tuple[float, float, float] | None = Field(
        default=None, serialization_alias="expectedSize"
    )


class Shape3dJobResponse(BaseModel):
    id: str
    status: JobStatus
    prompt: str
    printer: str
    created_at: datetime = Field(serialization_alias="createdAt")
    started_at: datetime | None = Field(default=None, serialization_alias="startedAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")
    can_print: bool | None = Field(default=None, serialization_alias="canPrint")
    size_mm: tuple[float, float, float] | None = Field(default=None, serialization_alias="sizeMm")
    triangle_count: int | None = Field(default=None, serialization_alias="triangleCount")
    blockers: list[str] = Field(default_factory=list)
    advice: list[str] = Field(default_factory=list)
    error: str | None = None
    source: str = "mesh"
    # A cuanto se escalo el lado mas largo, y DE DONDE salio esa medida
    # ("user" / "estimate" / "default"). Las dos viajan juntas: el numero solo
    # dejaria creer que alguien lo eligio cuando puede ser el relleno del
    # programa.
    target_mm: float | None = Field(default=None, serialization_alias="targetMm")
    target_mm_source: str | None = Field(default=None, serialization_alias="targetMmSource")
    target_mm_reference: str | None = Field(
        default=None, serialization_alias="targetMmReference"
    )
    # Solo en "cad": el codigo, que es la pieza EDITABLE.
    code: str | None = None
    retries: int = 0
    download_url: str | None = Field(default=None, serialization_alias="downloadUrl")
    # Quien lo pidio. Sin esto, el listado con ?all=true le muestra al admin
    # trabajos ajenos sin decirle de quien son.
    owner_id: str | None = Field(default=None, serialization_alias="ownerId")


class Shape3dJobsListResponse(BaseModel):
    jobs: list[Shape3dJobResponse] = Field(default_factory=list)


class SizeEstimateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # Que es el objeto. En el carril de foto sale del nombre del archivo, que es
    # la unica senal que hay cuando no se escribio nada.
    prompt: str = ""


class SizeEstimateResponse(BaseModel):
    """Una SUGERENCIA de tamano, no una cota: quien la pide la muestra y espera.

    Que la respuesta no traiga campo de "aplicar" no es casual — aplicarla es
    decision del usuario, y el servidor no tiene con que tomarla.
    """

    longest_mm: float = Field(serialization_alias="longestMm")
    # Contra que objeto la comparo el modelo. Puede venir vacio: un numero sin
    # referencia sigue sirviendo, y fingir una referencia seria peor.
    reference: str = ""


class SavedPromptResponse(BaseModel):
    id: str
    name: str
    # Dato del usuario, no copia: viaja literal y no lleva clave de traduccion.
    prompt: str
    negative_prompt: str = Field(default="", serialization_alias="negativePrompt")
    mode: str


class SavedPromptsResponse(BaseModel):
    prompts: list[SavedPromptResponse]


class CreateSavedPromptRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str = Field(default="", max_length=4000, alias="negativePrompt")
    mode: str = "text-to-image"

    model_config = {"populate_by_name": True}


class TtsCapabilitiesResponse(BaseModel):
    available: bool
    voices: list[str] = Field(default_factory=list)
    reason: str | None = None
    # Que paquete hay que bajar. Sin esto la pantalla puede explicar el problema
    # pero no ofrecer el boton, que es lo unico que le sirve al usuario.
    missing_pack: str | None = Field(default=None, serialization_alias="missingPack")


class SynthesizeSpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    voice: str = Field(min_length=1)
    language: str | None = None


class PromptPresetResponse(BaseModel):
    id: str
    mode: str
    # El NOMBRE es copia y viaja como clave; el PROMPT es el texto que va al
    # modelo y se manda literal, porque traducirlo cambiaria lo que genera.
    label_key: str = Field(serialization_alias="labelKey")
    prompt: str
    negative_prompt: str = Field(default="", serialization_alias="negativePrompt")


class PromptPresetsResponse(BaseModel):
    presets: list[PromptPresetResponse]


class MasteringPresetResponse(BaseModel):
    id: str
    label_key: str = Field(serialization_alias="labelKey")
    description_key: str = Field(serialization_alias="descriptionKey")
    target_lufs: float = Field(serialization_alias="targetLufs")


class SeparationStemResponse(BaseModel):
    """Un stem de un modelo de separacion. `id` es el valor que va en
    download?stem=; la copia viaja como clave de traduccion."""

    id: str
    label_key: str = Field(serialization_alias="labelKey")


class SeparationModelResponse(BaseModel):
    """Un modelo del catalogo de separacion (separation_models.SEPARATION_MODELS).

    `name` es nombre propio del modelo y viaja tal cual (no se traduce);
    `installed` sale de mirar el disco, como todas las capacidades.
    `stems` viene ORDENADO: el primero es el que el usuario quiere (el que
    sirve downloadUrl del job); `category` agrupa el picker de la UI
    ("karaoke" | "cleanup"). `architecture` es informativo: el usuario elige
    un modelo por lo que hace, no por como esta construido, y el backend
    resuelve el motor solo. `warningKey` (opcional) es lo que la UI tiene que
    mostrar ANTES de que el usuario elija el modelo, no despues.
    """

    id: str
    name: str
    installed: bool
    primary_stem: str = Field(serialization_alias="primaryStem")
    category: str
    architecture: str
    description_key: str = Field(serialization_alias="descriptionKey")
    warning_key: str | None = Field(default=None, serialization_alias="warningKey")
    # Etiqueta corta del picker. La manda el modelo y no la arma la UI: cada
    # advertencia dice algo distinto ("lento" contra "separa peor").
    badge_key: str | None = Field(default=None, serialization_alias="badgeKey")
    stems: list[SeparationStemResponse] = Field(default_factory=list)


class CleanupStepResponse(BaseModel):
    """Un paso de la cadena de limpieza (cleanup_chain.CLEANUP_CHAIN).

    La lista viene en el ORDEN DE EJECUCION, que tiene causalidad documentada:
    quitar ruido, quitar eco, quitar reverb. `family` es lo que ataca el paso y
    `covers` todas las familias que resuelve en su pasada (deecho_dereverb
    resuelve dos): dos pasos que comparten una entrada de `covers` son
    excluyentes, y esa es la regla que la UI aplica sin hard-codear ids.
    `name` es nombre propio del modelo y va tal cual; la copia viaja como clave.
    """

    id: str
    name: str
    family: str
    covers: list[str] = Field(default_factory=list)
    installed: bool
    description_key: str = Field(serialization_alias="descriptionKey")


class LossyQualityResponse(BaseModel):
    """Un escalon de calidad para los destinos con perdida.

    Viaja con los bitrates POR FORMATO y no con un texto armado: la UI escribe
    la copia y el backend es dueno de los numeros, igual que en los presets de
    mastering.
    """

    id: str
    bitrates: dict[str, str]


class AudioCapabilitiesResponse(BaseModel):
    denoise_modes: list[str] = Field(serialization_alias="denoiseModes")
    # Formatos de salida y cuales de ellos tienen perdida: con esto la UI puede
    # avisar que una conversion sin-perdida -> con-perdida es irreversible sin
    # duplicar la lista de formatos de su lado.
    output_formats: list[str] = Field(default_factory=list, serialization_alias="outputFormats")
    lossy_formats: list[str] = Field(default_factory=list, serialization_alias="lossyFormats")
    lossy_qualities: list[LossyQualityResponse] = Field(
        default_factory=list, serialization_alias="lossyQualities"
    )
    default_lossy_quality: str = Field(
        default="maximum", serialization_alias="defaultLossyQuality"
    )
    restore_available: bool = Field(serialization_alias="restoreAvailable")
    restore_modes: list[str] = Field(default_factory=list, serialization_alias="restoreModes")
    # Acabado profesional (EBU R128). Siempre disponible: lo hace ffmpeg, que ya
    # viene con la app, sin descargar nada.
    mastering_presets: list[MasteringPresetResponse] = Field(
        default_factory=list, serialization_alias="masteringPresets"
    )
    # Catalogo completo de modelos de separacion (instalados o no): la UI arma
    # el picker y los botones de descarga por modelo con esto.
    separation_models: list[SeparationModelResponse] = Field(
        default_factory=list, serialization_alias="separationModels"
    )
    # Cadena de limpieza, en ORDEN DE EJECUCION. Los mismos modelos aparecen en
    # separationModels con category "cleanup": alli se corren de a uno para
    # quedarse con los DOS stems; aca se encadenan para quedarse con uno limpio.
    cleanup_steps: list[CleanupStepResponse] = Field(
        default_factory=list, serialization_alias="cleanupSteps"
    )
    # A partir de cuantas pasadas avisar que el resultado puede sonar
    # sobreprocesado. Viaja como dato para que el umbral tenga una sola fuente.
    cleanup_overprocessing_threshold: int = Field(
        default=3, serialization_alias="cleanupOverprocessingThreshold"
    )


class VideoCapabilitiesResponse(BaseModel):
    interp_engines: list[str] = Field(default_factory=list, serialization_alias="interpEngines")


class LeverResponse(BaseModel):
    id: str
    label: str
    status: LeverStatus
    detail: str
    fixable: bool


class CapabilitiesResponse(BaseModel):
    levers: list[LeverResponse]


class FixLeverResponse(BaseModel):
    lever: LeverResponse


class AudioTrackResponse(BaseModel):
    index: int
    codec: str
    channels: int
    is_default: bool = Field(serialization_alias="isDefault")
    language: str | None = None


class SubtitleTrackResponse(BaseModel):
    index: int
    codec: str
    language: str | None = None


class AnalyzeVideoResponse(BaseModel):
    upload_token: str = Field(serialization_alias="uploadToken")
    audio_tracks: list[AudioTrackResponse] = Field(serialization_alias="audioTracks")
    subtitle_tracks: list[SubtitleTrackResponse] = Field(serialization_alias="subtitleTracks")


class SupportedModelResponse(BaseModel):
    key: str
    label: str
    category: str
    description: str
    scales: list[int]


class VideoProfileResponse(BaseModel):
    key: str
    label: str
    category: str
    description: str
    model_key: str = Field(serialization_alias="modelKey")
    scale: int
    video_codec: str = Field(serialization_alias="videoCodec")
    video_preset: str = Field(serialization_alias="videoPreset")
    crf: int
    keep_audio: bool = Field(serialization_alias="keepAudio")


class EngineInfoResponse(BaseModel):
    engine: str
    configured_binary: str = Field(serialization_alias="configuredBinary")
    configured_models_dir: str = Field(serialization_alias="configuredModelsDir")
    available: bool
    default_model: str = Field(serialization_alias="defaultModel")
    allowed_scales: list[int] = Field(serialization_alias="allowedScales")
    supported_models: list[SupportedModelResponse] = Field(serialization_alias="supportedModels")
    video_profiles: list[VideoProfileResponse] = Field(serialization_alias="videoProfiles")
    ffmpeg_available: bool = Field(serialization_alias="ffmpegAvailable")
    # El navegador necesita el limite para poder avisar ANTES de subir. Se manda
    # el del servidor y no una copia en el frontend, que se desincroniza en
    # cuanto alguien toca el .env.
    max_upload_mb: int = Field(serialization_alias="maxUploadMb")
    max_video_upload_mb: int = Field(serialization_alias="maxVideoUploadMb")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    engine: str
    gpu_concurrency: int = Field(serialization_alias="gpuConcurrency")
    queue_depth: int = Field(serialization_alias="queueDepth")
    video_queue_depth: int = Field(serialization_alias="videoQueueDepth")


class DeviceInfoResponse(BaseModel):
    id: str
    kind: Literal["cpu", "gpu", "npu"]
    name: str
    backend: Literal["cpu", "directml", "winml"]
    # EP activo por dispositivo (Fase 1b, selector read-only): nativo si un
    # plugin EP del vendor está registrado y sano, sino DirectML/CPU baseline.
    active_ep: str = Field(default="", serialization_alias="activeEp")
    ep_label: str = Field(default="", serialization_alias="epLabel")
    ep_state: Literal["", "native", "ready", "baseline", "preparing", "error", "cpu_fallback"] = Field(
        default="", serialization_alias="epState"
    )
    ep_detail: str = Field(default="", serialization_alias="epDetail")


class DevicesResponse(BaseModel):
    devices: list[DeviceInfoResponse]
    default_device_id: str = Field(serialization_alias="defaultDeviceId")


class ModelResponse(BaseModel):
    id: str
    name: str
    kind: str
    source: str
    scale: int | None = None
    arch: str | None = None
    size_bytes: int = Field(serialization_alias="sizeBytes")
    status: str
    error: str | None = None


class ModelsResponse(BaseModel):
    models: list[ModelResponse]


class HfModelSearchResultResponse(BaseModel):
    id: str
    author: str | None = None
    pipeline_tag: str | None = Field(default=None, serialization_alias="pipelineTag")
    downloads: int
    likes: int
    tags: list[str]
    # Compatibilidad DETECTADA de la metadata en vivo (siblings + gated, que ya
    # vienen en la respuesta de busqueda con full=true): cero requests extra por
    # resultado. Los DOS caminos la calculan, cada uno con su CompatStrategy;
    # available_precisions queda vacio en upscalers porque su instalador elige
    # el archivo de pesos solo.
    compat: str | None = None
    compat_reason_key: str | None = Field(
        default=None, serialization_alias="compatReasonKey"
    )
    compat_reason_params: dict[str, str] = Field(
        default_factory=dict, serialization_alias="compatReasonParams"
    )
    available_precisions: list[str] = Field(
        default_factory=list, serialization_alias="availablePrecisions"
    )


class ModelSearchResponse(BaseModel):
    results: list[HfModelSearchResultResponse]


class PrecisionCostResponse(BaseModel):
    precision: str
    download_bytes: int = Field(serialization_alias="downloadBytes")
    estimated_peak_bytes: int = Field(serialization_alias="estimatedPeakBytes")


class DeviceCapacityResponse(BaseModel):
    id: str
    name: str
    kind: str
    # null = no se pudo medir. Nunca 0: el frontend no avisa de lo que no sabe.
    free_vram_bytes: int | None = Field(default=None, serialization_alias="freeVramBytes")


class DiskCapacityResponse(BaseModel):
    target_path: str = Field(serialization_alias="targetPath")
    free_bytes: int = Field(serialization_alias="freeBytes")


class CheckpointCandidateResponse(BaseModel):
    path: str
    size_bytes: int = Field(serialization_alias="sizeBytes")
    architecture: str | None = None
    installable: bool | None = None
    reason_key: str = Field(serialization_alias="reasonKey")
    reason_params: dict[str, str] = Field(
        default_factory=dict, serialization_alias="reasonParams"
    )


class PreflightResponse(BaseModel):
    repo_id: str = Field(serialization_alias="repoId")
    compat: str | None = None
    compat_reason_key: str | None = Field(
        default=None, serialization_alias="compatReasonKey"
    )
    compat_reason_params: dict[str, str] = Field(
        default_factory=dict, serialization_alias="compatReasonParams"
    )
    degraded: bool
    reference_width: int = Field(serialization_alias="referenceWidth")
    reference_height: int = Field(serialization_alias="referenceHeight")
    precisions: list[PrecisionCostResponse]
    devices: list[DeviceCapacityResponse]
    disk: DiskCapacityResponse | None = None
    checkpoints: list[CheckpointCandidateResponse] = Field(default_factory=list)
    free_ram_bytes: int | None = Field(
        default=None,
        serialization_alias="freeRamBytes",
    )


class VoiceStepResponse(BaseModel):
    id: str
    label_key: str = Field(serialization_alias="labelKey")
    # Descripcion en lenguaje llano. SIEMPRE visible en la UI, no solo en un
    # tooltip: usar hover como unico mecanismo para informacion critica es un
    # fallo de accesibilidad conocido.
    description_key: str = Field(serialization_alias="descriptionKey")
    kind: str
    default_enabled: bool = Field(serialization_alias="defaultEnabled")


class VoiceDeliveryResponse(BaseModel):
    id: str
    label_key: str = Field(serialization_alias="labelKey")
    description_key: str = Field(serialization_alias="descriptionKey")
    lufs: float
    true_peak_db: float = Field(serialization_alias="truePeakDb")


class VoiceCatalogResponse(BaseModel):
    steps: list[VoiceStepResponse]
    deliveries: list[VoiceDeliveryResponse]


class CapabilityResponse(BaseModel):
    id: str
    domain: str
    label_key: str = Field(serialization_alias="labelKey")
    status: str
    provisioning: str
    job_kind: str | None = Field(default=None, serialization_alias="jobKind")
    strategies: list[str] = Field(default_factory=list)
    missing_packs: list[str] = Field(default_factory=list, serialization_alias="missingPacks")
    unavailable_reason_key: str | None = Field(
        default=None, serialization_alias="unavailableReasonKey"
    )
    setup_reason_key: str | None = Field(default=None, serialization_alias="setupReasonKey")
    # Ajustes que la tarjeta puede prender de un click sin mandar a nadie a
    # Ajustes ni al .env. Vacio si lo que falta es un pack, un modelo o un
    # ajuste que exige reiniciar.
    activatable_settings: list[str] = Field(
        default_factory=list, serialization_alias="activatableSettings"
    )


class CapabilityDomainResponse(BaseModel):
    domain: str
    label_key: str = Field(serialization_alias="labelKey")
    capabilities: list[CapabilityResponse] = Field(default_factory=list)
    # Las no implementadas viajan separadas para que el frontend les pueda dar el
    # encabezado de mapa de ruta sin tener que filtrar por su cuenta.
    roadmap: list[CapabilityResponse] = Field(default_factory=list)


class CapabilityTreeResponse(BaseModel):
    domains: list[CapabilityDomainResponse] = Field(default_factory=list)


class UpscalerPreflightResponse(BaseModel):
    repo_id: str = Field(serialization_alias="repoId")
    compat: str | None = None
    compat_reason_key: str | None = Field(
        default=None, serialization_alias="compatReasonKey"
    )
    compat_reason_params: dict[str, str] = Field(
        default_factory=dict, serialization_alias="compatReasonParams"
    )
    degraded: bool
    # Peso del archivo que el instalador va a bajar. None si el repo no tiene
    # ninguno, que la clasificacion ya reporta como incompatible.
    download_bytes: int | None = Field(default=None, serialization_alias="downloadBytes")
    devices: list[DeviceCapacityResponse] = Field(default_factory=list)
    disk: DiskCapacityResponse | None = None
    free_ram_bytes: int | None = Field(default=None, serialization_alias="freeRamBytes")


class CreateDownloadJobRequest(BaseModel):
    url: str
    # 1080p y no 4K por defecto: el pedido caro tiene que ser una eleccion.
    max_height: int = Field(default=1080, alias="maxHeight")
    audio_only: bool = Field(default=False, alias="audioOnly")
    audio_format: str = Field(default="mp3", alias="audioFormat")
    audio_bitrate_kbps: int | None = Field(default=None, alias="audioBitrateKbps")
    video_container: str = Field(default="mp4", alias="videoContainer")
    # Una URL de playlist es tambien una URL de video. El default es el item suelto
    # para que nadie dispare 200 descargas por pegar un link.
    include_playlist: bool = Field(default=False, alias="includePlaylist")
    playlist_limit: int = Field(default=10, alias="playlistLimit")
    subtitle_languages: list[str] = Field(default_factory=list, alias="subtitleLanguages")

    model_config = ConfigDict(populate_by_name=True)


class DownloadJobResponse(BaseModel):
    id: str
    status: JobStatus
    url: str
    max_height: int = Field(serialization_alias="maxHeight")
    audio_only: bool = Field(serialization_alias="audioOnly")
    audio_format: str = Field(default="mp3", serialization_alias="audioFormat")
    audio_bitrate_kbps: int | None = Field(default=None, serialization_alias="audioBitrateKbps")
    video_container: str = Field(default="mp4", serialization_alias="videoContainer")
    media_title: str | None = Field(default=None, serialization_alias="mediaTitle")
    media_uploader: str | None = Field(default=None, serialization_alias="mediaUploader")
    extractor: str | None = None
    created_at: datetime = Field(serialization_alias="createdAt")
    started_at: datetime | None = Field(default=None, serialization_alias="startedAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    downloaded_bytes: int = Field(default=0, serialization_alias="downloadedBytes")
    total_bytes: int | None = Field(default=None, serialization_alias="totalBytes")
    # Los nombres de archivo producidos. Son la entrada del pipeline de mejora, que es
    # el punto de tener esto adentro de Upflow y no al lado.
    output_files: list[str] = Field(default_factory=list, serialization_alias="outputFiles")
    # Donde quedaron. Es configuracion que el usuario ya controla desde Ajustes, no una
    # fuga: sin esto la UI decia el nombre del archivo y no donde buscarlo.
    output_directory: str = Field(default="", serialization_alias="outputDirectory")
    error: str | None = None
    owner_id: str | None = Field(default=None, serialization_alias="ownerId")


class DownloadJobsListResponse(BaseModel):
    jobs: list[DownloadJobResponse] = Field(default_factory=list)


class MediaProbeResponse(BaseModel):
    """Lo que hay en una URL, antes de comprometerse a descargarlo."""

    title: str
    duration_seconds: int | None = Field(default=None, serialization_alias="durationSeconds")
    uploader: str | None = None
    extractor: str
    is_playlist: bool = Field(serialization_alias="isPlaylist")
    entry_count: int = Field(serialization_alias="entryCount")
    available_heights: list[int] = Field(serialization_alias="availableHeights")
    # La miniatura es lo que vuelve reconocible un video antes de bajarlo.
    thumbnail_url: str | None = Field(default=None, serialization_alias="thumbnailUrl")


class TranscribeJobResponse(BaseModel):
    id: str
    status: JobStatus
    original_filename: str = Field(serialization_alias="originalFilename")
    model_id: str = Field(serialization_alias="modelId")
    language: str | None = None
    device: str | None = None
    # Que se pidio de salida ("text" | "video" | "video_burned" | "dubbed_video")
    # y, solo en doblaje, a que idioma. Son las dos decisiones que cambian lo que
    # el usuario recibe y no viajaban en la respuesta.
    output_mode: str = Field(default="text", serialization_alias="outputMode")
    target_language: str | None = Field(default=None, serialization_alias="targetLanguage")
    created_at: datetime = Field(serialization_alias="createdAt")
    started_at: datetime | None = Field(default=None, serialization_alias="startedAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    # El TEXTO es el resultado, a diferencia del resto de los jobs. Viaja en la
    # respuesta para que la UI no tenga que descargar un archivo para mostrarlo.
    text: str | None = None
    error: str | None = None
    owner_id: str | None = Field(default=None, serialization_alias="ownerId")
    download_url: str | None = Field(default=None, serialization_alias="downloadUrl")
    # Solo cuando el job pidio el video con subtitulos y ffmpeg ya lo dejo listo.
    video_url: str | None = Field(default=None, serialization_alias="videoUrl")
    # Cuantas lineas del doblaje no entraron en su hueco ni al maximo de
    # velocidad: se avisa en vez de entregar un doblaje corrido en silencio.
    dub_overflow_segments: int | None = Field(
        default=None, serialization_alias="dubOverflowSegments"
    )


class TranscribeJobsListResponse(BaseModel):
    jobs: list[TranscribeJobResponse] = Field(default_factory=list)


class ProvisionJobResponse(BaseModel):
    job_id: str = Field(serialization_alias="jobId")
    pack: str
    status: str
    error: str | None = None
    status_url: str = Field(serialization_alias="statusUrl")


class InstallModelRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    repo_id: str = Field(alias="repoId")
    # Solo la usa el camino de generacion; el de upscalers la ignora.
    precision: Precision | None = None
    checkpoint_path: str | None = Field(
        default=None,
        alias="checkpointPath",
        serialization_alias="checkpointPath",
    )


class CreateInstallResponse(BaseModel):
    install_id: str = Field(serialization_alias="installId")
    status_url: str = Field(serialization_alias="statusUrl")


class CreateConversionResponse(BaseModel):
    conversion_id: str = Field(serialization_alias="conversionId")
    status_url: str = Field(serialization_alias="statusUrl")


class UpdateCheckResponse(BaseModel):
    current_version: str = Field(serialization_alias="currentVersion")
    latest_version: str | None = Field(default=None, serialization_alias="latestVersion")
    update_available: bool = Field(serialization_alias="updateAvailable")
    release_url: str | None = Field(default=None, serialization_alias="releaseUrl")
    published_at: str | None = Field(default=None, serialization_alias="publishedAt")
    checked_at: datetime = Field(serialization_alias="checkedAt")
    error: str | None = None


class InstallStatusResponse(BaseModel):
    install_id: str = Field(serialization_alias="installId")
    repo_id: str = Field(serialization_alias="repoId")
    status: str
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    model_id: str | None = Field(default=None, serialization_alias="modelId")
    error: str | None = None
    conversion_id: str | None = Field(default=None, serialization_alias="conversionId")


class ConversionStatusResponse(BaseModel):
    conversion_id: str = Field(serialization_alias="conversionId")
    repo_id: str = Field(serialization_alias="repoId")
    status: JobStatus
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    stage: str | None = None
    stages: list[dict[str, Any]] | None = None
    model_id: str | None = Field(default=None, serialization_alias="modelId")
    error: str | None = None


class CreateGenerationJobRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    prompt: str = Field(min_length=1, max_length=2000)
    negative_prompt: str | None = Field(default=None, alias="negativePrompt", max_length=2000)
    model_id: str = Field(alias="modelId")
    steps: int = Field(default=25, ge=1, le=100)
    guidance: float = Field(default=7.5, ge=0, le=30)
    # 32 y no 64: el video usa 832x480. La regla de 64 para imagenes la
    # aplica el job manager, que sabe que modelo es.
    width: int = Field(default=512, ge=64, le=1024, multiple_of=32)
    height: int = Field(default=512, ge=64, le=1024, multiple_of=32)
    seed: int | None = Field(default=None, ge=0)
    # Scheduler alternativo; None = el que declara el repo del modelo.
    scheduler: Literal["lcm", "euler_a", "euler_trailing"] | None = None
    device: str | None = None
    # Token de una imagen ya subida con POST /generation/init-image. Presente =
    # imagen a imagen. Se sube aparte para no volver multipart el contrato JSON
    # de este endpoint, igual que hace el flujo de video con /video/analyze.
    init_image_token: str | None = Field(default=None, alias="initImageToken")
    # Token de la máscara de inpainting (PNG blanco=editar, negro=conservar),
    # subida con el mismo POST /generation/init-image. Requiere initImageToken.
    mask_image_token: str | None = Field(default=None, alias="maskImageToken")
    # None = default por modo: 0.85 con máscara (inpainting), 0.6 sin (img2img).
    strength: float | None = Field(default=None, gt=0, le=1)
    auto_upscale: bool = Field(default=False, alias="autoUpscale")
    upscale_model_name: str | None = Field(default=None, alias="upscaleModelName")
    upscale_scale: int | None = Field(default=None, alias="upscaleScale", ge=2, le=4)
    upscale_model_id: str | None = Field(default=None, alias="upscaleModelId")
    # Solo para modelos de video. El tope de 81 cuadros es el que entra en 16 GB
    # de VRAM sin que el decode se caiga a CPU.
    frames: int | None = Field(default=None, ge=1, le=81)
    fps: int | None = Field(default=None, ge=1, le=60)


class InitImageResponse(BaseModel):
    init_image_token: str = Field(serialization_alias="initImageToken")
    original_filename: str = Field(serialization_alias="originalFilename")
    width: int
    height: int


class InsertObjectRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # Tokens de /generation/init-image: destino, imagen origen y máscara del
    # objeto (la que devolvió /editor/segment, re-subida por el cliente).
    target_token: str = Field(alias="targetToken")
    source_token: str = Field(alias="sourceToken")
    source_mask_token: str = Field(alias="sourceMaskToken")
    # Modo reemplazo: máscara de un objeto del DESTINO (tap/MobileSAM sobre la
    # imagen base). El objeto insertado se adapta a su bbox — posición y tamaño
    # salen de ahí, x/y/width/height se ignoran — y la armonización cubre
    # también lo que sobre del objeto reemplazado.
    target_mask_token: str | None = Field(default=None, alias="targetMaskToken")
    x: int = Field(default=0, ge=-4096, le=8192)
    y: int = Field(default=0, ge=-4096, le=8192)
    width: int = Field(default=8, ge=8, le=4096)
    height: int = Field(default=8, ge=8, le=4096)
    feather_px: int = Field(default=6, alias="featherPx", ge=0, le=64)
    match_color: bool = Field(default=True, alias="matchColor")
    # Pase de armonización con inpaint 9ch a strength parcial. Requiere un
    # modelo de inpainting dedicado: uno de 4 canales fuerza strength 1.0 y
    # reinventaría el objeto recién pegado.
    harmonize: bool = False
    # Fracción del objeto que la armonización re-genera, medida desde el borde
    # hacia adentro. La máscara sale con intensidad continua (máxima en la
    # costura, casi nula en el centro) y el inpaint la aplica por difusión
    # diferencial. 1.0 = máscara uniforme, o sea el comportamiento clásico.
    harmonize_blend: float = Field(
        default=DEFAULT_HARMONIZE_BLEND, alias="harmonizeBlend", ge=0.0, le=1.0
    )
    model_id: str | None = Field(default=None, alias="modelId")
    prompt: str | None = Field(default=None, max_length=2000)
    device: str | None = None


class InsertObjectResponse(BaseModel):
    composite_token: str = Field(serialization_alias="compositeToken")
    composite_png_base64: str = Field(serialization_alias="compositePngBase64")
    job_id: str | None = Field(default=None, serialization_alias="jobId")


class GenerationJobResponse(BaseModel):
    id: str
    status: JobStatus
    prompt: str
    negative_prompt: str | None = Field(default=None, serialization_alias="negativePrompt")
    model_id: str = Field(serialization_alias="modelId")
    steps: int
    guidance: float
    width: int
    height: int
    seed: int | None = None
    # La semilla se resolvió al azar en el server: el modal la muestra igual
    # (reproducible) pero aclara que no la eligió el usuario.
    seed_was_random: bool = Field(default=False, serialization_alias="seedWasRandom")
    scheduler: str | None = None
    # Clase de velocidad del modelo (turbo/lightning/lcm) cuando el server
    # ancló parámetros por ella.
    speed_class: str | None = Field(default=None, serialization_alias="speedClass")
    # fp16/fp32 REAL del UNet cargado. fp32 = ~7x más lento (medido): el modal
    # lo muestra para que la regresión nunca sea silenciosa.
    precision: str | None = None
    device: str | None = None
    # Provider que las sesiones REALES usaron ("DirectML", "CPU (fallback)",
    # "Vulkan (sd.cpp)", EP nativo). None hasta que el job creó una sesión.
    execution_provider: str | None = Field(default=None, serialization_alias="executionProvider")
    # Solo con imagen de partida (img2img/inpaint); None en texto a imagen.
    strength: float | None = None
    auto_upscale: bool = Field(default=False, serialization_alias="autoUpscale")
    created_at: datetime = Field(serialization_alias="createdAt")
    started_at: datetime | None = Field(default=None, serialization_alias="startedAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    stages: list[dict[str, Any]] | None = None
    error: str | None = None
    owner_id: str | None = Field(default=None, serialization_alias="ownerId")
    download_url: str | None = Field(default=None, serialization_alias="downloadUrl")
    # El job termina COMPLETO aunque el agrandado falle: la imagen generada sirve
    # igual. Sin este campo el usuario recibe una imagen mas chica sin enterarse.
    upscale_error: str | None = Field(default=None, serialization_alias="upscaleError")
    # La URL de descarga no lleva extensión: sin esto la UI no sabe si el
    # resultado es una imagen o un clip que hay que reproducir.
    is_video: bool = Field(default=False, serialization_alias="isVideo")


class InstallVulkanModelRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    repo_id: str = Field(alias="repoId")
    # Archivo suelto del repo (.safetensors/.gguf): es lo que corre el lane
    # Vulkan tal cual, sin exportar nada.
    filename: str


class VulkanInstallStatusResponse(BaseModel):
    install_id: str = Field(serialization_alias="installId")
    repo_id: str = Field(serialization_alias="repoId")
    status: str
    progress_pct: float | None = Field(default=None, serialization_alias="progressPct")
    model_id: str | None = Field(default=None, serialization_alias="modelId")
    error: str | None = None


class EditorSegmentRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # Token de una imagen subida con POST /generation/init-image; coords del
    # click en píxeles de la imagen ORIGINAL.
    image_token: str = Field(alias="imageToken")
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    device: str | None = None


class GenerationModelSummary(BaseModel):
    id: str
    name: str
    # "installed" o "converting": el dropdown muestra las conversiones en curso
    # (deshabilitadas) para que una instalacion desde el installer no parezca
    # que "no trajo nada" durante los ~40 min de conversion.
    status: str = "installed"
    # Soporte REAL de inpainting: chequeado contra el mapeo de clases
    # (generation_pipeline_modes), no contra existencia de la clase. El picker filtra
    # con esto en vez de descubrir el rechazo al crear el job.
    supports_inpaint: bool = Field(default=True, serialization_alias="supportsInpaint")
    # Motores que solo saben QUITAR (rellenan continuando el entorno): se ofrecen
    # en el modo Quitar y se esconden cuando hay que poner algo concreto.
    erase_only: bool = Field(default=False, serialization_alias="eraseOnly")
    # Clase de velocidad derivada del nombre del repo (turbo/lightning/lcm).
    # La UI marca el modelo como rápido y el editor esconde los turbo (512px,
    # sin negative prompt: no sirven para inpaint).
    speed: str | None = None
    # Checkpoint de inpainting dedicado (unet 9ch): SOLO edita con máscara.
    # Generate lo deshabilita; el Editor lo prefiere (bordes que sí calzan y
    # strength parcial real).
    inpaint_only: bool = Field(default=False, serialization_alias="inpaintOnly")


class GenerationCapabilitiesResponse(BaseModel):
    available: bool
    reason: str | None = None
    models: list[GenerationModelSummary] = Field(default_factory=list)
    devices: list[str] = Field(default_factory=list)
    cpu_only: bool = Field(default=False, serialization_alias="cpuOnly")


class VideoModelSummary(BaseModel):
    id: str
    name: str
    # Los destilados generan en 4 pasos en vez de 20: la diferencia entre dos
    # minutos y diez. La UI lo marca para que se elija con conocimiento.
    fast: bool = False
    default_steps: int = Field(serialization_alias="defaultSteps")
    default_guidance: float = Field(serialization_alias="defaultGuidance")


class VideoGenerationCapabilitiesResponse(BaseModel):
    available: bool
    models: list[VideoModelSummary] = Field(default_factory=list)
    default_frames: int = Field(serialization_alias="defaultFrames")
    default_fps: int = Field(serialization_alias="defaultFps")
    max_frames: int = Field(serialization_alias="maxFrames")


class RealtimePresetResponse(BaseModel):
    id: str
    label_key: str = Field(serialization_alias="labelKey")
    description_key: str = Field(serialization_alias="descriptionKey")


class RealtimeCapabilitiesResponse(BaseModel):
    available: bool
    presets: list[RealtimePresetResponse] = Field(default_factory=list)
    # Magpie es GPL-3.0 y corre aparte; no viaja en el instalador.
    reason: str | None = None


class StartRealtimeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    preset: str
    max_frame_rate: int | None = Field(default=None, alias="maxFrameRate", ge=24, le=480)


class RealtimeStartedResponse(BaseModel):
    pid: int
    preset: str


class CpuFallbackReportResponse(BaseModel):
    model_id: str = Field(serialization_alias="modelId")
    device_id: str = Field(serialization_alias="deviceId")
    hot_ops: list[str] = Field(serialization_alias="hotOps")
    clean: bool


class OnnxDiagnosticEntryResponse(BaseModel):
    model_id: str = Field(serialization_alias="modelId")
    device_id: str = Field(serialization_alias="deviceId")
    report: CpuFallbackReportResponse | None = None


class OnnxDiagnosticsResponse(BaseModel):
    entries: list[OnnxDiagnosticEntryResponse]


class ScanOnnxDiagnosticResponse(BaseModel):
    report: CpuFallbackReportResponse


class LoginRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    current_password: str = Field(alias="currentPassword")
    new_password: str = Field(alias="newPassword", min_length=8)


class SetupRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    username: str = Field(min_length=3)
    password: str = Field(min_length=8)


class QuotaStatusResponse(BaseModel):
    max_concurrent: int = Field(serialization_alias="maxConcurrent")
    max_queued: int = Field(serialization_alias="maxQueued")
    max_jobs_per_day: int = Field(serialization_alias="maxJobsPerDay")
    max_gpu_seconds_per_day: int = Field(serialization_alias="maxGpuSecondsPerDay")
    used_jobs_today: int = Field(serialization_alias="usedJobsToday")
    used_gpu_seconds_today: float = Field(serialization_alias="usedGpuSecondsToday")


class MeResponse(BaseModel):
    user_id: str | None = Field(serialization_alias="userId")
    username: str
    role: str
    permissions: list[str]
    must_change_password: bool = Field(serialization_alias="mustChangePassword")
    auth_mode: str = Field(serialization_alias="authMode")
    quota: QuotaStatusResponse


class UserSummaryResponse(BaseModel):
    id: str
    username: str
    role: str
    disabled: bool
    must_change_password: bool = Field(serialization_alias="mustChangePassword")
    quota_overrides: dict[str, int] = Field(default_factory=dict, serialization_alias="quotaOverrides")
    created_at: datetime = Field(serialization_alias="createdAt")
    used_jobs_today: int = Field(serialization_alias="usedJobsToday")
    used_gpu_seconds_today: float = Field(serialization_alias="usedGpuSecondsToday")


class UsersListResponse(BaseModel):
    users: list[UserSummaryResponse]


class CreateUserRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    username: str = Field(min_length=3)
    role: str = Field(default="user")


class CreateUserResponse(BaseModel):
    user: UserSummaryResponse
    temporary_password: str = Field(serialization_alias="temporaryPassword")


class UpdateUserRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    role: str | None = None
    disabled: bool | None = None
    quota_overrides: dict[str, int] | None = Field(default=None, alias="quotaOverrides")
    reset_password: bool = Field(default=False, alias="resetPassword")


class UpdateUserResponse(BaseModel):
    user: UserSummaryResponse
    temporary_password: str | None = Field(default=None, serialization_alias="temporaryPassword")


class OwnedJobSummaryResponse(BaseModel):
    id: str
    kind: str
    status: JobStatus
    original_filename: str | None = Field(default=None, serialization_alias="originalFilename")
    created_at: datetime = Field(serialization_alias="createdAt")
    finished_at: datetime | None = Field(default=None, serialization_alias="finishedAt")


class UserJobsResponse(BaseModel):
    jobs: list[OwnedJobSummaryResponse]


class JobsListResponse(BaseModel):
    jobs: list[JobResponse]


class VideoJobsListResponse(BaseModel):
    jobs: list[VideoJobResponse]


class AudioJobsListResponse(BaseModel):
    jobs: list[AudioJobResponse]


class GenerationJobsListResponse(BaseModel):
    jobs: list[GenerationJobResponse]


class EditableSettingStatusResponse(BaseModel):
    key: str
    configured: bool
    # Solo para los flags booleanos: un interruptor necesita saber si esta en
    # true o false, y eso no es un secreto. Para hf_token y el texto libre viaja
    # None -- el valor nunca sale del servidor.
    value: str | None = None
    requires_restart: bool = Field(default=False, serialization_alias="requiresRestart")


class EditableSettingsResponse(BaseModel):
    settings: list[EditableSettingStatusResponse]


class UpdateSettingRequest(BaseModel):
    key: str = Field(min_length=1)
    value: str


class UpdateSettingResponse(BaseModel):
    key: str
