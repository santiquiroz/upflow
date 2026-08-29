from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from app.services.generation_variants import Precision


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


TERMINAL_JOB_STATUSES = (JobStatus.completed, JobStatus.failed, JobStatus.cancelled)


@dataclass(slots=True)
class UpscaleJob:
    source_path: Path
    original_filename: str
    model_name: str
    scale: int
    output_format: str
    model_id: str | None = None
    device: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    output_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    owner_id: str | None = None


@dataclass(slots=True, frozen=True)
class UpdateStatus:
    current_version: str
    latest_version: str | None
    update_available: bool
    release_url: str | None
    published_at: str | None
    checked_at: datetime
    error: str | None


@dataclass(slots=True)
class VideoUpscaleJob:
    source_path: Path
    original_filename: str
    model_name: str
    scale: int
    output_container: str
    video_codec: str
    video_preset: str
    crf: int
    keep_audio: bool
    # Alto de salida pedido. Presente = el usuario pidio una RESOLUCION y no un
    # multiplicador: `scale` sigue siendo el escalado entero del modelo y el
    # redimensionado final lleva a la medida exacta. Ver target_resolution.py.
    target_height: int | None = None
    fps_multiplier: int = 1
    target_fps: str | None = None
    audio_enhance: str | None = None
    audio_restore: str | None = None
    # Which audio streams to keep in the output (Fase A Task 2). None means
    # "keep_audio decides alone" (existing behavior); a list selects specific
    # ffprobe stream indices, consumed by the pipeline in Task 3.
    audio_track_indices: list[int] | None = None
    # Copy subtitle streams into the output (Fase A Task 2). Forces an mkv
    # container upgrade when the requested container can't carry subtitles
    # losslessly -- see VideoJobManager._resolve_output_container.
    keep_subtitles: bool = False
    # Elegible output codec for the (enhanced/restored) primary audio track
    # (Fase C Task 8): "auto" (default) re-encodes to lossless FLAC only when
    # a restore actually ran (mirrors _resolve_output_container's mkv
    # upgrade), "flac" always wants lossless, "aac" always forces the
    # pre-existing lossy path regardless of restore. See
    # VideoUpscaler._prepare_processed_audio.
    audio_output_format: str = "auto"
    # Frame-interpolation engine (Task 4.2): "rife" (default, always) or
    # "gmfss" (opt-in, much higher quality, 10x or more slower -- even higher
    # on short clips due to model load overhead). Only consulted when
    # interpolation is actually requested (fps_multiplier>1 or target_fps set).
    interp_engine: str = "rife"
    model_id: str | None = None
    device: str | None = None
    # Upscale runtime override (SP11): None|auto -> Auto rule; ncnn|onnx force one.
    backend: str | None = None
    # Video encoder (SP12): "auto" (default) picks a hardware encoder AMF/NVENC/QSV
    # by the job's GPU and falls back to software; "software" forces libx264/libx265.
    # Default is "auto" because software x265 slow at 4x costs ~112 min/episode vs
    # ~16 min on the GPU -- the software default was the dominant wall-time cost.
    video_encoder: str = "auto"
    # ffprobe output captured during job validation, reused by the pipeline so the
    # same file isn't probed twice. In-memory only: the API response is built field
    # by field, so this never serializes (it holds the absolute source path).
    probe: dict[str, Any] | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    output_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    owner_id: str | None = None


@dataclass(slots=True)
class AudioJob:
    source_path: Path
    original_filename: str
    denoise: str | None = None
    restore: str | None = None
    # Cadena de mejora de voz. Los pasos activos vienen como lista de ids del
    # step_catalog de voice_chain; el orden lo fija el catalogo, no el request,
    # porque el orden de la cadena tiene causalidad documentada.
    voice_steps: list[str] = field(default_factory=list)
    voice_delivery: str | None = None
    voice_presence_db: float | None = None
    # Acabado profesional: normaliza la sonoridad al estandar elegido (EBU R128).
    # None = no se toca el volumen, que es el comportamiento de siempre.
    master: str | None = None
    # Cadena de limpieza: ids de modelos de cleanup_chain.CLEANUP_CHAIN, una
    # pasada por id. Igual que voice_steps, el orden lo fija el catalogo y no el
    # request; el manager los guarda YA normalizados (ordenados y sin
    # redundancias) para que pipeline, etapas y respuesta vean lo mismo.
    # Produce UN archivo: los stems removidos de cada pasada no se guardan.
    cleanup_steps: list[str] = field(default_factory=list)
    # Modo separacion (karaoke): parte la mezcla en los stems que declare el
    # modelo elegido — dos en karaoke y limpieza, cuatro en los multi-stem — y
    # los devuelve TODOS. Exclusivo: el manager rechaza combinarlo con los
    # demas pasos porque se aplicarian a un stem ambiguo, y con cleanup_steps
    # porque una cadena produce un archivo y la separacion varios.
    separate: bool = False
    # Id del catalogo de separacion (separation_models.SEPARATION_MODELS).
    # El manager lo resuelve al default cuando separate=True y no viene.
    separation_model: str | None = None
    # Modelos EXTRA a combinar con el principal ("maxima calidad"). Vacio = un
    # solo modelo, que es el caso normal. Todos tienen que declarar los mismos
    # stems: promediar un instrumental con un bajo no da nada.
    ensemble_models: list[str] = field(default_factory=list)
    # Pistas de practica "minus-one": por cada stem elegido se hornea la
    # cancion SIN ese instrumento (mix decodificado - g*stem). Solo con
    # separate y un modelo de 3+ stems; los derivados viven en
    # stem_output_paths con id minus_<stem>.
    practice_stems: list[str] = field(default_factory=list)
    # Guia opcional (0-30): porcentaje del stem removido que queda sonando de
    # fondo como referencia. g = 1 - percent/100; 0 = quitarlo del todo.
    practice_guide_percent: int = 0
    # Transcripcion por stem (F3a): stems CON altura (nunca "drums" ni un id
    # derivado minus_*) a los que generar MIDI/MusicXML (+tab si guitar/bass).
    # Solo con separate y requiere el pack music-transcription instalado.
    transcribe_stems: list[str] = field(default_factory=list)
    # Standalone-module output format (Fase C Task 9): "wav" (lossless, no
    # re-encode -- current is already PCM from decode/denoise/restore),
    # "flac" (lossless, ~50% smaller, default), "mp3" / "m4a" (con perdida). See
    # AudioPipeline._write_output.
    output_format: str = "flac"
    # Calidad de los destinos con perdida (mp3/m4a): tres escalones con nombre,
    # no un bitrate libre. Se ignora en destinos sin perdida.
    lossy_quality: str = "maximum"
    device: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    output_path: Path | None = None
    # Solo con separate: TODOS los stems por id, en el orden del catalogo. El
    # primero es tambien output_path (el que sirve downloadUrl sin ?stem=), asi
    # que esta duplicado a proposito — este dict es la fuente de verdad de
    # ?stem=<id> y no hay ningun stem que viva solo en output_path.
    #
    # Es un dict y no un segundo Path porque los modelos multi-stem entregan
    # cuatro: con un unico `secondary_output_path` los tres ultimos se servian
    # todos desde el mismo archivo.
    stem_output_paths: dict[str, Path] = field(default_factory=dict)
    # Artefactos de transcripcion (F3a): {stem_id: {formato: Path}}. Los
    # formatos validos son "midi" | "musicxml" | "tab" (este ultimo solo para
    # guitar/bass; ausente cuando tab_writer devolvio None).
    transcription_output_paths: dict[str, dict[str, Path]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    owner_id: str | None = None


@dataclass(slots=True)
class TranscribeJob:
    """Audio a texto.

    A diferencia del resto de los jobs, el resultado es TEXTO y no un archivo de
    medios: `text` es la respuesta, y `output_path` es solo la copia en .txt para
    poder descargarla con el mismo patron que los demas.
    """

    source_path: Path
    original_filename: str
    model_id: str
    # None deja que el modelo detecte el idioma.
    language: str | None = None
    # Segmentos con tiempo. `text` es su concatenacion; los segmentos son lo que
    # permite escribir un .srt, que sin tiempos no existiria.
    segments: list[Any] = field(default_factory=list)
    # "text" entrega solo la transcripcion; "video" ademas devuelve el archivo
    # original con la pista de subtitulos adentro. El modo decide si el fuente
    # se conserva hasta el final o se borra apenas termina de transcribirse.
    output_mode: str = "text"
    # Solo en modo doblaje: a que idioma hablar. Sin esto no hay nada que doblar.
    target_language: str | None = None
    # Letra japonesa a romaji Hepburn sobre los segmentos ya cronometrados:
    # aplica parejo a la transcripcion y a cualquier subtitulo que salga de ella.
    romanize: bool = False
    # Cuantas lineas del doblaje no entraron en su hueco ni al maximo de
    # velocidad. Se cuenta para avisarlo en vez de entregar un doblaje corrido.
    dub_overflow_segments: int | None = None
    subtitled_video_path: Path | None = None
    device: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress_pct: float | None = None
    text: str | None = None
    output_path: Path | None = None
    error: str | None = None
    owner_id: str | None = None


@dataclass(slots=True)
class KaraokeJob:
    """Cancion a video de karaoke, en DOS pasadas con revision en el medio.

    `status` es el estado de CADA pasada por la cola (queued/running/...);
    `phase` es el estado del NEGOCIO (preparing → review → rendering →
    completed). La separacion importa: un fallo en el render vuelve a `review`
    sin perder el instrumental, que es lo caro.
    """

    source_path: Path
    original_filename: str
    asr_model_id: str
    separation_model_id: str | None = None
    cleanup_steps: list[str] = field(default_factory=list)
    restore_mode: str | None = None
    language: str | None = None
    romanize: bool = False
    translate_to: str | None = None
    device: str | None = None
    # Deteccion de cantantes (F2a): clustering sin supervision sobre el stem
    # vocal. `singer_count` solo tiene sentido con la deteccion prendida.
    detect_singers: bool = False
    singer_count: int = 2
    phase: str = "preparing"
    segments: list[Any] = field(default_factory=list)
    # Traducciones por INDICE de segmento; vacias donde no hubo que traducir.
    translated_lines: list[str] = field(default_factory=list)
    # Cantante por INDICE de segmento ("s1".."sN"), misma convencion que las
    # traducciones; vacia cuando no se pidio deteccion.
    line_singers: list[str] = field(default_factory=list)
    # Nombres visibles renombrados en review, por id de cantante; el id es el
    # default cuando no hay renombre.
    singer_names: dict[str, str] = field(default_factory=dict)
    instrumental_path: Path | None = None
    # El stem vocal retenido cuando hay deteccion: el render con mute lo gatea.
    vocals_path: Path | None = None
    work_dir: Path | None = None
    # Parametros de la etapa de render; llegan con el endpoint de render.
    background_kind: str = "generated"
    background_path: Path | None = None
    subtitle_size: str = "medium"
    subtitle_position: str = "bottom"
    subtitle_color: str = "#FFFF00"
    subtitle_highlight_color: str = "#FFFFFF"
    # Render per-singer (F2a): color base por cantante y cantante a mutear.
    singer_colors: dict[str, str] = field(default_factory=dict)
    mute_singer: str | None = None
    # La pista de ensayo (instrumental + voces gateadas), solo con mute.
    practice_audio_path: Path | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress_pct: float | None = None
    output_path: Path | None = None
    error: str | None = None
    owner_id: str | None = None


@dataclass(slots=True)
class Shape3dJob:
    """Texto a malla 3D.

    El resultado NO es solo el archivo: viaja con el veredicto del banco. Una
    malla generada que no cierra no es una pieza, y entregarla sin decirlo seria
    mandar a imprimir con confianza prestada.
    """

    prompt: str
    printer: str = "ender-3"
    # "mesh"  = Shap-E desde texto: FORMA sin cotas, se escala con `target_mm`.
    # "photo" = Shap-E img2img desde una foto: una INTERPRETACION del objeto,
    #           tampoco con cotas, asi que se escala igual que "mesh".
    # "cad"   = OpenSCAD escrito por un modelo, que da COTAS y por eso se
    #           verifica contra `expected_size` en vez de escalarse.
    source: str = "mesh"
    # Solo en "photo": la foto ya staged en uploads, resuelta por la API antes
    # de encolar. El prompt queda vacio en ese modo.
    image_path: Path | None = None
    # Solo en "mesh": la malla se escala para que su lado mas largo mida esto.
    target_mm: float | None = None
    # De donde salio `target_mm`: "user" (la escribio el usuario), "estimate" (la
    # sugirio el modelo y el usuario la confirmo) o "default" (la puso el
    # programa porque habia que escalar a algo). Sin esto el resultado no puede
    # decir si su medida es una decision o un relleno, y una malla que no tiene
    # cotas no se puede permitir esa ambiguedad.
    target_mm_source: str | None = None
    # Contra que objeto comparo el modelo, cuando la medida vino de una
    # estimacion. Es lo que la vuelve revisable en vez de un numero suelto.
    target_mm_reference: str | None = None
    # Solo en "cad": lo que la pieza TIENE que medir. Si no coincide, el error
    # vuelve al modelo en vez de entregar algo que no entra.
    expected_size: tuple[float, float, float] | None = None
    # Solo en "cad": el codigo, que es la pieza EDITABLE. Entregar solo el STL
    # seria entregar algo que no se puede ajustar, justo lo contrario de tener cotas.
    code: str | None = None
    retries: int = 0
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    output_path: Path | None = None
    can_print: bool | None = None
    size_mm: tuple[float, float, float] | None = None
    triangle_count: int | None = None
    blockers: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)
    error: str | None = None
    owner_id: str | None = None


@dataclass(slots=True)
class DownloadJob:
    """Traer un video o audio de una URL.

    A diferencia del resto, la entrada no es un archivo sino una URL, y la SALIDA es lo
    que despues puede entrar al pipeline de mejora: por eso `output_paths` es una lista
    (una playlist produce varios) y aterriza en el directorio de uploads.
    """

    url: str
    max_height: int = 1080
    audio_only: bool = False
    audio_format: str = "mp3"
    audio_bitrate_kbps: int | None = None
    video_container: str = "mp4"
    include_playlist: bool = False
    playlist_limit: int = 10
    subtitle_languages: list[str] = field(default_factory=list)
    # Lo que el probe encontro antes de descargar: titulo, duracion, sitio.
    media_title: str | None = None
    media_uploader: str | None = None
    extractor: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    progress_pct: float | None = None
    downloaded_bytes: int = 0
    total_bytes: int | None = None
    output_paths: list[Path] = field(default_factory=list)
    error: str | None = None
    owner_id: str | None = None
    # Separar en pistas apenas termine de bajar. Es un pedido del usuario, no una
    # etapa del motor: la descarga sigue siendo un trabajo entero por si sola.
    then_separate: bool = False
    then_separation_model: str | None = None
    # Los trabajos de separacion que disparo, para que la UI los siga.
    followup_job_ids: list[str] = field(default_factory=list)
    # Que la descarga salga bien y el encadenado no es un estado real: la descarga
    # queda completed y esto dice por que no hay pistas.
    followup_error: str | None = None


@dataclass(slots=True)
class GenerationJob:
    prompt: str
    model_id: str
    negative_prompt: str | None = None
    steps: int = 25
    guidance: float = 7.5
    width: int = 512
    height: int = 512
    seed: int | None = None
    # Scheduler alternativo ("lcm" | "euler_a" | "euler_trailing"); None = el
    # del repo. Lo puede fijar la API o el anclaje por clase de velocidad.
    scheduler: str | None = None
    device: str | None = None
    # Imagen de partida ya staged en disco. Presente = imagen a imagen.
    init_image_path: Path | None = None
    # Cuanto se aparta del original: 0 lo devuelve casi igual, 1 lo ignora.
    strength: float = 0.6
    # Pase generativo por tiles: recorre la imagen a su tamaño real en vez de
    # reducirla, para AGREGARLE detalle en vez de reinterpretarla. Inventa
    # textura que no estaba, y por eso se pide explícitamente.
    tiled_detail: bool = False
    # Máscara de inpainting (blanco=editar, negro=conservar); requiere init.
    mask_image_path: Path | None = None
    # Preparación de la máscara en el motor. None = los defaults del motor. La
    # armonización de objetos manda 0/0 porque su máscara ya viene con su propio
    # perfil continuo: volver a dilatarla empujaría el 255 de la costura hacia
    # adentro y se comería el centro que la máscara viene a preservar.
    mask_dilate_px: int | None = None
    mask_feather_px: int | None = None
    auto_upscale: bool = False
    upscale_model_name: str | None = None
    upscale_scale: int | None = None
    upscale_model_id: str | None = None
    # Solo para el lane de video: cuántos cuadros y a qué ritmo. En un job de
    # imagen quedan en None y nada los mira.
    frames: int | None = None
    fps: int | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    output_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    owner_id: str | None = None


@dataclass(slots=True)
class ConversionJob:
    repo_id: str
    precision: Precision = "fp16"
    checkpoint_path: str | None = None
    # Merge de inpainting 9ch: repo_id es el checkpoint del USUARIO; el job
    # descarga base+inpaint oficiales, mergea (add-difference) y exporta el
    # resultado como un modelo nuevo "<repo> (inpainting)".
    inpaint_merge: bool = False
    # Optimización por fusión de grafo: repo_id/checkpoint_path apuntan al origen
    # torch del modelo YA instalado `source_model_id`, del que se re-exporta el
    # UNet en fp32 para fusionarlo. El resultado se registra como variante nueva.
    optimize: bool = False
    source_model_id: str | None = None
    source_model_name: str | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    status: JobStatus = JobStatus.queued
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    model_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class InstallJob:
    id: str
    repo_id: str
    precision: Precision = "fp16"
    checkpoint_path: str | None = None
    status: Any = None
    progress_pct: float | None = None
    model_id: str | None = None
    error: str | None = None
    conversion_id: str | None = None
