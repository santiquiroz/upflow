"""El estudio de karaoke: la cancion entra, se revisa, y recien ahi se paga el render.

El modo karaoke de Transcribir es una caja cerrada: modelo de separacion fijo,
sin limpieza, sin mejora, sin ver la letra antes del video. Aca el trabajo corre
en DOS pasadas por la misma cola: `preparar` (separar con el modelo elegido,
limpiar, mejorar, transcribir, traducir) deja el instrumental escuchable y la
letra editable; `render` quema el ASS bilingue sobre el fondo elegido.

`status` es el estado de cada pasada; `phase` el del negocio. Un fallo del
render vuelve a `review` sin tirar el instrumental, que es lo caro.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import soundfile as sf

from app.config import APOLLO_MODE, AUDIOSR_MODE, Settings
from app.models import JobStatus, KaraokeJob
from app.services.audio_excerpt import probe_duration_seconds
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.quotas import QuotaService
from app.services.cleanup_chain import cleanup_steps_from_selection
from app.services.device_semaphores import DeviceSemaphores
from app.services.devices_service import AUTO_DEVICE_ID, DevicesService
from app.services.engines.singer_clustering import (
    SINGER_COUNT_DEFAULT,
    SINGER_COUNT_MAX,
    SINGER_COUNT_MIN,
    cluster_singers,
    mute_time_spans,
    singer_label,
)
from app.services.engines.singer_embedding import (
    SpeakerEmbedder,
    line_embeddings,
    xvector_speaker_embedder,
)
from app.services.engines.transcribe_onnx import TranscribeRequest, is_english_only
from app.services.job_manager_base import QueuedJobManager
from app.services.karaoke_subtitles import (
    KaraokeStyle,
    build_style_lines,
    hex_to_ass_color,
    line_from_segment,
    render_karaoke_ass,
)
from app.services.karaoke_video import (
    BACKGROUND_KINDS,
    build_karaoke_command,
    build_picture_probe_command,
    build_practice_mix_command,
    has_real_picture,
)
from app.services.missing_pack import missing_pack_message
from app.services.media_decode import (
    SEPARATION_CHANNELS,
    SEPARATION_SAMPLE_RATE,
    build_decode_to_wav_command,
)
from app.services.model_registry import ModelKind, ModelRegistry
from app.services.process_runner import run_guarded_process
from app.services.restorer_registry import validate_restore_mode_ready
from app.services.romanization import romanize_segments
from app.services.subtitles import segments_to_text
from app.services.translate import TranslationEngine, parse_pair
from app.services.vendor_paths import translation_dir

logger = logging.getLogger(__name__)

RESTORE_MODES = (APOLLO_MODE, AUDIOSR_MODE)

# Fases del NEGOCIO, no de la cola. `review` es la unica donde el usuario tiene
# la mano: escuchar, editar la letra y disparar el render.
PHASES = ("preparing", "review", "rendering", "completed", "failed", "cancelled")

_LANGUAGE_LENGTH = 2

# El stem del que salen los embeddings de cantante. Los modelos de karaoke lo
# emiten con este id; uno que no lo tenga (reverb_hq, KARA_2) no sirve para
# detectar cantantes y se rechaza al crear.
VOCALS_STEM_ID = "vocals"


def _discard_dir(work_dir: Path | None) -> None:
    if work_dir is None:
        return
    try:
        for archivo in work_dir.glob("*"):
            archivo.unlink(missing_ok=True)
        work_dir.rmdir()
    except OSError:
        logger.exception("no se pudo limpiar %s", work_dir)


class KaraokeJobManager(QueuedJobManager[KaraokeJob]):
    queue_full_message = "Karaoke job queue is full; try again later"
    worker_name_prefix = "karaoke-worker"

    def __init__(
        self,
        settings: Settings,
        transcribe_engine: Any,
        device_semaphores: DeviceSemaphores,
        *,
        registry: ModelRegistry,
        separators: dict[str, Any],
        restorers: dict[str, Any],
        devices: DevicesService | None = None,
        quota_service: QuotaService | None = None,
        translation: TranslationEngine | None = None,
        embedder: SpeakerEmbedder | None = None,
    ) -> None:
        super().__init__(settings, quota_service=quota_service, worker_count=1)
        self.transcribe_engine = transcribe_engine
        self.device_semaphores = device_semaphores
        self.registry = registry
        # Los MISMOS motores que audio y transcribe: montar otro juego
        # duplicaria los .onnx en VRAM.
        self.separators = separators
        self.restorers = restorers
        self.devices = devices
        self.translation = translation or TranslationEngine(translation_dir(settings))
        # El encoder de voz del pack de conversion, reusado (decision #6 del
        # spec F2a): un pack nuevo duplicaria 30 MB por la misma huella.
        self.embedder = embedder or xvector_speaker_embedder(
            settings.voice_conversion_xvector_path
        )

    # ------------------------------------------------------------ create

    async def create_job(
        self,
        *,
        source_path: Path,
        original_filename: str,
        asr_model_id: str,
        separation_model_id: str | None = None,
        cleanup_steps: list[str] | None = None,
        restore_mode: str | None = None,
        language: str | None = None,
        romanize: bool = False,
        translate_to: str | None = None,
        detect_singers: bool = False,
        singer_count: int | None = None,
        device: str | None = None,
        job_id: str | None = None,
        owner: AuthenticatedUser | None = None,
    ) -> KaraokeJob:
        # TODO se valida al crear: descubrir un paso invalido despues de
        # separar y transcribir seria tirar los minutos caros del trabajo.
        self._validate_asr_model(asr_model_id)
        self._validate_language(language)
        self._validate_language_fits_the_model(asr_model_id, language)
        self._validate_separation_model(separation_model_id)
        cleanup_steps_from_selection(list(cleanup_steps or []))
        self._validate_restore_mode(restore_mode)
        self._validate_translation(language, translate_to)
        self._validate_singer_detection(detect_singers, singer_count, separation_model_id)
        await self._validate_device(device)
        if owner is not None and self.quota_service is not None:
            self.quota_service.check_admission(owner)

        job = KaraokeJob(
            source_path=source_path,
            original_filename=original_filename,
            asr_model_id=asr_model_id,
            separation_model_id=separation_model_id,
            cleanup_steps=list(cleanup_steps or []),
            restore_mode=restore_mode,
            language=language,
            romanize=romanize,
            translate_to=translate_to,
            detect_singers=detect_singers,
            singer_count=(
                singer_count if singer_count is not None else SINGER_COUNT_DEFAULT
            ),
            device=device,
            owner_id=owner.id if owner is not None else None,
        )
        if job_id is not None:
            job.id = job_id
        self.jobs[job.id] = job
        self._enqueue(job)
        return job

    def _validate_asr_model(self, model_id: str) -> None:
        entry = self.registry.get(model_id)
        if entry is None or entry.kind != ModelKind.asr_onnx:
            raise ValueError(f"Unknown speech recognition model: {model_id!r}")

    def _validate_language(self, language: str | None) -> None:
        if language is None:
            return
        if len(language) != _LANGUAGE_LENGTH or not language.isalpha():
            raise ValueError(
                f"language must be a two-letter ISO 639-1 code, got {language!r}"
            )

    def _validate_language_fits_the_model(self, model_id: str, language: str | None) -> None:
        if language is None or language.lower() == "en":
            return
        entry = self.registry.get(model_id)
        nombre = entry.name if entry is not None else model_id
        if is_english_only(nombre) or is_english_only(model_id):
            raise ValueError(
                f"El modelo {nombre} solo entiende ingles, y pediste otro idioma. "
                "Elegi un modelo multilingue (los que NO terminan en .en) o "
                "cambia el idioma a ingles."
            )

    def _validate_separation_model(self, model_id: str | None) -> None:
        from app.services.engines.separation_models import SEPARATION_MODELS

        if model_id is None:
            return
        spec = SEPARATION_MODELS.get(model_id)
        if spec is None:
            known = ", ".join(sorted(SEPARATION_MODELS))
            raise ValueError(
                f"Modelo de separacion desconocido: {model_id!r}. Validos: {known}."
            )
        if self.separators.get(spec.architecture) is None:
            raise ValueError(
                f"El modelo {model_id!r} necesita el motor {spec.architecture!r} "
                "y el servidor se construyo sin el."
            )

    def _validate_restore_mode(self, mode: str | None) -> None:
        if mode is None:
            return
        if mode not in RESTORE_MODES:
            raise ValueError(
                f"Modo de mejora desconocido: {mode!r}. Validos: {', '.join(RESTORE_MODES)}."
            )
        validate_restore_mode_ready(self.settings, mode)
        if self.restorers.get(mode) is None:
            raise ValueError(f"El servidor se construyo sin el motor de mejora {mode!r}.")

    def _validate_translation(self, language: str | None, translate_to: str | None) -> None:
        if translate_to is None:
            return
        if language is None:
            # La traduccion necesita saber DESDE que idioma: la deteccion
            # automatica recien conoce el idioma al final de la transcripcion.
            raise ValueError(
                "Para traducir la letra elegi el idioma del audio: la "
                "deteccion automatica no alcanza para elegir el modelo de "
                "traduccion antes de empezar."
            )
        pair = parse_pair(language, translate_to)
        if not self.translation.available(pair):
            raise ValueError(
                f"El par de traduccion {pair.source}->{pair.target} no esta "
                "instalado. Instalalo desde Transcribir (doblaje) o elegi otro idioma."
            )

    def _validate_singer_detection(
        self,
        detect_singers: bool,
        singer_count: int | None,
        separation_model_id: str | None,
    ) -> None:
        if not detect_singers:
            if singer_count is not None:
                raise ValueError(
                    "singer_count solo tiene sentido con detect_singers activado."
                )
            return
        count = singer_count if singer_count is not None else SINGER_COUNT_DEFAULT
        if not SINGER_COUNT_MIN <= count <= SINGER_COUNT_MAX:
            raise ValueError(
                f"singer_count tiene que estar entre {SINGER_COUNT_MIN} y "
                f"{SINGER_COUNT_MAX}; llego {count}."
            )
        self._validate_model_emits_vocals(separation_model_id)
        if not self.embedder.available():
            raise ValueError(
                missing_pack_message(
                    "voice-conversion",
                    detail="La deteccion de cantantes usa su encoder de voz.",
                )
            )

    def _validate_model_emits_vocals(self, separation_model_id: str | None) -> None:
        from app.services.engines.separation_models import (
            DEFAULT_SEPARATION_MODEL,
            SEPARATION_MODELS,
        )

        spec = SEPARATION_MODELS[separation_model_id or DEFAULT_SEPARATION_MODEL]
        if VOCALS_STEM_ID not in spec.stem_ids():
            raise ValueError(
                f"El modelo {spec.id!r} no emite el stem {VOCALS_STEM_ID!r}, y la "
                "deteccion de cantantes necesita las voces solas para sacar la huella."
            )

    async def _validate_device(self, device: str | None) -> None:
        if device is None:
            return
        if device == AUTO_DEVICE_ID:
            raise ValueError(
                "device 'auto' is not supported for karaoke jobs; "
                "pin a concrete device (cpu|dml:N)"
            )
        if self.devices is not None:
            await asyncio.to_thread(self.devices.validate, device)

    # ------------------------------------------------------------ review

    def update_lyrics(
        self,
        job_id: str,
        lines: list[dict[str, Any]],
        singers: list[dict[str, Any]] | None = None,
    ) -> KaraokeJob:
        """Reemplaza texto, traduccion y/o cantante por indice, en `review`.

        Editar el texto descarta los tiempos POR PALABRA de esa linea: las
        palabras nuevas no son las que el modelo cronometro, y repartir por
        letras (el fallback que ya existe) es honesto; mantener tiempos de
        palabras que ya no estan no lo seria. Reasignar el cantante NO toca
        el texto ni sus tiempos.

        `singers` renombra cantantes y es un reemplazo TOTAL de la lista de
        renombres (contrato F2a): lo que no venga vuelve a su id default.
        """
        job = self._job_in_review(job_id)
        segmentos = list(job.segments)
        traducciones = list(job.translated_lines)
        cantantes = list(job.line_singers)
        while len(traducciones) < len(segmentos):
            traducciones.append("")
        for cambio in lines:
            index = int(cambio["index"])
            if not 0 <= index < len(segmentos):
                raise ValueError(f"No hay linea {index}: el trabajo tiene {len(segmentos)}.")
            nuevo_texto = cambio.get("text")
            if nuevo_texto is not None and nuevo_texto != segmentos[index].text:
                segmentos[index] = replace(segmentos[index], text=nuevo_texto, words=())
            nueva_traduccion = cambio.get("translation")
            if nueva_traduccion is not None:
                traducciones[index] = nueva_traduccion
            nuevo_cantante = cambio.get("singer")
            if nuevo_cantante is not None:
                self._validate_known_singer(job, str(nuevo_cantante))
                cantantes[index] = str(nuevo_cantante)
        renombres = (
            self._validated_singer_names(job, singers) if singers is not None else None
        )
        job.segments = segmentos
        job.translated_lines = traducciones
        job.line_singers = cantantes
        if renombres is not None:
            job.singer_names = renombres
        return job

    def _known_singer_ids(self, job: KaraokeJob) -> tuple[str, ...]:
        if not job.detect_singers:
            return ()
        return tuple(singer_label(i) for i in range(job.singer_count))

    def _validate_known_singer(self, job: KaraokeJob, singer_id: str) -> None:
        conocidos = self._known_singer_ids(job)
        if not conocidos:
            raise ValueError(
                "Este trabajo no detecto cantantes: crealo con detect_singers "
                "para poder asignarlos."
            )
        if singer_id not in conocidos:
            raise ValueError(
                f"Cantante desconocido: {singer_id!r}. Validos: {', '.join(conocidos)}."
            )

    def _validated_singer_names(
        self, job: KaraokeJob, singers: list[dict[str, Any]]
    ) -> dict[str, str]:
        renombres: dict[str, str] = {}
        for cantante in singers:
            singer_id = str(cantante.get("id") or "")
            self._validate_known_singer(job, singer_id)
            label = str(cantante.get("label") or "").strip()
            if not label:
                raise ValueError(f"El cantante {singer_id!r} necesita un nombre no vacio.")
            renombres[singer_id] = label
        return renombres

    def request_render(
        self,
        job_id: str,
        *,
        background_kind: str = "generated",
        background_path: Path | None = None,
        subtitle_size: str = "medium",
        subtitle_position: str = "bottom",
        subtitle_color: str = "#FFFF00",
        subtitle_highlight_color: str = "#FFFFFF",
        singer_colors: dict[str, str] | None = None,
        mute_singer: str | None = None,
    ) -> KaraokeJob:
        """Valida los parametros del render y re-encola el trabajo."""
        job = self._job_in_review(job_id)
        if background_kind not in BACKGROUND_KINDS:
            raise ValueError(
                f"Fondo desconocido: {background_kind!r}. "
                f"Validos: {', '.join(BACKGROUND_KINDS)}."
            )
        if background_kind in ("image", "video") and background_path is None:
            raise ValueError(f"El fondo {background_kind!r} necesita un archivo.")
        # El estilo se valida ACA, construyendolo: un color invalido recien en
        # el render seria descubrirlo despues de la unica espera larga.
        build_style_lines(
            KaraokeStyle(
                size=subtitle_size,
                position=subtitle_position,
                base_color=subtitle_color,
                highlight_color=subtitle_highlight_color,
            )
        )
        colores = dict(singer_colors or {})
        for singer_id, color in colores.items():
            self._validate_known_singer(job, singer_id)
            hex_to_ass_color(color)  # mismo criterio eager que el estilo
        if mute_singer is not None:
            self._validate_known_singer(job, mute_singer)
            if job.vocals_path is None or not job.vocals_path.exists():
                raise ValueError(
                    "El stem de voz de este trabajo ya no esta en disco: "
                    "no se puede mutear por cantante."
                )
        job.background_kind = background_kind
        job.background_path = background_path
        job.subtitle_size = subtitle_size
        job.subtitle_position = subtitle_position
        job.subtitle_color = subtitle_color
        job.subtitle_highlight_color = subtitle_highlight_color
        job.singer_colors = colores
        job.mute_singer = mute_singer
        job.phase = "rendering"
        job.status = JobStatus.queued
        job.error = None
        job.started_at = None
        job.finished_at = None
        job.progress_pct = None
        self._enqueue(job)
        return job

    async def validate_source_background(self, job_id: str) -> bool:
        """Si el archivo original trae imagen real que sirva de fondo."""
        job = self.jobs.get(job_id)
        if job is None:
            return False
        command = build_picture_probe_command(
            self.settings.ffprobe_binary_path, job.source_path
        )
        stdout, _stderr, returncode = await run_guarded_process(
            command, self.settings.subprocess_timeout
        )
        if returncode != 0:
            return False
        try:
            return has_real_picture(json.loads(stdout.decode("utf-8", errors="ignore")))
        except json.JSONDecodeError:
            return False

    def _job_in_review(self, job_id: str) -> KaraokeJob:
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.phase != "review":
            raise ValueError(
                f"El trabajo esta en {job.phase!r}; esto solo se puede en 'review'."
            )
        return job

    # ------------------------------------------------------------ execution

    def _admit(self, job: KaraokeJob):
        return self.device_semaphores.acquire(job.device)

    async def _run_engine(self, job: KaraokeJob) -> None:
        if job.phase == "rendering":
            await self._run_render(job)
        else:
            await self._run_prepare(job)

    def _cleanup_source(self, job: KaraokeJob) -> None:
        # En `review` el fuente sigue vivo a proposito: puede ser el fondo del
        # render. Se borra recien cuando el trabajo termina de verdad.
        if job.phase in ("review", "rendering"):
            return
        self._unlink_source_safely(job.source_path)
        _discard_dir(job.work_dir)
        job.work_dir = None

    # ------------------------------------------------------------ etapa 1

    async def _run_prepare(self, job: KaraokeJob) -> None:
        work_dir = self.settings.outputs_path / f"{job.id}.karaoke-studio"
        work_dir.mkdir(parents=True, exist_ok=True)
        job.work_dir = work_dir
        try:
            decoded = work_dir / "mezcla.wav"
            await self._decode_for_separation(job.source_path, decoded)
            job.progress_pct = 10.0
            instrumental = await self._separate(job, work_dir, decoded)
            decoded.unlink(missing_ok=True)
            job.progress_pct = 50.0
            instrumental = await self._cleanup(job, work_dir, instrumental)
            job.progress_pct = 65.0
            instrumental = await self._restore(job, work_dir, instrumental)
            job.progress_pct = 80.0
            await self._transcribe(job)
            self._translate(job)
            if job.romanize:
                job.segments = romanize_segments(list(job.segments))
            if job.detect_singers:
                await self._assign_singers(job)
            job.instrumental_path = await self._encode_instrumental(work_dir, instrumental)
            job.progress_pct = 100.0
            job.phase = "review"
        except asyncio.CancelledError:
            job.phase = "cancelled"
            raise
        except Exception:
            job.phase = "failed"
            raise

    async def _decode_for_separation(self, source: Path, destination: Path) -> None:
        command = build_decode_to_wav_command(
            ffmpeg=str(self.settings.ffmpeg_binary_path),
            source=source,
            destination=destination,
            sample_rate=SEPARATION_SAMPLE_RATE,
            channels=SEPARATION_CHANNELS,
        )
        await self._run_process(command, "decodificar el audio")

    async def _separate(self, job: KaraokeJob, work_dir: Path, decoded: Path) -> Path:
        from app.services.engines.separation_models import (
            DEFAULT_SEPARATION_MODEL,
            SEPARATION_MODELS,
        )

        spec = SEPARATION_MODELS[job.separation_model_id or DEFAULT_SEPARATION_MODEL]
        separator = self.separators.get(spec.architecture)
        if separator is None:
            raise RuntimeError(
                f"El karaoke necesita el motor {spec.architecture!r} "
                "y el servidor se construyo sin el."
            )
        stem_wavs = [work_dir / f"{stem.id}.wav" for stem in spec.stems]
        await separator.run(
            decoded,
            stem_wavs,
            job.device or self.settings.default_device,
            model_id=spec.id,
        )
        principal = work_dir / f"{spec.main_stem.id}.wav"
        # Con deteccion de cantantes el stem vocal SOBREVIVE al descarte: los
        # embeddings salen de ahi y el render con mute lo gatea. Muere con la
        # carpeta de trabajo cuando el trabajo termina de verdad.
        retenido = work_dir / f"{VOCALS_STEM_ID}.wav" if job.detect_singers else None
        for sobrante in stem_wavs:
            if sobrante != principal and sobrante != retenido:
                sobrante.unlink(missing_ok=True)
        if retenido is not None and retenido != principal:
            job.vocals_path = retenido
        return principal

    async def _cleanup(self, job: KaraokeJob, work_dir: Path, current: Path) -> Path:
        from app.services.engines.separation_models import SEPARATION_MODELS

        steps = cleanup_steps_from_selection(job.cleanup_steps)
        device = job.device or self.settings.default_device
        for index, step in enumerate(steps):
            spec = SEPARATION_MODELS[step.model_id]
            separator = self.separators.get(spec.architecture)
            if separator is None:
                raise RuntimeError(
                    f"La limpieza {step.model_id!r} necesita el motor "
                    f"{spec.architecture!r} y el servidor se construyo sin el."
                )
            clean = work_dir / f"cleanup-{index}-{step.model_id}.wav"
            removed = work_dir / f"cleanup-{index}-{step.model_id}-removed.wav"
            await separator.run(current, (clean, removed), device, model_id=step.model_id)
            removed.unlink(missing_ok=True)
            if current != clean:
                current.unlink(missing_ok=True)
            current = clean
        return current

    async def _restore(self, job: KaraokeJob, work_dir: Path, current: Path) -> Path:
        if job.restore_mode is None:
            return current
        restorer = self.restorers.get(job.restore_mode)
        if restorer is None:
            raise RuntimeError(
                f"El servidor se construyo sin el motor de mejora {job.restore_mode!r}."
            )
        restored = work_dir / f"restored-{job.restore_mode}.wav"
        await restorer.run(current, restored, job.device or self.settings.default_device)
        current.unlink(missing_ok=True)
        return restored

    async def _transcribe(self, job: KaraokeJob) -> None:
        entry = self.registry.get(job.asr_model_id)
        models_root = self.settings.models_path.resolve()
        target = (self.settings.models_path / (entry.file_path or "")).resolve()
        if not target.is_relative_to(models_root):
            raise RuntimeError(f"Model path escapes models directory: {entry.file_path!r}")
        if not target.is_dir():
            raise RuntimeError(f"Model folder missing on disk: {entry.file_path!r}")

        def on_progress(done: int, total: int) -> None:
            # La transcripcion es el tramo 80→100 de la etapa de preparacion.
            job.progress_pct = round(80 + done / max(total, 1) * 20, 1)

        job.segments = await self.transcribe_engine.run(
            model_id=job.asr_model_id,
            model_dir=target,
            audio_path=job.source_path,
            request=TranscribeRequest(language=job.language),
            device=job.device or self.settings.default_device,
            progress_cb=on_progress,
        )

    def _translate(self, job: KaraokeJob) -> None:
        if job.translate_to is None or job.language is None:
            job.translated_lines = []
            return
        # Se traduce el texto ORIGINAL, antes de la romanizacion: el modelo de
        # traduccion espera japones real, no romaji.
        pair = parse_pair(job.language, job.translate_to)
        textos = [getattr(s, "text", "") for s in job.segments]
        job.translated_lines = self.translation.translate(textos, pair)

    async def _encode_instrumental(self, work_dir: Path, instrumental: Path) -> Path:
        """El instrumental queda en FLAC: lossless para el render y liviano
        para servirlo al navegador durante la revision."""
        destino = work_dir / "instrumental.flac"
        command = [
            str(self.settings.ffmpeg_binary_path),
            "-y",
            "-i",
            str(instrumental),
            "-c:a",
            "flac",
            str(destino),
        ]
        await self._run_process(command, "comprimir el instrumental")
        instrumental.unlink(missing_ok=True)
        return destino

    async def _assign_singers(self, job: KaraokeJob) -> None:
        """Huella por linea sobre el stem vocal + clustering a s1..sN.

        Las ventanas salen de los tiempos de la transcripcion (hecha sobre la
        mezcla, decision #7 del spec F2a); el audio, del stem vocal retenido.
        Corre en un thread: el encoder es ONNX en CPU y bloquearia el loop.
        """
        if job.vocals_path is None or not job.vocals_path.exists():
            raise RuntimeError(
                "Se pidio detectar cantantes pero el stem vocal no se retuvo."
            )
        spans = [(float(s.start), float(s.end)) for s in job.segments]
        if not spans:
            job.line_singers = []
            return
        vocals_path = job.vocals_path
        embedder = self.embedder
        singer_count = job.singer_count

        def calcular() -> list[str]:
            audio, rate = sf.read(vocals_path, dtype="float32", always_2d=True)
            huellas = line_embeddings(embedder, audio, rate, spans)
            return cluster_singers(huellas, spans, singer_count)

        job.line_singers = await asyncio.to_thread(calcular)

    # ------------------------------------------------------------ etapa 2

    async def _run_render(self, job: KaraokeJob) -> None:
        # Un intento anterior con mute pudo dejar una mezcla vieja apuntada: el
        # render vigente decide de nuevo si hay pista de practica o no.
        job.practice_audio_path = None
        try:
            subtitles = self._write_subtitles(job)
            pista = job.instrumental_path
            if job.mute_singer is not None:
                pista = await self._build_practice_mix(job)
            duration = await probe_duration_seconds(pista, self.settings) or 0.0
            background = (
                job.source_path if job.background_kind == "source" else job.background_path
            )
            destination = self.settings.outputs_path / f"{job.id}.karaoke.mp4"
            command = build_karaoke_command(
                ffmpeg=str(self.settings.ffmpeg_binary_path),
                background_kind=job.background_kind,
                background=background,
                duration_seconds=duration,
                instrumental=pista,
                subtitles=subtitles,
                destination=destination,
            )
            await self._run_process(command, "armar el video de karaoke")
            if not (destination.exists() and destination.stat().st_size > 0):
                raise RuntimeError("El render termino sin producir el video.")
            job.output_path = destination
            job.phase = "completed"
            self._discard_background_upload(job)
        except asyncio.CancelledError:
            job.phase = "review"
            raise
        except Exception:
            # El instrumental y la letra sobreviven: lo unico que fallo es el
            # armado final, y eso se reintenta gratis desde `review`.
            job.phase = "review"
            raise

    async def _build_practice_mix(self, job: KaraokeJob) -> Path:
        """Instrumental + voces con las lineas del cantante muteadas.

        Queda como artefacto en outputs y no en work_dir: ES la pista de
        ensayo, se descarga despues de completar, y work_dir muere antes.
        """
        spans = [
            (float(segmento.start), float(segmento.end))
            for segmento, cantante in zip(job.segments, job.line_singers)
            if cantante == job.mute_singer
        ]
        gated = job.work_dir / "vocals-gated.wav"
        vocals_path = job.vocals_path

        def gatear() -> None:
            audio, rate = sf.read(vocals_path, dtype="float32", always_2d=True)
            sf.write(gated, mute_time_spans(audio, rate, spans), rate, subtype="FLOAT")

        await asyncio.to_thread(gatear)
        destino = self.settings.outputs_path / f"{job.id}.practice.flac"
        command = build_practice_mix_command(
            ffmpeg=str(self.settings.ffmpeg_binary_path),
            instrumental=job.instrumental_path,
            vocals=gated,
            destination=destino,
        )
        await self._run_process(command, "armar la mezcla de practica")
        if not (destino.exists() and destino.stat().st_size > 0):
            raise RuntimeError("La mezcla de practica salio vacia.")
        gated.unlink(missing_ok=True)
        job.practice_audio_path = destino
        return destino

    def _write_subtitles(self, job: KaraokeJob) -> Path:
        lineas = [
            line_from_segment(segmento)
            for segmento in job.segments
            if getattr(segmento, "text", "").strip()
        ]
        traducciones = [
            (job.translated_lines[i] if i < len(job.translated_lines) else "")
            for i, segmento in enumerate(job.segments)
            if getattr(segmento, "text", "").strip()
        ]
        cantantes = [
            (job.line_singers[i] if i < len(job.line_singers) else "")
            for i, segmento in enumerate(job.segments)
            if getattr(segmento, "text", "").strip()
        ]
        estilo = KaraokeStyle(
            size=job.subtitle_size,
            position=job.subtitle_position,
            base_color=job.subtitle_color,
            highlight_color=job.subtitle_highlight_color,
        )
        destino = self.settings.outputs_path / f"{job.id}.ass"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(
            render_karaoke_ass(
                lineas,
                translations=traducciones,
                style=estilo,
                singers=cantantes,
                singer_colors=dict(job.singer_colors),
            ),
            encoding="utf-8",
        )
        return destino

    def _discard_background_upload(self, job: KaraokeJob) -> None:
        if job.background_path is not None:
            self._unlink_source_safely(job.background_path)
            job.background_path = None

    # ------------------------------------------------------------ helpers

    async def _run_process(self, command: list[str], que_fallo: str) -> None:
        _stdout, stderr, returncode = await run_guarded_process(
            command, self.settings.subprocess_timeout
        )
        if returncode != 0:
            detail = stderr.decode("utf-8", errors="ignore").strip()
            ultima = detail.splitlines()[-1] if detail else ""
            raise RuntimeError(f"No se pudo {que_fallo}: {ultima or 'ffmpeg fallo'}")

    def text_preview(self, job: KaraokeJob) -> str:
        return segments_to_text(list(job.segments))

    @staticmethod
    def _unlink_source_safely(source_path: Path) -> None:
        try:
            source_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to delete source upload %s", source_path)
