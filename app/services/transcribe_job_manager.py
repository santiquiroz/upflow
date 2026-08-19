from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from app.config import Settings
from app.services.dub_mux import build_dub_mux_command
from app.services.dubbing_pipeline import DubbingPipeline, DubbingUnavailable
from app.services.engines.tts_kokoro import SAMPLE_RATE as TTS_SAMPLE_RATE
from app.services.engines.tts_kokoro import KokoroTtsEngine, available_voices
from app.services.audio_excerpt import probe_duration_seconds
from app.services.karaoke_subtitles import render_karaoke_ass, split_line_proportionally
from app.services.media_decode import (
    SEPARATION_CHANNELS,
    SEPARATION_SAMPLE_RATE,
    build_decode_to_wav_command,
)
from app.services.karaoke_video import (
    build_karaoke_command,
    build_picture_probe_command,
    has_real_picture,
)
from app.services.subtitle_burn import build_subtitle_burn_command
from app.services.translate import TranslationEngine
from app.services.vendor_paths import kokoro_dir, translation_dir
from app.services.subtitle_mux import build_subtitle_mux_command
from app.services.subtitles import render_segments, segments_to_text
from app.models import TranscribeJob
from app.services.auth.identity import AuthenticatedUser
from app.services.job_manager_base import QueuedJobManager
from app.services.process_runner import run_guarded_process
from app.services.auth.quotas import QuotaService
from app.services.device_semaphores import DeviceSemaphores
from app.services.devices_service import AUTO_DEVICE_ID, DevicesService
from app.services.engines.transcribe_onnx import is_english_only, TranscribeRequest
from app.services.model_registry import ModelKind, ModelRegistry

# Los destinos que puede tener un trabajo: solo el texto, el video con la pista
# de subtitulos sumada, el video con el texto pintado en la imagen, el doblaje, y
# el karaoke (instrumental + letra sincronizada, todo en una pasada).
OUTPUT_MODES = ("text", "video", "video_burned", "dubbed_video", "karaoke")

logger = logging.getLogger(__name__)

# Codigos ISO 639-1 de dos letras es lo que espera whisper. Se valida la FORMA y no
# la lista completa: los modelos multilingues soportan casi cien idiomas y hardcodear
# la lista la dejaria desactualizada respecto de cualquier modelo nuevo.
_LANGUAGE_LENGTH = 2


def _discard_work_dir(work_dir: Path) -> None:
    """Borra la carpeta de trabajo con lo que tenga adentro, sin poder fallar.

    Sin poder fallar porque se llama desde rutas de limpieza: que la limpieza
    tape el error original seria cambiar un mensaje util por uno de permisos.
    """
    try:
        for archivo in work_dir.glob("*"):
            archivo.unlink(missing_ok=True)
        work_dir.rmdir()
    except OSError:
        logger.exception("no se pudo limpiar %s", work_dir)


class TranscribeJobManager(QueuedJobManager[TranscribeJob]):
    queue_full_message = "Transcribe job queue is full; try again later"
    worker_name_prefix = "transcribe-worker"

    def __init__(
        self,
        settings: Settings,
        engine: Any,
        device_semaphores: DeviceSemaphores,
        *,
        registry: ModelRegistry,
        devices: DevicesService | None = None,
        quota_service: QuotaService | None = None,
        separators: dict[str, Any] | None = None,
    ) -> None:
        # Un solo worker a propósito: la transcripción es CPU/GPU-bound y gatea
        # por device igual; N workers solo sumarían contención.
        super().__init__(settings, quota_service=quota_service, worker_count=1)
        self.engine = engine
        self.device_semaphores = device_semaphores
        self.registry = registry
        self.devices = devices
        # Los mismos motores que usa el modulo de audio. El karaoke necesita
        # separar, y montar un segundo juego de motores duplicaria la carga de
        # los .onnx en VRAM.
        self.separators = separators or {}

    async def create_job(
        self,
        *,
        source_path: Path,
        original_filename: str,
        model_id: str,
        language: str | None = None,
        device: str | None = None,
        job_id: str | None = None,
        owner: AuthenticatedUser | None = None,
        output_mode: str = "text",
        target_language: str | None = None,
    ) -> TranscribeJob:
        self._validate_model(model_id)
        self._validate_output_mode(output_mode)
        self._validate_dubbing(output_mode, target_language)
        self._validate_language(language)
        self._validate_language_fits_the_model(model_id, language)
        await self._validate_device(device)
        if owner is not None and self.quota_service is not None:
            self.quota_service.check_admission(owner)

        job = TranscribeJob(
            source_path=source_path,
            original_filename=original_filename,
            model_id=model_id,
            language=language,
            device=device,
            owner_id=owner.id if owner is not None else None,
            output_mode=output_mode,
            target_language=target_language,
        )
        if job_id is not None:
            job.id = job_id
        self.jobs[job.id] = job
        self._enqueue(job)
        return job

    @staticmethod
    def _validate_output_mode(output_mode: str) -> None:
        # Un modo desconocido se rechaza al crear el job y no al terminarlo:
        # descubrirlo despues de transcribir seria tirar todo el trabajo.
        if output_mode not in OUTPUT_MODES:
            raise ValueError(f"Modo de salida desconocido: {output_mode!r}")

    @staticmethod
    def _validate_dubbing(output_mode: str, target_language: str | None) -> None:
        # Se rechaza al CREAR el job: descubrirlo despues de transcribir seria
        # tirar todo ese trabajo.
        if output_mode == "dubbed_video" and not (target_language or "").strip():
            raise ValueError("Hace falta el idioma al que doblar.")

    def _validate_model(self, model_id: str) -> None:
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
        """Un modelo solo-ingles no puede transcribir otro idioma.

        Se rechaza ACA y no en el motor porque el usuario ya subio el archivo y
        eligio todo: enterarse a mitad de la transcripcion, con un mensaje en
        ingles de una libreria, no le dice que hacer.

        Tragarse el idioma tampoco serviria: el modelo devolveria el castellano
        escrito como si fuera ingles, y eso es peor que fallar — parece que
        funciono.
        """
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

    async def _validate_device(self, device: str | None) -> None:
        if device is None:
            return
        if device == AUTO_DEVICE_ID:
            raise ValueError(
                "device 'auto' is not supported for transcribe jobs; "
                "pin a concrete device (cpu|dml:N)"
            )
        if self.devices is not None:
            await asyncio.to_thread(self.devices.validate, device)

    def _resolve_model_dir(self, entry: Any) -> Path:
        models_root = self.settings.models_path.resolve()
        target = (self.settings.models_path / (entry.file_path or "")).resolve()
        if not target.is_relative_to(models_root):
            raise RuntimeError(f"Model path escapes models directory: {entry.file_path!r}")
        if not target.is_dir():
            raise RuntimeError(f"Model folder missing on disk: {entry.file_path!r}")
        return target

    def _admit(self, job: TranscribeJob):
        return self.device_semaphores.acquire(job.device)

    def _cleanup_source(self, job: TranscribeJob) -> None:
        self._unlink_source_safely(job.source_path)

    async def _run_engine(self, job: TranscribeJob) -> None:
        entry = self.registry.get(job.model_id)
        model_dir = self._resolve_model_dir(entry)
        device = job.device or self.settings.default_device

        def on_progress(done: int, total: int) -> None:
            job.progress_pct = round(done / max(total, 1) * 100, 1)

        segments = await self.engine.run(
            model_id=job.model_id,
            model_dir=model_dir,
            audio_path=job.source_path,
            request=TranscribeRequest(language=job.language),
            device=device,
            progress_cb=on_progress,
        )
        job.segments = segments
        job.text = segments_to_text(segments)
        job.output_path = self._write_transcript(job, job.text)
        if job.output_mode == "video":
            job.subtitled_video_path = await self._mux_subtitles_into_video(job)
        elif job.output_mode == "video_burned":
            job.subtitled_video_path = await self._burn_subtitles_into_video(job)
        elif job.output_mode == "dubbed_video":
            job.subtitled_video_path = await self._dub_video(job)
        elif job.output_mode == "karaoke":
            job.subtitled_video_path = await self._build_karaoke_video(job)

    async def _mux_subtitles_into_video(self, job: TranscribeJob) -> Path:
        """Suma la pista de subtitulos al contenedor SIN re-encodear el video.

        Se hace aca y no al descargar porque el fuente se borra al terminar el
        job: despues ya no habria video al que sumarle nada.
        """
        subtitle_path = self._write_subtitle_file(job)
        container = job.source_path.suffix.lstrip(".").lower() or "mkv"
        destination = self.settings.outputs_path / f"{job.id}.subtitled.{container}"
        command = build_subtitle_mux_command(
            ffmpeg=str(self.settings.ffmpeg_binary_path),
            video=job.source_path,
            subtitles=subtitle_path,
            destination=destination,
            container=container,
            language=job.language,
        )
        await self._run_ffmpeg(command, destination, "pegar los subtitulos al video")
        return destination

    async def _burn_subtitles_into_video(self, job: TranscribeJob) -> Path:
        """Re-encodea el video con los subtitulos pintados en la imagen.

        Se ve en cualquier reproductor, incluidas las redes que descartan las
        pistas de texto. Cuesta el re-encode entero, por eso no es el default.
        """
        subtitle_path = self._write_subtitle_file(job)
        container = job.source_path.suffix.lstrip(".").lower() or "mp4"
        destination = self.settings.outputs_path / f"{job.id}.subtitled.{container}"
        command = build_subtitle_burn_command(
            ffmpeg=str(self.settings.ffmpeg_binary_path),
            video=job.source_path,
            subtitles=subtitle_path,
            destination=destination,
        )
        await self._run_ffmpeg(command, destination, "quemar los subtitulos en el video")
        return destination

    async def _dub_video(self, job: TranscribeJob) -> Path:
        """Traduce, sintetiza y suma la voz doblada al video.

        La pista se arma en un hilo aparte: sintetizar es CPU pura y bloquearia
        el loop durante minutos en un video largo.
        """
        pipeline = self._dubbing_pipeline()
        total = max((float(s.end) for s in job.segments), default=0.0)

        def on_progress(done: int, total_lineas: int) -> None:
            job.progress_pct = round(done / max(total_lineas, 1) * 100, 1)

        try:
            resultado = await asyncio.to_thread(
                pipeline.build_track,
                list(job.segments),
                source_language=job.language,
                target_language=job.target_language or "",
                total_seconds=total,
                progress_cb=on_progress,
            )
        except DubbingUnavailable as exc:
            raise RuntimeError(str(exc)) from exc

        job.dub_overflow_segments = resultado.overflowing
        voz = self.settings.outputs_path / f"{job.id}.dub.wav"
        voz.parent.mkdir(parents=True, exist_ok=True)
        import soundfile

        soundfile.write(voz, resultado.track, TTS_SAMPLE_RATE, format="WAV", subtype="PCM_16")

        container = job.source_path.suffix.lstrip(".").lower() or "mkv"
        destination = self.settings.outputs_path / f"{job.id}.dubbed.{container}"
        command = build_dub_mux_command(
            ffmpeg=str(self.settings.ffmpeg_binary_path),
            video=job.source_path,
            dubbed_audio=voz,
            destination=destination,
            language=job.target_language,
        )
        await self._run_ffmpeg(command, destination, "pegar la voz doblada al video")
        return destination

    def _dubbing_pipeline(self) -> DubbingPipeline:
        from app.services.phonemize import text_to_phonemes

        return DubbingPipeline(
            translation=TranslationEngine(translation_dir(self.settings)),
            tts=KokoroTtsEngine(self.settings),
            tts_model_dir=kokoro_dir(self.settings),
            phonemize=text_to_phonemes,
            sample_rate=TTS_SAMPLE_RATE,
            available_voices=available_voices(kokoro_dir(self.settings)),
        )

    async def _build_karaoke_video(self, job: TranscribeJob) -> Path:
        """Instrumental + letra sincronizada, en un solo trabajo.

        Las tres piezas ya existian sueltas; lo que faltaba era el empalme, que
        es pegarle a la imagen la pista INSTRUMENTAL en vez de la original.

        La letra sale de la transcripcion del audio COMPLETO, no del instrumental
        —ahi justamente no hay voz que transcribir— y eso ya paso antes de llegar
        aca, que es por que este paso corre al final y no al principio.
        """
        instrumental = await self._separate_instrumental(job)
        try:
            subtitles = self._write_karaoke_subtitle_file(job)
            picture = job.source_path if await self._has_picture(job.source_path) else None
            duration = await probe_duration_seconds(instrumental, self.settings) or 0.0
            destination = self.settings.outputs_path / f"{job.id}.karaoke.mp4"
            command = build_karaoke_command(
                ffmpeg=str(self.settings.ffmpeg_binary_path),
                picture=picture,
                duration_seconds=duration,
                instrumental=instrumental,
                subtitles=subtitles,
                destination=destination,
            )
            await self._run_ffmpeg(command, destination, "armar el video de karaoke")
        finally:
            # En `finally` y no despues del ffmpeg: si el armado falla, el wav sin
            # comprimir y su carpeta se quedan para siempre, y son los archivos
            # mas pesados que produce el trabajo. Un fallo no puede costar disco.
            _discard_work_dir(instrumental.parent)
        return destination

    async def _separate_instrumental(self, job: TranscribeJob) -> Path:
        """El stem principal del modelo de karaoke, decodificado y separado."""
        from app.services.engines.separation_models import (
            DEFAULT_SEPARATION_MODEL,
            SEPARATION_MODELS,
        )

        spec = SEPARATION_MODELS[DEFAULT_SEPARATION_MODEL]
        separator = self.separators.get(spec.architecture)
        if separator is None:
            # Pedir karaoke y recibir un video con la voz intacta seria peor que
            # fallar diciendo que falta el motor.
            raise RuntimeError(
                "El karaoke necesita el motor de separacion "
                f"{spec.architecture!r} y el servidor se construyo sin el."
            )
        work_dir = self.settings.outputs_path / f"{job.id}.karaoke"
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            decoded = work_dir / "mezcla.wav"
            await self._decode_for_separation(job.source_path, decoded)
            stem_wavs = [work_dir / f"{stem.id}.wav" for stem in spec.stems]
            await separator.run(
                decoded,
                stem_wavs,
                job.device or self.settings.default_device,
                model_id=spec.id,
            )
        except Exception:
            # Un fallo de decodificado o de separacion deja la carpeta con lo que
            # alcanzo a escribir, y nadie mas conoce ese nombre.
            _discard_work_dir(work_dir)
            raise
        decoded.unlink(missing_ok=True)
        # `main_stem` y no `[0]`: la posicion es hoy la misma, pero cual es el
        # principal lo decide el catalogo.
        principal = work_dir / f"{spec.main_stem.id}.wav"
        for sobrante in stem_wavs:
            if sobrante != principal:
                sobrante.unlink(missing_ok=True)
        return principal

    async def _decode_for_separation(self, source: Path, destination: Path) -> None:
        command = build_decode_to_wav_command(
            ffmpeg=str(self.settings.ffmpeg_binary_path),
            source=source,
            destination=destination,
            sample_rate=SEPARATION_SAMPLE_RATE,
            channels=SEPARATION_CHANNELS,
        )
        await self._run_ffmpeg(command, destination, "decodificar el audio a separar")

    async def _has_picture(self, source: Path) -> bool:
        command = build_picture_probe_command(self.settings.ffprobe_binary_path, source)
        stdout, _stderr, returncode = await run_guarded_process(
            command, self.settings.subprocess_timeout
        )
        if returncode != 0:
            return False
        try:
            return has_real_picture(json.loads(stdout.decode("utf-8", errors="ignore")))
        except json.JSONDecodeError:
            return False

    def _write_karaoke_subtitle_file(self, job: TranscribeJob) -> Path:
        """La letra en ASS, con cada palabra encendiendose a su tiempo.

        ASS y no SRT porque SRT no tiene forma de expresar esto: habria que
        emitir una linea por palabra, parpadeando, que es peor que no resaltar.

        Mientras el motor entregue tiempos por LINEA, el reparto entre palabras
        es proporcional a la cantidad de letras. Es una aproximacion —una nota
        sostenida sobre una palabra corta se corre— y se reemplaza sola cuando
        los segmentos traigan tiempos por palabra.
        """
        lineas = [
            split_line_proportionally(segmento.text, segmento.start, segmento.end)
            for segmento in job.segments
            if getattr(segmento, "text", "").strip()
        ]
        destino = self.settings.outputs_path / f"{job.id}.ass"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(render_karaoke_ass(lineas), encoding="utf-8")
        return destino

    def _write_subtitle_file(self, job: TranscribeJob) -> Path:
        subtitle_path = self.settings.outputs_path / f"{job.id}.srt"
        subtitle_path.parent.mkdir(parents=True, exist_ok=True)
        subtitle_path.write_text(render_segments(list(job.segments), "srt"), encoding="utf-8")
        return subtitle_path

    async def _run_ffmpeg(self, command: list[str], destination: Path, que_hacia: str) -> None:
        # run_guarded_process y no create_subprocess_exec pelado: aplica el
        # timeout global y MATA el ffmpeg si el job se cancela — antes un cancel
        # dejaba el mux/burn corriendo hasta terminar.
        _stdout, stderr, returncode = await run_guarded_process(
            command, self.settings.subprocess_timeout
        )
        if returncode != 0 or not destination.exists():
            raise RuntimeError(
                f"No se pudo {que_hacia}. "
                f"ffmpeg dijo: {(stderr or b'').decode('utf-8', 'replace').strip()[-300:]}"
            )

    def _write_transcript(self, job: TranscribeJob, text: str) -> Path:
        # El texto ya vive en el job; el .txt existe para que la descarga use el
        # mismo camino que el resto de los jobs.
        destination = self.settings.outputs_path / f"{job.id}.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        return destination

    @staticmethod
    def _unlink_source_safely(source_path: Path) -> None:
        try:
            source_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to delete source upload %s", source_path)
