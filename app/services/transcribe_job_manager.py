from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from app.config import Settings
from app.exceptions import QueueFullError
from app.services.dub_mux import build_dub_mux_command
from app.services.dubbing_pipeline import DubbingPipeline, DubbingUnavailable
from app.services.engines.tts_kokoro import SAMPLE_RATE as TTS_SAMPLE_RATE
from app.services.engines.tts_kokoro import KokoroTtsEngine, available_voices
from app.services.subtitle_burn import build_subtitle_burn_command
from app.services.translate import TranslationEngine
from app.services.vendor_paths import kokoro_dir, translation_dir
from app.services.subtitle_mux import build_subtitle_mux_command
from app.services.subtitles import render_segments, segments_to_text
from app.models import JobStatus, TranscribeJob, utc_now
from app.services.auth.identity import AuthenticatedUser
from app.services.auth.quotas import QuotaService
from app.services.device_semaphores import DeviceSemaphores
from app.services.devices_service import AUTO_DEVICE_ID, DevicesService
from app.services.engines.transcribe_onnx import TranscribeRequest
from app.services.model_registry import ModelKind, ModelRegistry

# Los tres destinos que puede tener un trabajo: solo el texto, el video con la
# pista de subtitulos sumada, o el video con el texto pintado en la imagen.
OUTPUT_MODES = ("text", "video", "video_burned", "dubbed_video")

logger = logging.getLogger(__name__)

# Codigos ISO 639-1 de dos letras es lo que espera whisper. Se valida la FORMA y no
# la lista completa: los modelos multilingues soportan casi cien idiomas y hardcodear
# la lista la dejaria desactualizada respecto de cualquier modelo nuevo.
_LANGUAGE_LENGTH = 2


class TranscribeJobManager:
    def __init__(
        self,
        settings: Settings,
        engine: Any,
        device_semaphores: DeviceSemaphores,
        *,
        registry: ModelRegistry,
        devices: DevicesService | None = None,
        quota_service: QuotaService | None = None,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.device_semaphores = device_semaphores
        self.registry = registry
        self.devices = devices
        self.quota_service = quota_service
        self.jobs: dict[str, TranscribeJob] = {}
        self.queue: asyncio.Queue[TranscribeJob] = asyncio.Queue(
            maxsize=settings.max_queue_size
        )
        self._worker_task: asyncio.Task[None] | None = None
        self._active: dict[str, asyncio.Task[None]] = {}

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None

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

    def get_job(self, job_id: str) -> TranscribeJob | None:
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job is None or job.status in (
            JobStatus.completed,
            JobStatus.failed,
            JobStatus.cancelled,
        ):
            return False
        task = self._active.get(job_id)
        if task is not None:
            task.cancel()
        else:
            # Todavia en la cola: se marca y el worker lo saltea sin procesarlo.
            job.status = JobStatus.cancelled
            job.finished_at = utc_now()
        return True

    def _enqueue(self, job: TranscribeJob) -> None:
        try:
            self.queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            raise QueueFullError(
                "Transcribe job queue is full; try again later"
            ) from exc

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

    async def _worker(self) -> None:
        while True:
            job = await self.queue.get()
            if job.status == JobStatus.cancelled:
                self._unlink_source_safely(job.source_path)
                self.queue.task_done()
                continue
            await self._run_job(job)

    async def _process_next(self) -> bool:
        if self.queue.empty():
            return False
        job = await self.queue.get()
        if job.status == JobStatus.cancelled:
            self._unlink_source_safely(job.source_path)
            self.queue.task_done()
            return True
        await self._run_job(job)
        return True

    async def _run_job(self, job: TranscribeJob) -> None:
        async with self.device_semaphores.acquire(job.device):
            await self._execute_job(job)

    async def _execute_job(self, job: TranscribeJob) -> None:
        job.status = JobStatus.running
        job.started_at = utc_now()
        run_task = asyncio.ensure_future(self._run_engine(job))
        self._active[job.id] = run_task
        try:
            await run_task
            job.status = JobStatus.completed
        except asyncio.CancelledError:
            run_task.cancel()
            current = asyncio.current_task()
            if current is not None and current.cancelling() > 0:
                # El WORKER se cancelo (stop()): el job falla y se re-lanza para que
                # el worker muera de verdad.
                job.status = JobStatus.failed
                job.error = "Job cancelled"
                raise
            job.status = JobStatus.cancelled
            job.error = None
        except Exception as exc:  # noqa: BLE001
            job.status = JobStatus.failed
            job.error = str(exc)
        finally:
            self._active.pop(job.id, None)
            job.finished_at = utc_now()
            self._unlink_source_safely(job.source_path)
            self.queue.task_done()
            self._record_quota_usage(job)

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

    def _write_subtitle_file(self, job: TranscribeJob) -> Path:
        subtitle_path = self.settings.outputs_path / f"{job.id}.srt"
        subtitle_path.parent.mkdir(parents=True, exist_ok=True)
        subtitle_path.write_text(render_segments(list(job.segments), "srt"), encoding="utf-8")
        return subtitle_path

    async def _run_ffmpeg(self, command: list[str], destination: Path, que_hacia: str) -> None:
        process = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
        )
        _stdout, stderr = await process.communicate()
        if process.returncode != 0 or not destination.exists():
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

    def _record_quota_usage(self, job: TranscribeJob) -> None:
        if self.quota_service is None or job.started_at is None or job.finished_at is None:
            return
        duration = (job.finished_at - job.started_at).total_seconds()
        self.quota_service.record_usage(job.owner_id, duration)

    @staticmethod
    def _unlink_source_safely(source_path: Path) -> None:
        try:
            source_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to delete source upload %s", source_path)
