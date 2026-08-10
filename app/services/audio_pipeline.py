from __future__ import annotations

import logging
import shutil
from pathlib import Path

from typing import TYPE_CHECKING

from app.config import Settings
from app.models import AudioJob
from app.services.engines.audio_enhance import AudioEnhancer
from app.services.engines.voice_enhance import VoiceEnhancer
from app.services.voice_chain import effective_voice_steps, steps_from_selection
from app.services.restorer_registry import AudioRestorer
from app.services.process_runner import run_guarded_process
from app.services.progress import (
    advance_audio_stage,
    apply_audio_chunk_progress,
    complete_audio_stages,
)

if TYPE_CHECKING:
    from app.services.engines.separator_base import OnnxStemSeparator

logger = logging.getLogger(__name__)

# DeepFilterNet/RNNoise expect 48kHz PCM; decode every accepted container
# (wav/mp3/flac/m4a/ogg/opus) to that so the denoise step is format-agnostic.
DECODE_SAMPLE_RATE = 48000

# Codec args for the final "finalizing" re-encode, keyed by output_format
# (Fase C Task 9). "wav" is deliberately absent: current is already PCM WAV
# from decode/denoise/restore, so it is moved into place with no re-encode
# (see AudioPipeline._write_output).
_OUTPUT_FORMAT_CODEC_ARGS: dict[str, list[str]] = {
    "flac": ["-c:a", "flac"],
    "mp3": ["-c:a", "libmp3lame", "-b:a", "192k"],
}


class AudioPipeline:
    """Orquesta la cadena de audio standalone:

        decode -> [limpieza] -> [denoise] -> [restore] -> [voz] -> [mastering]

    El orden es causal, no una lista de opciones:

    * La LIMPIEZA va primero (apenas decodificado) porque sus modelos son
      mascaras entrenadas sobre material de banda completa: cualquier paso
      previo que ya haya recortado el espectro les cambia justamente la entrada
      que aprendieron a leer. Ademas es el paso que QUITA defectos, y quitar
      antes de reconstruir evita reconstruir un defecto.
    * denoise/restore reconstruyen la senal; voz y mastering la nivelan. Nivelar
      va ultimo siempre: mide sobre la senal terminada.

    Cada paso es opcional; el manager garantiza que se pidio al menos uno. Los
    archivos intermedios viven en un temp dir por job que se borra en finally,
    asi un fallo nunca deja trabajo colgado.
    """

    def __init__(
        self,
        settings: Settings,
        audio_enhancers: dict[str, AudioEnhancer],
        restorers: dict[str, AudioRestorer],
        voice_enhancer: VoiceEnhancer | None = None,
        separators: "dict[str, OnnxStemSeparator] | None" = None,
    ) -> None:
        self.settings = settings
        self.audio_enhancers = audio_enhancers
        self.restorers = restorers
        self.voice_enhancer = voice_enhancer
        # Motor de separacion por ARQUITECTURA del modelo ("mdx" | "vr"). El
        # usuario elige un id de modelo, nunca una arquitectura: el catalogo
        # dice cual es y aca se resuelve el motor.
        self.separators = separators or {}

    async def run(self, job: AudioJob) -> Path:
        work_dir = self.settings.temp_path / f"audio-{job.id}"
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            return await self._run_chain(job, work_dir)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    async def _run_chain(self, job: AudioJob, work_dir: Path) -> Path:
        advance_audio_stage(job, "decoding")
        current = work_dir / "decoded.wav"
        # La cadena de limpieza corre los mismos motores que la separacion, asi
        # que necesita el mismo decode: 44100 estereo.
        await self._decode_to_wav(
            job.source_path,
            current,
            force_stereo=job.separate or bool(job.cleanup_steps),
        )

        if job.separate:
            return await self._run_separation(job, work_dir, current)

        if job.cleanup_steps:
            current = await self._run_cleanup_chain(job, work_dir, current)

        if job.denoise:
            advance_audio_stage(job, "denoising")
            denoised = work_dir / "denoised.wav"
            await self._denoise(job.denoise, current, denoised)
            current = denoised

        if job.restore:
            advance_audio_stage(job, "restoring")
            restored = work_dir / "restored.wav"
            await self._restore(job.restore, current, restored, job.device)
            current = restored

        voice_steps = self._voice_steps_without_double_loudnorm(job)
        if voice_steps:
            if self.voice_enhancer is None:
                # Pedir la cadena y que se ignore en silencio seria peor que
                # fallar: el usuario creeria que se aplico.
                raise RuntimeError(
                    "El job pide mejora de voz pero el pipeline se construyo sin "
                    "motor de voz."
                )
            advance_audio_stage(job, "voicing")
            voiced = work_dir / "voiced.wav"
            await self.voice_enhancer.run(
                steps_from_selection(
                    voice_steps,
                    delivery=job.voice_delivery,
                    presence_db=job.voice_presence_db,
                    rnnoise_model=str(self.settings.rnnoise_model_path),
                ),
                current,
                voiced,
                work_dir,
            )
            current = voiced

        if job.master:
            advance_audio_stage(job, "mastering")
            mastered = work_dir / "mastered.wav"
            if await self._master(job, current, mastered):
                current = mastered

        advance_audio_stage(job, "finalizing")
        output_path = self.settings.outputs_path / f"{job.id}.{job.output_format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await self._write_output(current, output_path, job.output_format)
        self._validate_output(output_path)
        complete_audio_stages(job)
        return output_path

    async def _run_cleanup_chain(self, job: AudioJob, work_dir: Path, decoded: Path) -> Path:
        """Una pasada por paso, encadenadas: la salida limpia alimenta la siguiente.

        Devuelve UN archivo. El stem removido de cada pasada (el eco, la reverb,
        el ruido) se escribe en el work dir y muere con el: en modo cadena no hay
        `stems[]` ni `?stem=`. Quien quiera escuchar lo que una pasada saco corre
        ESE modelo solo en modo separacion, que sigue existiendo y para eso esta.

        La cancelacion no necesita nada especial: cada pasada es un await sobre
        el separador, que ya la maneja adentro (cancel_event + espera del hilo),
        y entre pasadas cae en el await siguiente.
        """
        from app.services.cleanup_chain import cleanup_steps_from_selection, is_overprocessing
        from app.services.engines.separation_models import SEPARATION_MODELS
        from app.services.progress import cleanup_stage_key

        steps = cleanup_steps_from_selection(job.cleanup_steps)
        job.metadata["cleanupPasses"] = len(steps)
        if is_overprocessing(len(steps)):
            # El aviso viaja en el job, no solo en la UI: un agente por MCP toma
            # la misma decision con la misma informacion que alguien mirando la
            # pantalla.
            job.metadata["cleanupOverprocessed"] = (
                f"{len(steps)} pasadas con perdida encadenadas: el resultado "
                "puede sonar sobreprocesado."
            )

        current = decoded
        device = job.device or self.settings.default_device
        for index, step in enumerate(steps):
            spec = SEPARATION_MODELS[step.model_id]
            separator = self._require_separator(spec.architecture, step.model_id)
            stage_key = cleanup_stage_key(step.model_id)
            advance_audio_stage(job, stage_key)
            # main_stem es, por contrato del catalogo, el stem que el usuario
            # quiere — en los modelos de limpieza, el audio SIN el defecto.
            clean = work_dir / f"cleanup-{index}-{step.model_id}.wav"
            removed = work_dir / f"cleanup-{index}-{step.model_id}-removed.wav"
            await separator.run(
                current,
                clean,
                removed,
                device,
                model_id=step.model_id,
                on_chunk=lambda done, total, key=stage_key: apply_audio_chunk_progress(
                    job, done, total, key
                ),
            )
            current = clean

        return await self._match_denoiser_sample_rate(job, work_dir, current)

    async def _match_denoiser_sample_rate(
        self, job: AudioJob, work_dir: Path, current: Path
    ) -> Path:
        """Vuelve a 48 kHz cuando despues de la limpieza corre el denoise clasico.

        Los separadores emiten SIEMPRE 44100 (es su sample rate de
        entrenamiento), y DeepFilterNet/RNNoise estan especificados a 48000, que
        es a lo que decodifica el resto del pipeline. Sin esto la combinacion
        limpieza + reduccion de ruido le entregaria al denoiser una tasa que no
        es la suya. Solo se paga cuando ambas cosas se piden juntas.
        """
        if not job.denoise:
            return current
        resampled = work_dir / "cleaned-48k.wav"
        await self._decode_to_wav(current, resampled)
        return resampled

    async def _run_separation(self, job: AudioJob, work_dir: Path, decoded: Path) -> Path:
        """Modo separacion: DOS salidas nombradas por stem, mismo formato ambas.

        job.output_path (lo devuelto) es el stem principal del modelo (el que
        el usuario quiere: instrumental en karaoke, dry/no_echo en limpieza); el
        restante queda en job.secondary_output_path. El manager ya garantizo
        que el modo corre solo.
        """
        from app.services.engines.separation_models import (
            DEFAULT_SEPARATION_MODEL,
            SEPARATION_MODELS,
        )

        advance_audio_stage(job, "separating")
        model_id = job.separation_model or DEFAULT_SEPARATION_MODEL
        spec = SEPARATION_MODELS.get(model_id)
        if spec is None:
            # El manager valida al crear; esto cubre llamadas directas.
            known = ", ".join(sorted(SEPARATION_MODELS))
            raise RuntimeError(
                f"Modelo de separacion desconocido: {model_id!r}. Validos: {known}."
            )
        separator = self._require_separator(spec.architecture, model_id)
        main_wav = work_dir / f"{spec.main_stem.id}.wav"
        other_wav = work_dir / f"{spec.other_stem.id}.wav"
        await separator.run(
            decoded,
            main_wav,
            other_wav,
            job.device or self.settings.default_device,
            model_id=spec.id,
            on_chunk=lambda done, total: apply_audio_chunk_progress(job, done, total),
        )

        advance_audio_stage(job, "finalizing")
        output_format = job.output_format
        outputs_dir = self.settings.outputs_path
        outputs_dir.mkdir(parents=True, exist_ok=True)
        main_out = outputs_dir / f"{job.id}.{spec.main_stem.id}.{output_format}"
        other_out = outputs_dir / f"{job.id}.{spec.other_stem.id}.{output_format}"
        await self._write_output(main_wav, main_out, output_format)
        await self._write_output(other_wav, other_out, output_format)
        self._validate_output(main_out)
        self._validate_output(other_out)
        job.secondary_output_path = other_out
        complete_audio_stages(job)
        return main_out

    def _require_separator(self, architecture: str, model_id: str):
        # Mismo criterio que la cadena de voz: pedir la separacion y que se
        # ignore en silencio seria peor que fallar.
        separator = self.separators.get(architecture)
        if separator is None:
            raise RuntimeError(
                f"El job pide el modelo {model_id!r} (arquitectura {architecture}) "
                "pero el pipeline se construyo sin ese motor de separacion."
            )
        return separator

    def _voice_steps_without_double_loudnorm(self, job: AudioJob) -> list[str]:
        """Con mastering activo, el paso `loudness` de la cadena de voz sobra.

        Aplicar ambos normalizaba DOS veces: primero el loudnorm single-pass de
        la cadena de voz (la forma que la propia doc de mastering explica que
        "bombea") y despues el de dos pasadas midiendo audio ya normalizado.
        Gana el mastering, y el motivo queda en metadata — nunca en silencio.
        """
        steps = effective_voice_steps(job.voice_steps, job.master)
        if steps != job.voice_steps:
            job.metadata["voiceLoudnessSkipped"] = (
                "mastering activo aplica la normalización final (dos pasadas)"
            )
        return steps

    async def _master(self, job: AudioJob, source: Path, destination: Path) -> bool:
        """Normaliza la sonoridad en dos pasadas y deja lo medido en el job.

        Devuelve False si no se pudo medir: sin medicion se entrega el audio sin
        masterizar en vez de fallar el trabajo entero por el paso de acabado, y
        el motivo queda registrado para que no sea un silencio.
        """
        from app.services.audio_mastering import (
            build_master_command,
            build_measure_command,
            mastering_preset,
            parse_loudness_measurement,
        )

        ffmpeg = self.settings.ffmpeg_binary_path
        measure_output = await self._run_process_capturing(
            build_measure_command(ffmpeg, source, job.master)
        )
        measurement = parse_loudness_measurement(measure_output)
        if measurement is None:
            job.metadata["masteringSkipped"] = "no se pudo medir la sonoridad"
            return False

        await self._run_process(
            build_master_command(ffmpeg, source, destination, job.master, measurement),
            "Audio mastering failed",
        )
        # Lo medido viaja al job: el usuario ve de cuanto a cuanto se movio.
        job.metadata["loudnessBefore"] = measurement.input_i
        job.metadata["loudnessTarget"] = mastering_preset(job.master).target_lufs
        return True

    async def _write_output(self, current: Path, output_path: Path, output_format: str) -> None:
        if output_format == "wav":
            shutil.move(str(current), str(output_path))
            return
        codec_args = _OUTPUT_FORMAT_CODEC_ARGS.get(output_format, _OUTPUT_FORMAT_CODEC_ARGS["flac"])
        command = [
            str(self.settings.ffmpeg_binary_path),
            "-y",
            "-i",
            str(current),
            *codec_args,
            str(output_path),
        ]
        await self._run_process(command, "Audio encode failed while writing the final output file")

    async def _decode_to_wav(
        self, source_path: Path, output_wav: Path, force_stereo: bool = False
    ) -> None:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        # Los separadores trabajan a 44100 estereo: decodificar directo a eso
        # evita un resample extra y le baja el surround a 2 canales a ffmpeg,
        # que tiene el downmix canonico.
        sample_rate = 44100 if force_stereo else DECODE_SAMPLE_RATE
        channel_args = ["-ac", "2"] if force_stereo else []
        command = [
            str(self.settings.ffmpeg_binary_path),
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            *channel_args,
            str(output_wav),
        ]
        await self._run_process(command, "Audio decode failed; the uploaded file is not a supported audio format")

    async def _denoise(self, mode: str, input_wav: Path, output_wav: Path) -> None:
        enhancer = self.audio_enhancers.get(mode)
        if enhancer is None:
            raise RuntimeError(f"Denoise mode {mode!r} requested but no engine is configured")
        await enhancer.run(input_wav, output_wav)

    async def _restore(self, mode: str, input_wav: Path, output_wav: Path, device: str | None) -> None:
        restorer = self.restorers.get(mode)
        if restorer is None:
            raise RuntimeError(f"Restore mode {mode!r} requested but no engine is configured")
        resolved_device = device or self.settings.default_device
        await restorer.run(input_wav, output_wav, resolved_device)

    async def _run_process(self, command: list[str], failure_message: str) -> None:
        _, stderr, returncode = await run_guarded_process(command, self.settings.subprocess_timeout)
        if returncode != 0:
            detail = stderr.decode("utf-8", errors="ignore").strip()
            raise RuntimeError(detail.splitlines()[-1] if detail else failure_message)

    async def _run_process_capturing(self, command: list[str]) -> str:
        """Corre y devuelve stderr. loudnorm imprime su medicion AHI, no en stdout,
        y un exit distinto de cero se trata como "no se pudo medir": el paso de
        acabado nunca puede tumbar un trabajo que ya proceso el audio."""
        _, stderr, _returncode = await run_guarded_process(
            command, self.settings.subprocess_timeout
        )
        return stderr.decode("utf-8", errors="ignore")

    def _validate_output(self, output_path: Path) -> None:
        if not (output_path.exists() and output_path.stat().st_size > 0):
            raise RuntimeError("Audio processing finished but no output file was produced")
