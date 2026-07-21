from __future__ import annotations

import logging
import shutil
from pathlib import Path

from app.config import Settings
from app.models import AudioJob
from app.services.engines.audio_enhance import AudioEnhancer
from app.services.restorer_registry import AudioRestorer
from app.services.process_runner import run_guarded_process
from app.services.progress import advance_audio_stage, complete_audio_stages

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
    """Orchestrates the standalone audio chain: decode -> [denoise] -> [restore].

    Denoise runs first (clean the noise), restore second (rebuild the band the
    codec dropped). Each step is optional; the manager guarantees at least one
    is requested. Intermediate files live in a per-job temp dir removed in
    finally, so a failure never leaks work files.
    """

    def __init__(
        self,
        settings: Settings,
        audio_enhancers: dict[str, AudioEnhancer],
        restorers: dict[str, AudioRestorer],
    ) -> None:
        self.settings = settings
        self.audio_enhancers = audio_enhancers
        self.restorers = restorers

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
        await self._decode_to_wav(job.source_path, current)

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

        advance_audio_stage(job, "finalizing")
        output_path = self.settings.outputs_path / f"{job.id}.{job.output_format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        await self._write_output(current, output_path, job.output_format)
        self._validate_output(output_path)
        complete_audio_stages(job)
        return output_path

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

    async def _decode_to_wav(self, source_path: Path, output_wav: Path) -> None:
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.settings.ffmpeg_binary_path),
            "-y",
            "-i",
            str(source_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(DECODE_SAMPLE_RATE),
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

    def _validate_output(self, output_path: Path) -> None:
        if not (output_path.exists() and output_path.stat().st_size > 0):
            raise RuntimeError("Audio processing finished but no output file was produced")
