from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from app.config import AUDIO_ENHANCE_MODES, DEEPFILTER_MODE, Settings
from app.services.process_runner import run_guarded_process


class AudioEnhancer:
    def __init__(self, settings: Settings, mode: str) -> None:
        self._validate_mode(mode)
        self.settings = settings
        self.mode = mode

    @staticmethod
    def _validate_mode(mode: str) -> None:
        if mode not in AUDIO_ENHANCE_MODES:
            raise ValueError(f"Unknown audio enhance mode: {mode!r}")

    def available(self) -> bool:
        return self.settings.audio_enhance_available(self.mode)

    async def run(self, input_wav: Path, output_wav: Path) -> None:
        if not self.available():
            raise RuntimeError(
                f"Audio enhance mode {self.mode!r} is not available. "
                "Run scripts/download-deepfilternet.ps1 first."
            )

        output_wav.parent.mkdir(parents=True, exist_ok=True)

        if self.mode == DEEPFILTER_MODE:
            await self._run_deepfilter(input_wav, output_wav)
        else:
            await self._run_rnnoise(input_wav, output_wav)

    async def _run_deepfilter(self, input_wav: Path, output_wav: Path) -> None:
        # deep-filter only accepts an output *directory* (-o) and writes the
        # enhanced file under the input's own basename inside it. Pointing -o
        # at output_wav.parent would overwrite the SOURCE wav in place
        # whenever input and output share a directory, so the run is isolated
        # in a unique temp dir and the result promoted to the exact
        # output_wav afterwards -- one safe code path for same-dir,
        # different-dir and same-name layouts.
        temp_dir = output_wav.parent / f".dfn-{uuid4().hex}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            command = self._build_deepfilter_command(input_wav, temp_dir)
            await self._execute(command, "DeepFilterNet enhancement process failed")
            self._promote_deepfilter_output(temp_dir / input_wav.name, output_wav)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
        self._validate_output(output_wav)

    def _build_deepfilter_command(self, input_wav: Path, out_dir: Path) -> list[str]:
        return [
            str(self.settings.deepfilter_binary_path),
            "-o",
            str(out_dir),
            str(input_wav),
        ]

    @staticmethod
    def _promote_deepfilter_output(produced_path: Path, output_wav: Path) -> None:
        if not produced_path.exists():
            raise RuntimeError(
                "DeepFilterNet process completed but no output file was produced "
                f"(expected {produced_path.name} in the temp output dir)"
            )
        produced_path.replace(output_wav)

    async def _run_rnnoise(self, input_wav: Path, output_wav: Path) -> None:
        command = self._build_rnnoise_command(input_wav, output_wav)
        await self._execute(command, "RNNoise ffmpeg filter process failed")
        self._validate_output(output_wav)

    def _build_rnnoise_command(self, input_wav: Path, output_wav: Path) -> list[str]:
        model_arg = self._escape_filter_path(self.settings.rnnoise_model_path)
        return [
            str(self.settings.ffmpeg_binary_path),
            "-y",
            "-i",
            str(input_wav),
            "-af",
            f"arnndn=m={model_arg}",
            str(output_wav),
        ]

    @staticmethod
    def _escape_filter_path(path: Path) -> str:
        # ffmpeg filter option values use ':' as the key=value separator, so
        # a Windows drive-letter colon (and any literal backslash, itself an
        # escape character in filter syntax) breaks parsing if passed as-is.
        # Forward slashes are accepted as path separators on Windows, so
        # converting to them first sidesteps backslash-escaping entirely;
        # only the colon then needs a leading backslash.
        posix_style = str(path).replace("\\", "/")
        return posix_style.replace(":", r"\:")

    async def _execute(self, command: list[str], failure_message: str) -> None:
        _, stderr, returncode = await run_guarded_process(command, self.settings.subprocess_timeout)
        if returncode != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="ignore") or failure_message)

    def _validate_output(self, output_wav: Path) -> None:
        if not self._is_non_empty_file(output_wav):
            raise RuntimeError(
                f"Audio enhance mode {self.mode!r} completed but no output file was produced"
            )

    @staticmethod
    def _is_non_empty_file(path: Path) -> bool:
        return path.exists() and path.stat().st_size > 0
