from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from app.config import Settings
from app.services.process_runner import is_non_empty_file, run_guarded_process
from app.services.voice_chain import (
    ChainStep,
    FfmpegStage,
    ModelStage,
    plan_stages,
)

# Resuelve un paso de modelo a algo ejecutable. Se inyecta para que el motor no
# conozca ni DeepFilterNet ni Demucs ni lo que venga: solo sabe que hay etapas.
ModelStageRunner = Callable[[str, Path, Path], Awaitable[None]]


class VoiceEnhancer:
    """Ejecuta una cadena de mejora de voz mezclando DSP y modelos.

    Cada FfmpegStage es UNA pasada de ffmpeg con todos sus filtros fusionados;
    cada ModelStage es su propia invocacion. La fusion importa: cada pasada es
    un ciclo completo de decode y encode sobre todo el audio.
    """

    def __init__(
        self,
        settings: Settings,
        model_stage_runner: ModelStageRunner | None = None,
    ) -> None:
        self.settings = settings
        self._run_model_stage = model_stage_runner

    async def run(
        self,
        steps: list[ChainStep],
        input_wav: Path,
        output_wav: Path,
        work_dir: Path,
    ) -> None:
        stages = plan_stages(steps)
        if not stages:
            # Cadena vacia: no hay nada que aplicar, pero el caller espera el
            # archivo de salida, asi que se copia tal cual en vez de fallar.
            output_wav.write_bytes(input_wav.read_bytes())
            return

        current = input_wav
        for index, stage in enumerate(stages):
            is_last = index == len(stages) - 1
            target = output_wav if is_last else work_dir / f"voice-{index}.wav"
            await self._run_stage(stage, current, target)
            current = target

    async def _run_stage(self, stage: FfmpegStage | ModelStage, source: Path, target: Path) -> None:
        if isinstance(stage, FfmpegStage):
            await self._run_ffmpeg_stage(stage, source, target)
            return
        if self._run_model_stage is None:
            raise RuntimeError(
                f"La cadena pide el modelo {stage.model_capability!r} pero este motor "
                "se construyo sin un resolutor de pasos por modelo."
            )
        await self._run_model_stage(stage.model_capability, source, target)
        self._require_output(target, f"el paso por modelo {stage.model_capability!r}")

    async def _run_ffmpeg_stage(self, stage: FfmpegStage, source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(self.settings.ffmpeg_binary_path),
            "-y",
            "-i",
            str(source),
            "-af",
            stage.filter_argument,
            str(target),
        ]
        _, stderr, returncode = await run_guarded_process(
            command, self.settings.subprocess_timeout
        )
        if returncode != 0:
            detail = stderr.decode("utf-8", errors="ignore").strip()
            steps = ", ".join(stage.step_ids)
            raise RuntimeError(
                f"La cadena de voz fallo en los pasos [{steps}]: {detail or 'ffmpeg fallo'}"
            )
        self._require_output(target, f"los pasos [{', '.join(stage.step_ids)}]")

    @staticmethod
    def _require_output(path: Path, what: str) -> None:
        if not is_non_empty_file(path):
            raise RuntimeError(f"{what} no produjo ningun archivo de salida.")
