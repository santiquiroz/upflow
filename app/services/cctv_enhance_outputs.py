"""Rotulo, cuadros, informe y sumas del carril IA de CCTV (spec §4.7 puntos 8, 9, 10 y 12).

La banda bilingue y la marca van en el -vf del encode, despues de los pasos
post-IA, y el comentario en -metadata: el video nunca sale sin rotulo. Los
cuadros exportados salen de ese video, asi que llevan la banda quemada, y se
marcan con XMP DigitalSourceType. El informe tiene la estructura del clasico,
con cada paso atado al proceso y al motor (ffmpeg o el modelo ONNX con su
sha256 y su licencia) que lo corrieron.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from app.models import CctvOptions
from app.services.backend_registry import get_builtin_onnx_model
from app.services.cctv_ai_models import generative_label
from app.services.cctv_chain import AiLanePlan, Limitation, ResolvedStep
from app.services.cctv_enhance_plan import EnhancePlan
from app.services.cctv_ingest import sha256_file
from app.services.cctv_job_runner import IngestedSource, trim_start_of
from app.services.cctv_job_validation import parse_case_details
from app.services.cctv_report import (
    GENERATIVE_UPSCALE,
    HostFacts,
    OutputArtifact,
    ProcessRun,
    ReportParts,
    StepRun,
    ffmpeg_engine,
)
from app.services.cctv_report_model import EngineInfo, OnnxRuntimeInfo
from app.services.ffmpeg_capabilities import FfmpegCapabilities
from app.services.frame_export import StillPair, StillRequest, StillSource
from app.services.handover_package import STILLS_DIRNAME
from app.services.label_band import LabelAssets, band_text, label_band_args, write_label_assets
from app.services.restore_models import RestoreModelSpec

AI_VISUAL_MODE = "ai-visual"
AI_VISUALIZATION_ROLE = "ai-visualization"
ONNX_ENGINE = "onnxruntime"
OSD_ENGINE = "Upflow (nearest-neighbor paste)"
DECODE_LABEL = "AI lane decode (ffmpeg)"
FRAMES_LABEL = "AI lane frames (ONNX Runtime, one thread)"
ENCODE_LABEL = "AI lane encode (ffmpeg)"
AUDIO_LABEL = "AI lane audio (ffmpeg)"
# Todos los builtin del carril son de la familia Real-ESRGAN (THIRD_PARTY_NOTICES.md).
BUILTIN_UPSCALER_LICENSE = "BSD-3-Clause"
OSD_STEP = "osd_protect"


@dataclass(frozen=True, slots=True)
class ModelFile:
    name: str
    path: Path
    license: str


@dataclass(frozen=True, slots=True)
class EnhanceLabel:
    assets: LabelAssets
    args: tuple[str, ...]
    text: str


@dataclass(frozen=True, slots=True)
class EnhanceRuns:
    decode: ProcessRun
    frames: ProcessRun
    encode: ProcessRun
    audio: ProcessRun | None = None

    def processes(self) -> tuple[ProcessRun, ...]:
        runs = (self.audio, self.decode, self.frames, self.encode)
        return tuple(run for run in runs if run is not None)


@dataclass(frozen=True, slots=True)
class EnhanceReportFacts:
    options: CctvOptions
    lane: AiLanePlan
    source: IngestedSource
    caps: FfmpegCapabilities
    host: HostFacts
    osd: Mapping[str, Any]
    runs: EnhanceRuns
    models: Mapping[str, ModelFile]
    outputs: tuple[OutputArtifact, ...]
    stills: tuple[StillPair, ...]
    generative: bool
    app_version: str
    base_dir: Path
    warnings: tuple[str, ...] = ()


# --- Rotulo ---


def write_enhance_label(directory: Path, plan: EnhancePlan, version: str, job_id: str) -> EnhanceLabel:
    directory.mkdir(parents=True, exist_ok=True)
    width, height = plan.encoded_size
    assets = write_label_assets(directory, width, height, version, job_id)
    args = label_band_args(assets, version, job_id, prefix=plan.encode_filters, head_graph=plan.encode_graph)
    return EnhanceLabel(assets, tuple(args), band_text(version, job_id))


def labeled_size(plan: EnhancePlan, label: EnhanceLabel) -> tuple[int, int]:
    width, height = plan.encoded_size
    return width, height + label.assets.band_height


# --- Cuadros exportados ---


def output_times(rate: Fraction, frames: int) -> tuple[float, ...]:
    return tuple(float(n / rate) for n in range(frames))


def enhance_still_request(
    options: CctvOptions,
    lane: AiLanePlan,
    source: IngestedSource,
    enhanced: StillSource,
    job_dir: Path,
    xmp_packet: str,
) -> StillRequest:
    # El stream normaliza a CFR con los fps medidos: el cuadro procesado se busca por tiempo, no por indice.
    return StillRequest(
        StillSource(source.work, source.frame_times, source.keyframes),
        enhanced,
        tuple(sorted(set(options.still_frames))),
        job_dir / STILLS_DIRNAME,
        trim_start_of(lane.decode),
        retimed=True,
        xmp_packet=xmp_packet,
        label_burned_in=True,
    )


# --- Informe ---


def model_engine(model: ModelFile, ort_version: str | None, hash_file: Callable[[Path], str]) -> EngineInfo:
    return EngineInfo(
        name=ONNX_ENGINE,
        version=ort_version,
        model_name=model.name,
        model_sha256=hash_file(model.path),
        model_license=model.license,
    )


def ort_version(host: HostFacts) -> str | None:
    return None if host.onnx_runtime is None else host.onnx_runtime.version


def composite_engine(step_id: str, facts: EnhanceReportFacts, hash_file: Callable[[Path], str]) -> EngineInfo:
    if step_id == OSD_STEP:
        return EngineInfo(name=OSD_ENGINE)
    model = facts.models.get(step_id)
    version = ort_version(facts.host)
    # Sin archivo conocido (un motor sin informe) se declara el runtime, nunca ffmpeg.
    return EngineInfo(name=ONNX_ENGINE, version=version) if model is None else model_engine(model, version, hash_file)


def composite_runs(facts: EnhanceReportFacts, hash_file: Callable[[Path], str]) -> dict[str, StepRun]:
    return {
        step.id: StepRun(facts.runs.frames, composite_engine(step.id, facts, hash_file))
        for step in facts.lane.composite
    }


def ffmpeg_runs(steps: Sequence[ResolvedStep], run: ProcessRun, engine: EngineInfo) -> dict[str, StepRun]:
    return {step.id: StepRun(run, engine) for step in steps}


def enhance_step_runs(facts: EnhanceReportFacts, hash_file: Callable[[Path], str]) -> dict[str, StepRun]:
    engine = ffmpeg_engine(facts.caps)
    return {
        **ffmpeg_runs(facts.lane.decode, facts.runs.decode, engine),
        **composite_runs(facts, hash_file),
        **ffmpeg_runs(facts.lane.encode, facts.runs.encode, engine),
    }


def enhance_limitations(generative: bool) -> tuple[Limitation, ...]:
    return (GENERATIVE_UPSCALE,) if generative else ()


def enhance_report_parts(facts: EnhanceReportFacts, hash_file: Callable[[Path], str] = sha256_file) -> ReportParts:
    acquisition, case = parse_case_details(facts.options)
    source = facts.source
    return ReportParts(
        mode=AI_VISUAL_MODE,
        app_version=facts.app_version,
        commit=None,
        case=case,
        acquisition=acquisition,
        caps=facts.caps,
        host=facts.host,
        source=source.record,
        verified_copy=source.verified,
        ingest=source.ingest,
        diagnosis=None,
        osd=facts.osd,
        chain=(*facts.lane.decode, *facts.lane.composite, *facts.lane.encode),
        chain_run=facts.runs.encode,
        processes=facts.runs.processes(),
        outputs=facts.outputs,
        base_dir=facts.base_dir,
        stills=facts.stills,
        extra_limitations=enhance_limitations(facts.generative),
        warnings=facts.warnings,
        step_runs=enhance_step_runs(facts, hash_file),
    )


def enhanced_artifact(path: Path, probe: Mapping[str, Any] | None, label: str) -> OutputArtifact:
    return OutputArtifact(path, AI_VISUALIZATION_ROLE, probe, ai_applied=True, visible_label=label)


def still_artifacts(stills: Sequence[StillPair], label: str) -> tuple[OutputArtifact, ...]:
    return tuple(
        artifact
        for pair in stills
        for artifact in (
            OutputArtifact(pair.original.path, "still"),
            OutputArtifact(pair.processed.path, "still", ai_applied=True, visible_label=label),
        )
    )


def onnx_runtime_info() -> OnnxRuntimeInfo | None:
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    return OnnxRuntimeInfo(version=ort.__version__, providers=list(ort.get_available_providers()))


# --- Hechos del stream ---


def process_run(
    label: str, argv: Sequence[str], window: tuple[datetime, datetime], frames: tuple[int | None, int | None]
) -> ProcessRun:
    return ProcessRun(label, tuple(argv), window[0], window[1], 0, frames_in=frames[0], frames_out=frames[1])


def stream_runs(
    decode_argv: Sequence[str],
    encode_argv: Sequence[str],
    window: tuple[datetime, datetime],
    frames: tuple[int, int],
    audio: ProcessRun | None,
) -> EnhanceRuns:
    # Decode, etapa compuesta y encode corren a la vez, unidos por colas: comparten la ventana de tiempo.
    frames_in, frames_out = frames
    return EnhanceRuns(
        decode=process_run(DECODE_LABEL, decode_argv, window, (None, frames_in)),
        frames=process_run(FRAMES_LABEL, (), window, (frames_in, frames_out)),
        encode=process_run(ENCODE_LABEL, encode_argv, window, (frames_out, frames_out)),
        audio=audio,
    )


def upscaler_model_file(
    model_id: str, scale: int, precision: str | None, onnx_dir: Path, catalog: Sequence[Mapping[str, Any]]
) -> ModelFile:
    model = get_builtin_onnx_model(model_id, scale)
    if model is None:
        raise RuntimeError(f"{model_id!r} at {scale}x has no ONNX export.")
    filename = model.fp16_filename if precision == "fp16" else model.filename
    label = next((option["label"] for option in catalog if option["key"] == model_id), model_id)
    return ModelFile(f"{label} ({filename})", onnx_dir / filename, BUILTIN_UPSCALER_LICENSE)


def restore_model_file(spec: RestoreModelSpec, path: Path) -> ModelFile:
    return ModelFile(f"{spec.name} ({path.name})", path, spec.license_spdx)


def upscale_json(model_id: str | None, scale: int, generative: bool) -> dict[str, Any] | None:
    if scale == 1 or model_id is None:
        return None
    return {"model": model_id, "scale": scale, "generative": generative, "generativeLabel": generative_label(generative)}
