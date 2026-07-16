from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from app.models import AudioJob, UpscaleJob, VideoUpscaleJob, utc_now
from app.services.media_tools import parse_fps_fraction

StageStatus = Literal["pending", "active", "done"]

VIDEO_STAGE_WEIGHTS: dict[str, tuple[str, float]] = {
    "probing": ("Probing video", 2),
    "extracting_frames": ("Extracting frames", 8),
    "extracting_audio": ("Extracting audio", 4),
    "enhancing_audio": ("Enhancing audio", 4),
    "restoring_audio": ("Restoring audio", 4),
    "upscaling_frames": ("Upscaling frames", 55),
    "interpolating_frames": ("Interpolating frames", 15),
    "encoding_video": ("Encoding video", 13),
}

VIDEO_STAGE_ORDER: tuple[str, ...] = (
    "probing",
    "extracting_frames",
    "extracting_audio",
    "enhancing_audio",
    "restoring_audio",
    "upscaling_frames",
    "interpolating_frames",
    "encoding_video",
)

IMAGE_STAGE_WEIGHTS: dict[str, tuple[str, float]] = {
    "validating": ("Validating image", 10),
    "upscaling": ("Upscaling", 90),
}

IMAGE_STAGE_ORDER: tuple[str, ...] = ("validating", "upscaling")

AUDIO_STAGE_WEIGHTS: dict[str, tuple[str, float]] = {
    "decoding": ("Decoding audio", 10),
    "denoising": ("Denoising", 40),
    "restoring": ("Restoring", 45),
    "finalizing": ("Writing output", 5),
}

AUDIO_STAGE_ORDER: tuple[str, ...] = ("decoding", "denoising", "restoring", "finalizing")


@dataclass(frozen=True, slots=True)
class Stage:
    key: str
    label: str
    weight: float
    status: StageStatus = "pending"


def video_interpolation_active(job: VideoUpscaleJob) -> bool:
    return job.target_fps is not None or job.fps_multiplier > 1


def _video_stage_active(job: VideoUpscaleJob, key: str, has_audio: bool) -> bool:
    if key == "extracting_audio":
        return job.keep_audio and has_audio
    if key == "enhancing_audio":
        return job.keep_audio and has_audio and bool(job.audio_enhance)
    if key == "restoring_audio":
        return job.keep_audio and has_audio and bool(job.audio_restore)
    if key == "interpolating_frames":
        return video_interpolation_active(job)
    return True


def _normalize_weights(raw_stages: list[tuple[str, str, float]]) -> list[Stage]:
    total_weight = sum(weight for _, _, weight in raw_stages)
    return [
        Stage(key=key, label=label, weight=weight / total_weight)
        for key, label, weight in raw_stages
    ]


def build_video_stages(job: VideoUpscaleJob) -> list[Stage]:
    # hasAudio is stamped at probe; default True keeps audio stages when the
    # source has not been probed yet (stages get filtered once it is known).
    has_audio = bool(job.metadata.get("hasAudio", True))
    raw_stages = [
        (key, *VIDEO_STAGE_WEIGHTS[key])
        for key in VIDEO_STAGE_ORDER
        if _video_stage_active(job, key, has_audio)
    ]
    return _normalize_weights(raw_stages)


def build_image_stages() -> list[Stage]:
    raw_stages = [(key, *IMAGE_STAGE_WEIGHTS[key]) for key in IMAGE_STAGE_ORDER]
    return _normalize_weights(raw_stages)


def _audio_stage_active(job: AudioJob, key: str) -> bool:
    if key == "denoising":
        return bool(job.denoise)
    if key == "restoring":
        return bool(job.restore)
    return True


def build_audio_stages(job: AudioJob) -> list[Stage]:
    raw_stages = [
        (key, *AUDIO_STAGE_WEIGHTS[key])
        for key in AUDIO_STAGE_ORDER
        if _audio_stage_active(job, key)
    ]
    return _normalize_weights(raw_stages)


def apply_stage_transition(stages: list[Stage], current_stage_key: str) -> list[Stage]:
    keys = [stage.key for stage in stages]
    if current_stage_key not in keys:
        return stages
    current_index = keys.index(current_stage_key)
    return [
        replace(stage, status=_status_for_index(index, current_index))
        for index, stage in enumerate(stages)
    ]


def _status_for_index(index: int, current_index: int) -> StageStatus:
    if index < current_index:
        return "done"
    if index == current_index:
        return "active"
    return "pending"


def mark_all_done(stages: list[Stage]) -> list[Stage]:
    return [replace(stage, status="done") for stage in stages]


def compute_progress(stages: list[Stage], current_fraction: float = 0.0) -> float:
    done_weight = sum(stage.weight for stage in stages if stage.status == "done")
    active_weight = sum(stage.weight for stage in stages if stage.status == "active")
    clamped_fraction = min(1.0, max(0.0, current_fraction))
    return min(1.0, done_weight + active_weight * clamped_fraction)


def _parse_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parse_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def resolve_frames_total(probe: dict[str, Any], video_stream: dict[str, Any], fps: str) -> int | None:
    # None (not a guess) when nb_frames is missing/invalid (common on VFR) and
    # duration is also unusable -- callers must treat that as "no honest ETA".
    nb_frames = _parse_positive_int(video_stream.get("nb_frames"))
    if nb_frames is not None:
        return nb_frames

    duration = _parse_positive_float(probe.get("format", {}).get("duration"))
    if duration is None:
        return None
    fps_fraction = parse_fps_fraction(fps)
    if fps_fraction is None:
        return None
    return round(duration * float(fps_fraction))


def frame_stage_fraction(frames_done: int, frames_total: int | None) -> float:
    # framesTotal is None on VFR/unknown sources -- no honest fraction to report,
    # compute_progress then keeps the active stage at its floor (no faked ETA).
    if not frames_total:
        return 0.0
    return frames_done / frames_total


def _write_stage_metadata(
    job: UpscaleJob | VideoUpscaleJob | AudioJob,
    stages: list[Stage],
    stage_key: str,
    progress_override: float | None = None,
) -> None:
    job.metadata["stage"] = stage_key
    job.metadata["stages"] = [asdict(stage) for stage in stages]
    job.metadata["progress"] = (
        progress_override if progress_override is not None else compute_progress(stages)
    )
    job.metadata["stageStartedAt"] = utc_now().isoformat()
    job.metadata.setdefault("framesDone", 0)
    job.metadata.setdefault("framesTotal", None)


def advance_video_stage(job: VideoUpscaleJob, stage_key: str) -> None:
    stages = apply_stage_transition(build_video_stages(job), stage_key)
    _write_stage_metadata(job, stages, stage_key)


def complete_video_stages(job: VideoUpscaleJob) -> None:
    stages = mark_all_done(build_video_stages(job))
    _write_stage_metadata(job, stages, "completed", progress_override=1.0)


def advance_image_stage(job: UpscaleJob, stage_key: str) -> None:
    stages = apply_stage_transition(build_image_stages(), stage_key)
    _write_stage_metadata(job, stages, stage_key)


def complete_image_stages(job: UpscaleJob) -> None:
    stages = mark_all_done(build_image_stages())
    _write_stage_metadata(job, stages, "completed", progress_override=1.0)


def advance_audio_stage(job: AudioJob, stage_key: str) -> None:
    stages = apply_stage_transition(build_audio_stages(job), stage_key)
    _write_stage_metadata(job, stages, stage_key)


def complete_audio_stages(job: AudioJob) -> None:
    stages = mark_all_done(build_audio_stages(job))
    _write_stage_metadata(job, stages, "completed", progress_override=1.0)


def apply_image_tile_progress(job: UpscaleJob, tiles_done: int, tiles_total: int) -> None:
    # Called from the ONNX engine's worker thread between tiles (see
    # onnx_upscaler._upscale_tiled) -- only ever invoked for the tiled path,
    # so tiles_total is always >= 2 and framesTotal is never a fake "1/1".
    stages = apply_stage_transition(build_image_stages(), "upscaling")
    fraction = frame_stage_fraction(tiles_done, tiles_total)
    job.metadata["stage"] = "upscaling"
    job.metadata["stages"] = [asdict(stage) for stage in stages]
    job.metadata["framesDone"] = tiles_done
    job.metadata["framesTotal"] = tiles_total
    job.metadata["progress"] = compute_progress(stages, current_fraction=fraction)
