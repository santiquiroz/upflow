from __future__ import annotations

import json
import subprocess
import threading
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.models import (
    CctvOptions,
    CctvStep,
    JobStatus,
    RedactionKeyframe,
    RedactionRequest,
    RedactionTrack,
    VideoUpscaleJob,
)
from app.schemas_cctv import CctvJobRequest, cctv_options, cctv_summary
from app.services import cctv_session
from app.services.cctv_artifacts import listed_artifacts
from app.services.cctv_chain import CctvChainError, ResolvedStep
from app.services.cctv_clarify_runner import FrameCountMismatch
from app.services.cctv_job_runner import build_cctv_runners
from app.services.ffmpeg_filters import FrameGeometry
from app.services.handover_package import check_files_unchanged, checksum_paths as package_checksum_paths
from app.services.label_band import AI_LABEL, LabelAssets, metadata_comment
from app.services.progress import build_cctv_stages
from app.services.redaction import (
    REDACTED_DIRNAME,
    REDACTED_LABEL,
    REDACTION_FRAMES,
    REDACTION_INVALID,
    REDACTION_OUTSIDE_FRAME,
    REDACTION_REQUIRED,
    REDACTION_STEPS,
    REDACTION_UNEXPECTED,
    RedactionLogFacts,
    blur_region,
    box_at,
    boxes_at,
    check_redaction,
    check_redaction_steps,
    pixelate_region,
    redact_frame,
    redaction_log,
)
from app.services.redaction_runner import (
    EncodeTarget,
    PassthroughFrameSource,
    RedactionPlan,
    RedactionRunner,
    build_encode_command,
    check_frame_count,
    redacted_name,
    render_redacted,
)
from ffmpeg_support import needs_ffmpeg
from test_video_job_manager_cctv import (
    FRAMES,
    HEIGHT,
    TOKEN,
    WIDTH,
    analyzed_session,
    make_manager,
    make_settings,
    real_manager,
    rejected,
    write_fake_session,
)

GEOMETRY = FrameGeometry(WIDTH, HEIGHT, Fraction(1))


def track(first: int, last: int, *keys: tuple[int, tuple[int, int, int, int]]) -> RedactionTrack:
    return RedactionTrack(first, last, tuple(RedactionKeyframe(frame, box) for frame, box in keys))


MOVING = track(10, 30, (12, (0, 0, 20, 20)), (22, (100, 50, 40, 30)))
STATIC = track(0, FRAMES - 1, (0, (200, 100, 40, 40)))


def redact_options(**overrides) -> CctvOptions:
    base = {"task": "redact", "session_token": TOKEN, "redaction": RedactionRequest((STATIC,))}
    return CctvOptions(**{**base, **overrides})


def noisy_frame(height: int = 64, width: int = 96, seed: int = 7) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def laplacian_variance(region: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(region, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


# --- Cajas por cuadro ---


def test_a_box_is_held_before_its_first_keyframe_and_after_its_last_inside_its_span() -> None:
    assert box_at(MOVING, 10) == (0, 0, 20, 20)
    assert box_at(MOVING, 30) == (100, 50, 40, 30)


def test_a_box_is_absent_outside_its_span() -> None:
    assert box_at(MOVING, 9) is None and box_at(MOVING, 31) is None


def test_a_box_moves_in_a_straight_line_between_keyframes_rounding_halves_up() -> None:
    assert box_at(MOVING, 17) == (50, 25, 30, 25)
    halfway = track(0, 2, (0, (0, 0, 2, 2)), (2, (1, 1, 3, 3)))
    assert box_at(halfway, 1) == (1, 1, 3, 3)


def test_boxes_at_collects_every_track_that_covers_the_frame() -> None:
    assert boxes_at((MOVING, STATIC), 5) == ((200, 100, 40, 40),)
    assert len(boxes_at((MOVING, STATIC), 20)) == 2


# --- Pixelado y desenfoque ---


def test_pixelation_leaves_a_few_flat_cells_and_the_rest_of_the_frame_untouched() -> None:
    frame = noisy_frame()
    before = frame.copy()
    redacted = redact_frame(frame, ((10, 8, 48, 36),), "pixelate")

    assert np.array_equal(frame, before)
    region = redacted[8:44, 10:58]
    assert len(np.unique(region.reshape(-1, 3), axis=0)) <= 6 * 5
    outside = redacted.copy()
    outside[8:44, 10:58] = before[8:44, 10:58]
    assert np.array_equal(outside, before)


def test_blur_removes_the_fine_detail_of_the_region() -> None:
    region = noisy_frame(80, 80)
    assert laplacian_variance(blur_region(region)) < laplacian_variance(region) * 0.02


def test_even_a_large_box_keeps_only_a_handful_of_pixelated_cells() -> None:
    region = noisy_frame(300, 300)
    cells = np.unique(pixelate_region(region).reshape(-1, 3), axis=0)
    assert len(cells) <= 36


def test_a_frame_without_boxes_is_returned_as_is() -> None:
    frame = noisy_frame()
    assert redact_frame(frame, (), "blur") is frame


# --- Validacion ---


@pytest.mark.parametrize(
    ("request_", "code"),
    [
        (RedactionRequest((STATIC,), "black"), REDACTION_INVALID),
        (RedactionRequest(()), REDACTION_INVALID),
        (RedactionRequest((STATIC,) * 33), REDACTION_INVALID),
        (RedactionRequest((track(0, FRAMES, (0, (0, 0, 10, 10))),)), REDACTION_FRAMES),
        (RedactionRequest((track(5, 4, (5, (0, 0, 10, 10))),)), REDACTION_FRAMES),
        (RedactionRequest((track(0, 9, (5, (0, 0, 10, 10)), (5, (2, 2, 10, 10))),)), REDACTION_FRAMES),
        (RedactionRequest((track(0, 9, (6, (0, 0, 10, 10)), (3, (2, 2, 10, 10))),)), REDACTION_FRAMES),
        (RedactionRequest((track(4, 9, (3, (0, 0, 10, 10))),)), REDACTION_FRAMES),
        (RedactionRequest((track(0, 9, (0, (WIDTH - 5, 0, 10, 10))),)), REDACTION_OUTSIDE_FRAME),
        (RedactionRequest((track(0, 9, (0, (0, 0, 1, 10))),)), REDACTION_OUTSIDE_FRAME),
    ],
)
def test_a_malformed_redaction_is_rejected_with_its_key(request_: RedactionRequest, code: str) -> None:
    with pytest.raises(CctvChainError) as caught:
        check_redaction(redact_options(redaction=request_), GEOMETRY, FRAMES)
    assert caught.value.code == code


def test_the_redacted_copy_needs_boxes_and_other_tasks_refuse_them() -> None:
    with pytest.raises(CctvChainError) as missing:
        check_redaction(redact_options(redaction=None), GEOMETRY, FRAMES)
    with pytest.raises(CctvChainError) as unexpected:
        check_redaction(CctvOptions(task="clarify", session_token=TOKEN, redaction=RedactionRequest((STATIC,))), GEOMETRY, FRAMES)
    assert (missing.value.code, unexpected.value.code) == (REDACTION_REQUIRED, REDACTION_UNEXPECTED)


def test_a_valid_redaction_passes() -> None:
    check_redaction(redact_options(redaction=RedactionRequest((MOVING, STATIC), "blur")), GEOMETRY, FRAMES)


def resolved(step_id: str) -> ResolvedStep:
    return ResolvedStep(step_id, step_id)


def test_the_redacted_copy_accepts_only_a_trim() -> None:
    check_redaction_steps(redact_options(), (resolved("trim"),))
    for steps, stills in (((resolved("denoise"),), ()), ((resolved("osd_protect"),), ()), ((), (3,))):
        with pytest.raises(CctvChainError) as caught:
            check_redaction_steps(redact_options(still_frames=stills), steps)
        assert caught.value.code == REDACTION_STEPS


async def test_the_manager_validates_and_admits_a_redaction_job_on_the_cpu(tmp_path: Path) -> None:
    manager = make_manager(tmp_path, tasks=("clarify", "roi_fusion", "redact"))
    write_fake_session(manager.settings)
    outside = RedactionRequest((track(0, 9, (0, (WIDTH - 4, 0, 10, 10))),))
    assert (await rejected(manager, redact_options(redaction=outside))).code == REDACTION_OUTSIDE_FRAME
    denoise = (CctvStep("denoise", {"filter": "hqdn3d"}),)
    assert (await rejected(manager, redact_options(steps=denoise))).code == REDACTION_STEPS
    job = await manager.create_cctv_job(cctv=redact_options(trim=(5, 60)), device="dml:0")
    assert job.device == "cpu" and job.metadata["cctv"]["lane"] == "classic"


# --- Pedido de la ruta y resumen ---


def test_the_route_request_becomes_redaction_options() -> None:
    body = CctvJobRequest.model_validate(
        {
            "token": TOKEN,
            "task": "redact",
            "trim": [2, 40],
            "redaction": {
                "style": "blur",
                "tracks": [{"firstFrame": 10, "lastFrame": 30, "keyframes": [{"frame": 12, "box": [0, 0, 20, 20]}]}],
            },
        }
    )
    options = cctv_options(body)
    assert options.redaction == RedactionRequest((track(10, 30, (12, (0, 0, 20, 20))),), "blur")
    assert options.trim == (2, 40)


def test_the_route_request_caps_the_boxes() -> None:
    tracks = [{"firstFrame": 0, "lastFrame": 1, "keyframes": [{"frame": 0, "box": [0, 0, 4, 4]}]}] * 33
    with pytest.raises(ValueError):
        CctvJobRequest.model_validate({"token": TOKEN, "task": "redact", "redaction": {"tracks": tracks}})


def completed_redact_job() -> VideoUpscaleJob:
    job = VideoUpscaleJob(
        source_path=Path("clip.mp4"), original_filename="clip.mp4", model_name="cctv-redact", scale=1,
        output_container="mp4", video_codec="h264", video_preset="medium", crf=16, keep_audio=False,
        cctv=redact_options(),
    )  # fmt: skip
    job.status = JobStatus.completed
    job.metadata["cctv"] = {
        "task": "redact",
        "lane": "classic",
        "outputs": {"redacted": "05_redacted/clip__upflow-redacted__j.mp4", "redaction": "redaction.json", "package": None},
        "redaction": {"style": "pixelate", "boxes": 1},
    }
    return job


def test_a_redaction_job_lists_its_copy_and_log_but_no_report_or_package() -> None:
    job = completed_redact_job()
    assert listed_artifacts(job) == ["redacted", "redaction", "sha256sums", "frame_index"]
    summary = cctv_summary(job)
    assert summary.redaction == {"style": "pixelate", "boxes": 1}
    assert summary.verify_url is not None


def test_the_redaction_job_has_its_own_stages() -> None:
    assert [stage.key for stage in build_cctv_stages(completed_redact_job())] == ["ingesting", "redacting", "reporting"]


def test_the_redaction_runner_is_registered(tmp_path: Path) -> None:
    assert isinstance(build_cctv_runners(make_settings(tmp_path))["redact"], RedactionRunner)


# --- Nombres, rotulo, comandos y paquete ---


def test_the_redacted_file_never_takes_the_original_name() -> None:
    assert redacted_name("cámara 1.mp4", "abc123") == "camara 1__upflow-redacted__abc123.mp4"


def test_the_label_says_redacted_copy_and_the_ai_label_is_unchanged() -> None:
    assert metadata_comment("1.0", "j") == "AI-enhanced visualization by Upflow 1.0 (job j); not original footage"
    assert "Redacted copy" in metadata_comment("1.0", "j", REDACTED_LABEL)
    assert REDACTED_LABEL.band_en.startswith("REDACTED COPY") and AI_LABEL.band_en.startswith("AI-ENHANCED")


def plan(first: int = 0, last: int = 9, prefilters: tuple[str, ...] = ()) -> RedactionPlan:
    return RedactionPlan(Path("work.mkv"), GEOMETRY, first, last, Fraction(25), RedactionRequest((MOVING,)), prefilters)


def test_the_decode_keeps_every_frame_and_the_encode_is_labeled_without_audio() -> None:
    source = PassthroughFrameSource(Path("ffmpeg"), Path("work.mkv"), WIDTH, HEIGHT, 2, "25/1", ("trim=start_frame=3:end_frame=9",))
    decode = source.build_command()
    assert decode[decode.index("-fps_mode") + 1] == "passthrough" and "cfr" not in decode
    assert decode[decode.index("-vf") + 1] == "trim=start_frame=3:end_frame=9"
    assets = LabelAssets(Path("band.png"), Path("mark.png"), 24, 4)
    target = EncodeTarget(Path("ffmpeg"), Path("out.mp4"), 4, "0.81.0", "job1")
    encode = build_encode_command(target, plan(), assets)
    graph = encode[encode.index("-vf") + 1]
    assert graph.startswith("setsar=1/1,null[lb_image]") and "pad=w=iw:h=ih+24" in graph
    assert "-an" in encode and encode[encode.index("-framerate") + 1] == "25/1"
    assert encode[encode.index("-metadata") + 1].startswith("comment=Redacted copy made by Upflow 0.81.0 (job job1)")


def test_an_odd_frame_is_padded_to_even_sides_for_h264() -> None:
    odd = RedactionPlan(Path("w.mkv"), FrameGeometry(321, 241, Fraction(2)), 0, 1, Fraction(25), RedactionRequest((STATIC,)), ())
    encode = build_encode_command(EncodeTarget(Path("f"), Path("o.mp4"), 1, "v", "j"), odd, LabelAssets(Path("b"), Path("m"), 2, 2))
    assert encode[encode.index("-vf") + 1].startswith("pad=w=322:h=242:x=0:y=0:color=black,setsar=2/1")


def test_the_redacted_folder_is_never_part_of_the_handover_package(tmp_path: Path) -> None:
    (tmp_path / "01_original").mkdir()
    (tmp_path / "01_original" / "clip.mp4").write_bytes(b"x")
    (tmp_path / REDACTED_DIRNAME).mkdir()
    (tmp_path / REDACTED_DIRNAME / "clip__upflow-redacted__j.mp4").write_bytes(b"y")
    assert package_checksum_paths(tmp_path) == ("01_original/clip.mp4",)


def test_the_log_records_the_boxes_the_source_and_the_notice() -> None:
    facts = RedactionLogFacts("0.81.0", "j", "clip.mp4", "ab" * 32, 0, 9, 10, "25/1", "05_redacted/x.mp4", "cd" * 32)
    log = redaction_log(RedactionRequest((MOVING,), "blur"), facts)
    assert log["tracks"][0]["keyframes"][1] == {"frame": 22, "box": [100, 50, 40, 30]}
    assert log["source"]["sha256"] == "ab" * 32 and log["output"]["audio"] is False
    assert "not part of the handover package" in log["notice"]


# --- Cuadros hacia el encoder ---


class FakeWriter:
    def __init__(self, fail_at: int | None = None) -> None:
        self.frames: list[np.ndarray] = []
        self.events: list[str] = []
        self.fail_at = fail_at

    def start(self) -> None:
        self.events.append("start")

    def write_frame(self, frame_hwc: np.ndarray) -> None:
        if self.fail_at is not None and len(self.frames) == self.fail_at:
            raise RuntimeError("encoder exploded")
        self.frames.append(frame_hwc)

    def finish(self) -> None:
        self.events.append("finish")

    def kill(self) -> None:
        self.events.append("kill")


def frames_of(count: int) -> list[np.ndarray]:
    return [noisy_frame(HEIGHT, WIDTH, seed=n)[np.newaxis, ...] for n in range(count)]


def iterate(frames: list[np.ndarray]):
    yield from frames


def test_frames_are_numbered_from_the_trim_start_so_boxes_match_the_index() -> None:
    writer, seen = FakeWriter(), []
    written = render_redacted(iterate(frames_of(4)), plan(first=9, last=12), writer, threading.Event(), seen.append)
    assert written == 4 and seen == [1, 2, 3, 4] and writer.events == ["start", "finish"]
    assert np.array_equal(writer.frames[0], frames_of(1)[0][0])
    changed = [not np.array_equal(out, original[0]) for out, original in zip(writer.frames, frames_of(4))]
    assert changed == [False, True, True, True]


def test_a_cancelled_render_kills_the_encoder() -> None:
    cancel = threading.Event()
    cancel.set()
    writer = FakeWriter()
    render_redacted(iterate([]), plan(), writer, cancel, lambda _: None)
    assert writer.events == ["start", "kill"]


def test_an_encoder_failure_kills_it_and_propagates() -> None:
    writer = FakeWriter(fail_at=1)
    with pytest.raises(RuntimeError, match="exploded"):
        render_redacted(iterate(frames_of(3)), plan(), writer, threading.Event(), lambda _: None)
    assert writer.events == ["start", "kill"]


def test_a_short_decode_fails_instead_of_shifting_the_boxes() -> None:
    with pytest.raises(FrameCountMismatch):
        check_frame_count(plan(first=0, last=9), 8)


# --- De punta a punta con ffmpeg real ---


def ffprobe_json(settings, path: Path) -> dict:
    command = [
        str(settings.ffprobe_binary_path), "-v", "error", "-count_frames", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ]  # fmt: skip
    return json.loads(subprocess.run(command, check=True, capture_output=True).stdout)


def decoded_frame(settings, path: Path, frame: int, width: int, height: int) -> np.ndarray:
    command = [
        str(settings.ffmpeg_binary_path), "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{frame})",
        "-frames:v", "1", "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]  # fmt: skip
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)


@needs_ffmpeg
async def test_a_real_redaction_job_writes_a_labeled_copy_outside_the_package(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    session = await analyzed_session(settings, TOKEN)
    manager = real_manager(settings)
    box = (200, 20, 96, 64)
    request = RedactionRequest((track(10, 30, (10, box)), track(0, 49, (0, (0, 0, 40, 40)), (49, (40, 40, 40, 40)))))
    job = await manager.create_cctv_job(cctv=redact_options(redaction=request, trim=(5, 44)))

    await manager._process_next()

    assert job.status == JobStatus.completed, job.error
    job_dir = cctv_session.cctv_job_dir(settings.outputs_path, job.id)
    output = job.output_path
    assert output.parent == job_dir / REDACTED_DIRNAME and "__upflow-redacted__" in output.name
    assert not list(job_dir.glob("*.zip")) and not (job_dir / "report.json").exists()
    assert check_files_unchanged(job_dir).ok
    sums = (job_dir / "SHA256SUMS.txt").read_text("utf-8")
    assert f"*{REDACTED_DIRNAME}/{output.name}" in sums and "*redaction.json" in sums
    log = json.loads((job_dir / "redaction.json").read_text("utf-8"))
    assert log["frames"] == {"first": 5, "last": 44, "out": 40}
    assert job.metadata["stage"] == "completed" and job.metadata["progress"] == 1.0

    probe = ffprobe_json(settings, output)
    streams = probe["streams"]
    assert [stream["codec_type"] for stream in streams] == ["video"]
    video = streams[0]
    assert int(video["nb_read_frames"]) == 40 and video["width"] == WIDTH and video["height"] > HEIGHT
    assert "Redacted copy made by Upflow" in probe["format"]["tags"]["comment"]

    x, y, w, h = box
    work = session / "work.mkv"
    original = decoded_frame(settings, work, 20, WIDTH, HEIGHT)[y : y + h, x : x + w]
    redacted = decoded_frame(settings, output, 15, video["width"], video["height"])[y : y + h, x : x + w]
    assert mean_abs_diff(redacted, pixelate_region(original)) * 5 < mean_abs_diff(redacted, original)
    untouched = decoded_frame(settings, output, 0, video["width"], video["height"])[y : y + h, x : x + w]
    source_start = decoded_frame(settings, work, 5, WIDTH, HEIGHT)[y : y + h, x : x + w]
    assert mean_abs_diff(untouched, source_start) * 5 < mean_abs_diff(untouched, pixelate_region(source_start))
