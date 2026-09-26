from __future__ import annotations

import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.services import cctv_enhance_plan as plan_module
from app.services.cctv_chain import ai_lane_plan, steps_from_request
from app.services.cctv_frame_index import FrameEntry, summarize_frame_index
from app.services.cctv_ingest import ReceivedAt, SourceRecord
from app.services.engines.frame_model_runner import FrameModelReport
from app.services.engines.frame_restorer import ComposedStageReport
from app.services.ffmpeg_filters import FrameGeometry
from ffmpeg_support import needs_ffmpeg

GEOMETRY = FrameGeometry(704, 576)
DEBLOCK = {"id": "ai_deblock", "params": {"filter": "drunet_deblock", "strength": 60}}
TRIM = {"id": "trim", "params": {"filter": "trim", "start_frame": 10, "end_frame": 19}}
CROP = {"id": "crop", "params": {"filter": "crop", "x": 100, "y": 50, "w": 320, "h": 240}}
GRAY = {"id": "gray", "params": {}}
OSD = {"id": "osd_protect", "params": {}}
UPSCALE = {"id": "ai_upscale", "params": {}}


def lane(*raw: dict):
    return ai_lane_plan(steps_from_request(list(raw), "ai"))


def deinterlace(mode: str) -> dict:
    return {"id": "deinterlace", "params": {"filter": "bwdif", "mode": mode}}


def frames(count: int, delta: float) -> tuple[FrameEntry, ...]:
    return tuple(FrameEntry(n, round(n * delta, 6), n % 25 == 0, "P", 900) for n in range(count))


def source(count: int = 50, delta: float = 0.04, header: str | None = "25/1") -> plan_module.EnhanceSource:
    return plan_module.EnhanceSource(GEOMETRY, summarize_frame_index(frames(count, delta)), header)


def build(*raw: dict, src: plan_module.EnhanceSource | None = None, boxes=(), scale: int = 1):
    return plan_module.build_enhance_plan(lane(*raw), src or source(), boxes, scale)


def test_cctv_plan_without_prefilters_decodes_the_whole_frame_at_the_measured_rate() -> None:
    plan = build(DEBLOCK)

    assert plan.prefilter_args == ()
    assert (plan.decoded.width, plan.decoded.height) == (704, 576)
    assert plan.rate_text == "25/1" and plan.frames_in == 50
    assert plan.strength == 60 and plan.upscale == 1 and plan.output_size == (704, 576)


def test_cctv_trim_rebases_timestamps_so_cfr_does_not_pad_the_start() -> None:
    plan = build(TRIM, DEBLOCK, GRAY)

    assert plan.prefilter_args == ("-vf", "trim=start_frame=10:end_frame=20,setpts=PTS-STARTPTS,format=pix_fmts=gray")
    assert plan.frames_in == 10


def test_cctv_crop_sets_the_decoded_size_before_the_source_exists() -> None:
    plan = build(CROP, DEBLOCK)

    assert (plan.decoded.width, plan.decoded.height) == (320, 240)
    assert plan.prefilter_args == ("-vf", "crop=w=320:h=240:x=100:y=50:exact=1")


def test_cctv_osd_boxes_move_with_the_crop_into_decoded_coordinates() -> None:
    plan = build(CROP, DEBLOCK, OSD, boxes=((120, 60, 40, 20),))

    assert plan.osd_boxes == ((20, 10, 40, 20),)


DENOISE = {"id": "denoise", "params": {"filter": "hqdn3d", "luma_spatial": 4}}
CLASSIC_DEBLOCK = {"id": "deblock", "params": {"filter": "deblock"}}


def osd_branch(graph: str) -> str:
    return next(segment for segment in graph.split(";") if segment.startswith("[o]"))


def test_cctv_osd_boxes_are_decoded_from_a_branch_without_deblock_or_denoise() -> None:
    plan = build(TRIM, CROP, DENOISE, CLASSIC_DEBLOCK, OSD, UPSCALE, boxes=((120, 60, 40, 20),), scale=2)

    graph = plan.prefilter_args[1]

    assert graph.startswith("trim=start_frame=10:end_frame=20,setpts=PTS-STARTPTS[t];[t]split=2[m][o];")
    assert "hqdn3d" not in osd_branch(graph) and "deblock" not in osd_branch(graph)
    assert "hqdn3d" in graph and "deblock=" in graph
    assert osd_branch(graph) == "[o]crop=w=320:h=240:x=100:y=50:exact=1,crop=w=40:h=20:x=20:y=10:exact=1[osd0]"
    assert graph.endswith("overlay=x=20:y=10:format=auto:shortest=1:repeatlast=0[out]")


def test_cctv_without_boxes_the_decode_stays_a_plain_chain() -> None:
    plan = build(CROP, DENOISE, OSD, UPSCALE, scale=2)

    assert ";" not in plan.prefilter_args[1]


def test_cctv_post_ai_levels_and_sharpen_do_not_touch_the_pasted_osd() -> None:
    plan = build(OSD, UPSCALE, LEVELS, SHARPEN, boxes=((120, 60, 40, 20),), scale=2)

    graph = plan.encode_graph

    assert graph is not None and graph.startswith("[in]split=2[m][o];")
    assert osd_branch(graph) == "[o]crop=w=80:h=40:x=240:y=120:exact=1[osd0]"
    assert graph.endswith("overlay=x=240:y=120:format=rgb:shortest=1:repeatlast=0[lb_image]")


@needs_ffmpeg
def test_cctv_real_encode_graph_keeps_the_osd_pixels_while_levels_change_the_rest() -> None:
    plan = build(OSD, UPSCALE, LEVELS, SHARPEN, boxes=((120, 60, 40, 20),), scale=2)
    width, height = plan.output_size
    frame = np.random.default_rng(1).integers(0, 256, (height, width, 3), dtype=np.uint8)
    command = [
        str(Settings().ffmpeg_binary_path), "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-i", "pipe:0", "-vf", f"{plan.encode_graph};[lb_image]null",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ]  # fmt: skip

    done = subprocess.run(command, input=frame.tobytes(), capture_output=True, check=True)

    result = np.frombuffer(done.stdout, np.uint8).reshape(height, width, 3)
    box = (slice(120, 160), slice(240, 320))
    np.testing.assert_array_equal(result[box], frame[box])
    assert not np.array_equal(result[:100], frame[:100])


def test_cctv_without_post_ai_steps_there_is_no_encode_graph() -> None:
    assert build(OSD, UPSCALE, boxes=((120, 60, 40, 20),), scale=2).encode_graph is None


def test_cctv_boxes_without_osd_protect_are_not_pasted() -> None:
    assert build(DEBLOCK, boxes=((120, 60, 40, 20),)).osd_boxes == ()


def test_cctv_send_field_doubles_the_rate_and_the_frame_count() -> None:
    plan = build(deinterlace("send_field"), DEBLOCK)

    assert plan.rate == Fraction(50) and plan.frames_in == 100


def test_cctv_send_frame_keeps_the_rate() -> None:
    assert build(deinterlace("send_frame"), DEBLOCK).rate == Fraction(25)


def test_cctv_rate_is_the_measured_one_when_the_container_lies() -> None:
    plan = build(DEBLOCK, src=source(delta=0.08, header="25/1"))

    assert plan.rate_text == "25/2"


def test_cctv_rate_snaps_to_the_header_within_the_millisecond_pts_resolution() -> None:
    # 29.97 fps en Matroska: los deltas quedan en 33/34 ms y la mediana da 30.303 fps.
    plan = build(DEBLOCK, src=source(delta=0.033, header="30000/1001"))

    assert plan.rate_text == "30000/1001"


def test_cctv_rate_without_a_measurement_falls_back_to_the_header() -> None:
    assert build(DEBLOCK, src=source(count=1, header="15/1")).rate_text == "15/1"


def test_cctv_rate_without_measurement_nor_header_fails_clearly() -> None:
    with pytest.raises(RuntimeError, match="frame rate"):
        build(DEBLOCK, src=source(count=1, header=None))


def test_cctv_ai_upscale_multiplies_the_output_size() -> None:
    plan = build(DEBLOCK, UPSCALE, scale=2)

    assert plan.upscale == 2 and plan.output_size == (1408, 1152)


def test_cctv_ai_upscale_without_a_scale_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="scale of 2"):
        build(DEBLOCK, UPSCALE, scale=1)


def test_cctv_without_ai_deblock_there_is_no_strength() -> None:
    assert build({"id": "denoise", "params": {"filter": "hqdn3d"}}).strength is None


def test_cctv_audio_is_trimmed_at_the_frame_timestamps_and_rebased() -> None:
    decode = lane(TRIM, DEBLOCK).decode
    atrim = plan_module.audio_trim(decode, frames(50, 0.04))

    command = plan_module.build_audio_command(Path("ffmpeg.exe"), Path("work.mkv"), atrim, Path("audio.m4a"))

    assert command[command.index("-af") + 1] == "atrim=start=0.4:end=0.8,asetpts=PTS-STARTPTS"
    assert command[command.index("-map") + 1] == "0:a:0" and command[-1] == "audio.m4a"


def test_cctv_audio_without_trim_has_no_filter() -> None:
    atrim = plan_module.audio_trim(lane(DEBLOCK).decode, frames(50, 0.04))

    command = plan_module.build_audio_command(Path("ffmpeg.exe"), Path("work.mkv"), atrim, Path("audio.m4a"))

    assert atrim is None and "-af" not in command


def test_cctv_enhance_metadata_records_the_stream_and_keeps_the_admission_fields() -> None:
    plan = build(TRIM, CROP, DEBLOCK)
    restore = FrameModelReport("drunet-deblock-color-u8", "dml:0", "fp16", None, True)
    report = ComposedStageReport(frames=12, duplicates_reused=3, upscaled=False, osd_boxes=0, restore=restore)
    record = SourceRecord("cam.mp4", 10, "2026-09-01T00:00:00Z", "ab" * 32, ReceivedAt("u", "l"), "MP4/MOV")
    base = {"task": "enhance", "lane": "ai", **plan_module.source_json(record)}

    stream = plan_module.stream_json(plan, report, 12)
    metadata = plan_module.enhance_metadata(base, stream, "02_processed/enhanced.mp4", ["cctv.lite", "cctv.lite"])

    assert metadata["lane"] == "ai" and metadata["sourceSha256"] == "ab" * 32
    assert metadata["receivedAt"] == {"utc": "u", "local": "l"}
    assert metadata["cfrNormalized"] is True and metadata["measuredFps"] == "25/1"
    assert (metadata["framesIn"], metadata["framesOut"], metadata["duplicatesReused"]) == (10, 12, 3)
    assert metadata["decodedSize"] == [320, 240] and metadata["outputSize"] == [320, 240]
    assert metadata["restore"]["model"] == "drunet-deblock-color-u8" and metadata["restore"]["ioBinding"] is True
    assert metadata["outputs"] == {"enhanced": "02_processed/enhanced.mp4"}
    assert metadata["warnings"] == ["cctv.lite"]


LEVELS = {"id": "levels", "params": {"filter": "eq", "gamma": 1.2}}
SCALE = {"id": "scale", "params": {"filter": "scale", "factor": 2}}
SHARPEN = {"id": "sharpen", "params": {"filter": "cas", "strength": 0.4}}


def test_cctv_ai_lane_puts_levels_scale_and_sharpen_after_the_ai_in_catalog_order() -> None:
    plan = build(SHARPEN, SCALE, DEBLOCK, LEVELS, scale=2)

    assert plan.encode_filters == (
        "eq=brightness=0.0:contrast=1.0:gamma=1.2:saturation=1.0",
        "scale=w=iw*2:h=ih*2:flags=neighbor+accurate_rnd+full_chroma_int+bitexact",
        "cas=strength=0.4",
    )
    assert plan.prefilter_args == ()


def test_cctv_ai_lane_encoded_size_follows_the_classic_scale_after_the_ai_upscale() -> None:
    plan = build(CROP, DEBLOCK, UPSCALE, SCALE, scale=2)

    assert plan.output_size == (640, 480)
    assert plan.encoded_size == (1280, 960)


def test_cctv_ai_label_never_becomes_an_ffmpeg_filter_of_the_chain() -> None:
    plan = build(DEBLOCK)

    assert plan.encode_filters == ()
    assert plan.encoded_size == plan.output_size == (704, 576)
