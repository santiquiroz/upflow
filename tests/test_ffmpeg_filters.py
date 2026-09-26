from __future__ import annotations

import json
import subprocess
from fractions import Fraction
from pathlib import Path, PureWindowsPath
from types import MappingProxyType

import numpy as np
import pytest

from app.config import Settings
from app.services.cctv_chain import (
    CCTV_CHAIN,
    INVALID_PARAM,
    MISSING_PARAM,
    CctvChainError,
    ResolvedStep,
    steps_from_request,
)
from app.services.ffmpeg_filters import (
    CROP_OUTSIDE_FRAME,
    NOT_A_FILTER,
    OSD_BOX_OUTSIDE_FRAME,
    FrameGeometry,
    build_filter,
    build_osd_graph,
    build_scale_to,
    compose_vf,
    escape_filter_path,
    osd_boxes_after,
    output_dims_after,
    scale_flags,
    vf_args,
)
from ffmpeg_support import needs_ffmpeg

EXACT_FLAGS = "+accurate_rnd+full_chroma_int+bitexact"
REQUIRED_PARAMS = {
    "trim": {"start_frame": 2, "end_frame": 5},
    "aspect": {"num": 2, "den": 1},
    "crop": {"w": 32, "h": 24, "x": 3, "y": 5},
}
CLIP_SIZE = (64, 48)
CLIP_FRAMES = 10
TRICKY_DIR = "osd dir, [x]; it's"


def classic(*raw: dict) -> tuple[ResolvedStep, ...]:
    return steps_from_request(list(raw), "classic")


def one(step_id: str, **params) -> ResolvedStep:
    return classic({"id": step_id, "params": params})[0]


def by_hand(step_id: str, filter_name: str, **params) -> ResolvedStep:
    return ResolvedStep(step_id, filter_name, MappingProxyType(params))


def classic_filter_cases() -> list[tuple[str, str]]:
    return [
        (step.id, spec.name)
        for step in CCTV_CHAIN
        if "classic" in step.lanes and step.id != "osd_protect"
        for spec in step.filters
    ]


def default_step(step_id: str, filter_name: str) -> ResolvedStep:
    return one(step_id, filter=filter_name, **REQUIRED_PARAMS.get(step_id, {}))


def test_escape_filter_path_escapes_the_drive_colon_for_both_parsing_levels() -> None:
    escaped = escape_filter_path(PureWindowsPath(r"C:\Program Files\Upflow\band.png"))

    assert escaped == r"C\\:/Program Files/Upflow/band.png"


def test_escape_filter_path_escapes_graph_separators_and_quotes() -> None:
    escaped = escape_filter_path("C:/a, [b]; it's/p.png")

    assert escaped == r"C\\:/a\, \[b\]\; it\\\'s/p.png"


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (one("trim", start_frame=2, end_frame=5), "trim=start_frame=2:end_frame=6"),
        (one("aspect", num=4, den=2), "setsar=sar=2/1:max=1000"),
        (one("deinterlace"), "bwdif=mode=send_frame:parity=auto"),
        (one("deblock", filter_type="strong"), "deblock=filter=strong:block=8"),
        (
            one("denoise", filter="hqdn3d", luma_spatial=3, chroma_spatial=2, luma_tmp=4, chroma_tmp=3),
            "hqdn3d=luma_spatial=3:chroma_spatial=2:luma_tmp=4:chroma_tmp=3",
        ),
        (
            one("denoise", filter="atadenoise", **{"0a": 0.04, "0b": 0.08, "1a": 0.04, "1b": 0.08}),
            "atadenoise=0a=0.04:0b=0.08:1a=0.04:1b=0.08:2a=0.02:2b=0.04:s=9",
        ),
        (
            one("denoise", filter="tmedian", radius=2),
            "tpad=start=2:start_mode=clone:stop=2:stop_mode=clone,tmedian=radius=2:percentile=0.5",
        ),
        (one("crop", w=320, h=240, x=11, y=7), "crop=w=320:h=240:x=11:y=7:exact=1"),
        (one("gray"), "format=pix_fmts=gray"),
        (one("levels", gamma=1.2), "eq=brightness=0.0:contrast=1.0:gamma=1.2:saturation=1.0"),
        (one("levels", filter="curves", preset="lighter"), "curves=preset=lighter"),
        (one("scale"), f"scale=w=iw*2:h=ih*2:flags=neighbor{EXACT_FLAGS}"),
        (one("sharpen", strength=0.25), "cas=strength=0.25"),
    ],
)
def test_build_filter_writes_the_ffmpeg_syntax(step: ResolvedStep, expected: str) -> None:
    assert build_filter(step) == expected


@pytest.mark.parametrize("algorithm", ["neighbor+full_chroma_inp", 3, "lanczos:param0=3"])
def test_scale_flags_accept_only_a_plain_algorithm_name(algorithm) -> None:
    with pytest.raises(CctvChainError) as error:
        scale_flags(algorithm)

    assert error.value.code == INVALID_PARAM


@pytest.mark.parametrize("algorithm", ["neighbor", "bicubic", "lanczos"])
def test_every_scale_carries_the_exact_rounding_flags(algorithm: str) -> None:
    built = build_filter(one("scale", factor=3, flags=algorithm))

    assert built == f"scale=w=iw*3:h=ih*3:flags={algorithm}{EXACT_FLAGS}"
    assert scale_flags(algorithm) == f"{algorithm}{EXACT_FLAGS}"


def test_build_scale_to_uses_fixed_dimensions_and_the_exact_flags() -> None:
    assert build_scale_to(1920, 1080) == f"scale=w=1920:h=1080:flags=neighbor{EXACT_FLAGS}"


@pytest.mark.parametrize(("width", "height"), [(0, 1080), (1920, -2), (True, 1080)])
def test_build_scale_to_rejects_sizes_that_are_not_positive_integers(width, height) -> None:
    with pytest.raises(CctvChainError) as error:
        build_scale_to(width, height)

    assert error.value.code == INVALID_PARAM


@pytest.mark.parametrize(
    "step",
    [
        by_hand("osd_protect", "osd_restore"),
        by_hand("ai_deblock", "drunet_deblock", strength=40),
        by_hand("ai_upscale", "onnx_upscale"),
        by_hand("ai_label", "label_band"),
        by_hand("stabilize", "vidstab", shakiness=5, smoothing=10),
        by_hand("lens", "lenscorrection", k1=0.1, k2=0.0),
    ],
)
def test_steps_that_are_not_a_single_ffmpeg_filter_are_refused(step: ResolvedStep) -> None:
    with pytest.raises(CctvChainError) as error:
        build_filter(step)

    assert error.value.code == NOT_A_FILTER


@pytest.mark.parametrize(
    "step",
    [
        by_hand("levels", "curves", preset="lighter,movie=x"),
        by_hand("sharpen", "cas", strength=float("nan")),
        by_hand("sharpen", "cas", strength=True),
        by_hand("crop", "crop", w=10, h=10, x="1:y=2", y=0),
    ],
)
def test_a_hand_built_step_cannot_inject_filtergraph_syntax(step: ResolvedStep) -> None:
    with pytest.raises(CctvChainError) as error:
        build_filter(step)

    assert error.value.code == INVALID_PARAM


def test_a_hand_built_step_without_a_parameter_is_refused() -> None:
    with pytest.raises(CctvChainError) as error:
        build_filter(by_hand("sharpen", "cas"))

    assert error.value.code == MISSING_PARAM


def test_compose_vf_joins_in_catalog_order_regardless_of_input_order() -> None:
    steps = (one("scale"), one("deinterlace"), one("gray"))

    assert compose_vf(steps) == (
        "bwdif=mode=send_frame:parity=auto,format=pix_fmts=gray," f"scale=w=iw*2:h=ih*2:flags=neighbor{EXACT_FLAGS}"
    )


def test_vf_args_emit_a_single_vf_for_the_whole_chain() -> None:
    steps = classic(
        {"id": "deinterlace"},
        {"id": "deblock"},
        {"id": "denoise"},
        {"id": "crop", "params": REQUIRED_PARAMS["crop"]},
        {"id": "levels"},
        {"id": "scale"},
        {"id": "sharpen"},
    )

    args = vf_args(steps)

    assert args.count("-vf") == 1
    assert args == ["-vf", compose_vf(steps)]
    assert args[1].count(",") == len(steps) - 1


def test_vf_args_are_empty_without_steps() -> None:
    assert vf_args(()) == []


def test_output_dims_after_follow_crop_scale_and_sar() -> None:
    steps = classic(
        {"id": "aspect", "params": {"num": 2, "den": 1}},
        {"id": "crop", "params": {"w": 400, "h": 300, "x": 11, "y": 7}},
        {"id": "scale", "params": {"factor": 2}},
    )

    geometry = output_dims_after(steps, 960, 1080)

    assert geometry == FrameGeometry(800, 600, Fraction(2))
    assert geometry.width % 2 == 0 and geometry.height % 2 == 0


def test_output_dims_after_keep_the_source_sar_without_an_aspect_step() -> None:
    assert output_dims_after(classic({"id": "denoise"}), 704, 576, Fraction(12, 11)) == FrameGeometry(
        704, 576, Fraction(12, 11)
    )


@pytest.mark.parametrize(
    "crop",
    [
        one("crop", w=400, h=300, x=600, y=0),
        by_hand("crop", "crop", w=400, h=300, x=-2, y=0),
        by_hand("crop", "crop", w=0, h=300, x=0, y=0),
    ],
)
def test_a_crop_that_leaves_the_frame_is_refused(crop: ResolvedStep) -> None:
    with pytest.raises(CctvChainError) as error:
        output_dims_after((crop,), 960, 1080)

    assert error.value.code == CROP_OUTSIDE_FRAME


def test_output_dims_after_cannot_guess_the_ai_upscale_factor() -> None:
    with pytest.raises(CctvChainError) as error:
        output_dims_after((by_hand("ai_upscale", "onnx_upscale"),), 960, 1080)

    assert error.value.code == NOT_A_FILTER


def test_osd_boxes_follow_the_crop_and_scale_and_snap_outwards_to_even_pixels() -> None:
    steps = classic(
        {"id": "crop", "params": {"w": 40, "h": 30, "x": 3, "y": 5}},
        {"id": "scale", "params": {"factor": 3}},
    )

    placed = osd_boxes_after(steps, [(5, 6, 9, 5)], FrameGeometry(64, 48))

    assert placed == ((6, 2, 28, 16),)


def test_osd_boxes_are_clipped_to_the_crop_and_dropped_when_outside_it() -> None:
    steps = classic({"id": "crop", "params": {"w": 40, "h": 30, "x": 4, "y": 4}})

    placed = osd_boxes_after(steps, [(0, 0, 10, 6), (50, 40, 10, 8)], FrameGeometry(64, 48))

    assert placed == ((0, 0, 6, 2),)


def test_osd_boxes_snapped_to_even_never_leave_an_odd_sized_frame() -> None:
    placed = osd_boxes_after((), [(60, 40, 3, 3)], FrameGeometry(63, 43))

    assert placed == ((60, 40, 3, 3),)


@pytest.mark.parametrize("box", [(60, 40, 10, 10), (-1, 0, 4, 4), (0, 0, 0, 4), (0, 0, 4.5, 4)])
def test_an_osd_box_outside_the_original_frame_is_refused(box) -> None:
    with pytest.raises(CctvChainError) as error:
        osd_boxes_after((), [box], FrameGeometry(64, 48))

    assert error.value.code == OSD_BOX_OUTSIDE_FRAME


def test_without_osd_boxes_the_graph_is_the_plain_chain() -> None:
    steps = classic({"id": "denoise"}, {"id": "osd_protect"})

    graph = build_osd_graph(steps, [], FrameGeometry(64, 48), pix_fmt="yuv420p")

    assert graph == (
        "[0:v]hqdn3d=luma_spatial=4.0:chroma_spatial=3.0:luma_tmp=6.0:chroma_tmp=4.5,format=pix_fmts=yuv420p[v]"
    )


def test_an_empty_chain_without_boxes_is_a_null_graph() -> None:
    assert build_osd_graph((), [], FrameGeometry(64, 48)) == "[0:v]null[v]"


def test_one_osd_box_is_restored_from_the_geometry_only_branch() -> None:
    steps = classic(
        {"id": "trim", "params": {"start_frame": 2, "end_frame": 5}},
        {"id": "deinterlace"},
        {"id": "denoise", "params": {"filter": "tmedian"}},
        {"id": "crop", "params": {"w": 40, "h": 30, "x": 4, "y": 4}},
        {"id": "levels", "params": {"brightness": 0.2}},
        {"id": "osd_protect"},
    )

    graph = build_osd_graph(steps, [(6, 6, 10, 4)], FrameGeometry(64, 48))

    assert graph == (
        "[0:v]trim=start_frame=2:end_frame=6,split=2[m][o];"
        "[m]bwdif=mode=send_frame:parity=auto,"
        "tpad=start=1:start_mode=clone:stop=1:stop_mode=clone,tmedian=radius=1:percentile=0.5,"
        "crop=w=40:h=30:x=4:y=4:exact=1,eq=brightness=0.2:contrast=1.0:gamma=1.0:saturation=1.0[p0];"
        "[o]bwdif=mode=send_frame:parity=auto,crop=w=40:h=30:x=4:y=4:exact=1,"
        "crop=w=10:h=4:x=2:y=2:exact=1[osd0];"
        "[p0][osd0]overlay=x=2:y=2:format=auto:shortest=1:repeatlast=0[v]"
    )


def test_several_osd_boxes_share_one_branch_and_chain_their_overlays() -> None:
    steps = classic({"id": "denoise"}, {"id": "gray"})

    graph = build_osd_graph(steps, [(0, 0, 8, 4), (40, 40, 8, 4)], FrameGeometry(64, 48), pix_fmt="gray")

    assert graph == (
        "[0:v]split=2[m][o];"
        "[m]hqdn3d=luma_spatial=4.0:chroma_spatial=3.0:luma_tmp=6.0:chroma_tmp=4.5,format=pix_fmts=gray[p0];"
        "[o]format=pix_fmts=gray,split=2[o0][o1];"
        "[o0]crop=w=8:h=4:x=0:y=0:exact=1[osd0];"
        "[o1]crop=w=8:h=4:x=40:y=40:exact=1[osd1];"
        "[p0][osd0]overlay=x=0:y=0:format=auto:shortest=1:repeatlast=0[p1];"
        "[p1][osd1]overlay=x=40:y=40:format=auto:shortest=1:repeatlast=0,format=pix_fmts=gray[v]"
    )


@pytest.mark.parametrize(
    "overrides", [{"input_label": "0:v];movie"}, {"output_label": "v[x]"}, {"pix_fmt": "gray,movie=x"}]
)
def test_graph_labels_and_pix_fmt_must_be_plain_tokens(overrides: dict) -> None:
    with pytest.raises(CctvChainError) as error:
        build_osd_graph((), [(0, 0, 4, 4)], FrameGeometry(64, 48), **overrides)

    assert error.value.code == INVALID_PARAM


def ffmpeg_path() -> Path:
    return Settings().ffmpeg_binary_path


def run_ffmpeg(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run([str(ffmpeg_path()), "-hide_banner", "-v", "error", *args], capture_output=True)


def make_clip(tmp_path: Path) -> Path:
    clip = tmp_path / "clip.mkv"
    width, height = CLIP_SIZE
    source = f"testsrc2=size={width}x{height}:rate=5"
    frames = str(CLIP_FRAMES)
    encode = ["-pix_fmt", "yuv420p", "-c:v", "ffv1"]
    result = run_ffmpeg("-y", "-f", "lavfi", "-i", source, "-frames:v", frames, *encode, str(clip))
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return clip


def gray_frames(result: subprocess.CompletedProcess[bytes], geometry: FrameGeometry) -> np.ndarray:
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    frame_size = geometry.width * geometry.height
    assert len(result.stdout) % frame_size == 0
    return np.frombuffer(result.stdout, np.uint8).reshape(-1, geometry.height, geometry.width)


def decode_gray(clip: Path, graph_option: str, graph: str, *maps: str) -> subprocess.CompletedProcess[bytes]:
    return run_ffmpeg("-i", str(clip), graph_option, graph, *maps, "-f", "rawvideo", "-pix_fmt", "gray", "-")


def probe_stream(path: Path) -> dict:
    ffprobe = Settings().ffprobe_binary_path
    entries = "stream=width,height,sample_aspect_ratio,nb_read_frames"
    command = [str(ffprobe), "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", entries]
    result = subprocess.run([*command, "-of", "json", str(path)], capture_output=True, check=True)
    return json.loads(result.stdout)["streams"][0]


@needs_ffmpeg
@pytest.mark.parametrize(("step_id", "filter_name"), classic_filter_cases())
def test_every_classic_filter_runs_in_the_real_binary_with_its_defaults(
    step_id: str, filter_name: str, tmp_path: Path
) -> None:
    step = default_step(step_id, filter_name)
    expected = output_dims_after((step,), *CLIP_SIZE)

    frames = gray_frames(decode_gray(make_clip(tmp_path), *vf_args((step,))), expected)

    kept = 4 if step_id == "trim" else CLIP_FRAMES
    assert frames.shape == (kept, expected.height, expected.width)


@needs_ffmpeg
def test_a_full_chain_in_one_vf_matches_output_dims_after(tmp_path: Path) -> None:
    steps = classic(
        {"id": "trim", "params": {"start_frame": 1, "end_frame": 6}},
        {"id": "aspect", "params": {"num": 2, "den": 1}},
        {"id": "deinterlace"},
        {"id": "deblock"},
        {"id": "denoise"},
        {"id": "crop", "params": {"w": 30, "h": 20, "x": 5, "y": 3}},
        {"id": "gray"},
        {"id": "levels", "params": {"gamma": 1.2}},
        {"id": "scale", "params": {"factor": 2}},
        {"id": "sharpen"},
    )
    output = tmp_path / "out.mkv"

    result = run_ffmpeg("-i", str(make_clip(tmp_path)), *vf_args(steps), "-c:v", "ffv1", str(output))

    assert result.returncode == 0, result.stderr.decode(errors="replace")
    expected = output_dims_after(steps, *CLIP_SIZE)
    stream = probe_stream(output)
    assert (stream["width"], stream["height"]) == (expected.width, expected.height) == (60, 40)
    assert Fraction(stream["sample_aspect_ratio"].replace(":", "/")) == expected.sar == Fraction(2)
    assert int(stream["nb_read_frames"]) == 6


def geometry_only_frames(clip: Path, steps: tuple[ResolvedStep, ...], geometry: FrameGeometry) -> np.ndarray:
    geometry_steps = tuple(step for step in steps if step.id in {"deinterlace", "crop", "gray", "scale"})
    return gray_frames(decode_gray(clip, *vf_args(geometry_steps)), geometry)


def box_region(frames: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return frames[:, y : y + h, x : x + w]


def outside_boxes(frames: np.ndarray, boxes: tuple[tuple[int, int, int, int], ...]) -> np.ndarray:
    mask = np.ones(frames.shape[1:], bool)
    for x, y, w, h in boxes:
        mask[y : y + h, x : x + w] = False
    return frames[:, mask]


@needs_ffmpeg
@pytest.mark.parametrize(
    ("raw_steps", "boxes", "pix_fmt"),
    [
        (
            [
                {"id": "levels", "params": {"brightness": 0.3}},
                {"id": "crop", "params": {"w": 40, "h": 30, "x": 3, "y": 5}},
                {"id": "scale", "params": {"factor": 2}},
            ],
            [(5, 6, 9, 5)],
            None,
        ),
        (
            [
                {"id": "deinterlace"},
                {"id": "denoise", "params": {"filter": "tmedian", "radius": 2}},
                {"id": "gray"},
                {"id": "levels", "params": {"brightness": -0.3}},
            ],
            [(0, 0, 16, 8), (41, 37, 11, 9)],
            "gray",
        ),
    ],
)
def test_the_real_osd_graph_puts_back_unprocessed_boxes(
    raw_steps: list[dict], boxes: list, pix_fmt: str | None, tmp_path: Path
) -> None:
    clip = make_clip(tmp_path)
    steps = classic(*raw_steps, {"id": "osd_protect"})
    source = FrameGeometry(*CLIP_SIZE)
    geometry = output_dims_after(steps, *CLIP_SIZE)
    placed = osd_boxes_after(steps, boxes, source)
    graph = build_osd_graph(steps, boxes, source, pix_fmt=pix_fmt)

    processed = gray_frames(decode_gray(clip, "-filter_complex", graph, "-map", "[v]"), geometry)
    reference = geometry_only_frames(clip, steps, geometry)

    assert processed.shape == reference.shape == (CLIP_FRAMES, geometry.height, geometry.width)
    for box in placed:
        assert np.array_equal(box_region(processed, box), box_region(reference, box))
    assert not np.array_equal(outside_boxes(processed, placed), outside_boxes(reference, placed))


def write_marker_png(directory: Path) -> Path:
    directory.mkdir(parents=True)
    png = directory / "marker.png"
    result = run_ffmpeg("-y", "-f", "lavfi", "-i", "color=c=white:size=8x8", "-frames:v", "1", str(png))
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return png


def overlay_movie(escaped_path: str, graph_option: str) -> subprocess.CompletedProcess[bytes]:
    main = "[0:v]" if graph_option == "-filter_complex" else "[in]"
    graph = f"movie={escaped_path}[marker];{main}[marker]overlay"
    source = ["-f", "lavfi", "-i", "color=c=black:size=8x8:duration=0.04"]
    return run_ffmpeg(*source, graph_option, graph, "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-")


@needs_ffmpeg
@pytest.mark.parametrize("graph_option", ["-vf", "-filter_complex"])
def test_escape_filter_path_reaches_the_real_binary_as_one_path(graph_option: str, tmp_path: Path) -> None:
    png = write_marker_png(tmp_path / TRICKY_DIR)
    assert ":" in str(png) and " " in str(png)

    result = overlay_movie(escape_filter_path(png), graph_option)

    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert set(result.stdout) == {255}


@needs_ffmpeg
def test_a_single_escape_level_is_not_enough_for_argv(tmp_path: Path) -> None:
    png = write_marker_png(tmp_path / "plain dir")
    single_level = str(png).replace("\\", "/").replace(":", "\\:")

    result = overlay_movie(single_level, "-filter_complex")

    assert result.returncode != 0


def centered_medians(values: list[int], radius: int) -> list[int]:
    padded = [values[0]] * radius + values + [values[-1]] * radius
    return [int(np.median(padded[index : index + 2 * radius + 1])) for index in range(len(values))]


@needs_ffmpeg
@pytest.mark.parametrize("radius", [1, 2])
def test_tmedian_keeps_every_frame_and_centres_its_window(radius: int) -> None:
    values = [10, 80, 20, 90, 30, 100, 40, 110, 50, 120]
    raw = np.concatenate([np.full((8, 8), value, np.uint8) for value in values]).tobytes()
    step = one("denoise", filter="tmedian", radius=radius)

    source = ["-f", "rawvideo", "-pix_fmt", "gray", "-s", "8x8", "-r", "5", "-i", "-"]
    output = ["-f", "rawvideo", "-pix_fmt", "gray", "-"]
    command = [str(ffmpeg_path()), "-v", "error", *source, *vf_args((step,)), *output]
    result = subprocess.run(command, input=raw, capture_output=True)

    frames = gray_frames(result, FrameGeometry(8, 8))
    assert [int(frame[0, 0]) for frame in frames] == centered_medians(values, radius)
