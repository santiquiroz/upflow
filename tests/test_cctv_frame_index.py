from __future__ import annotations

import pytest

from app.services.cctv_frame_index import FrameEntry, frame_index_csv, parse_frame_index, summarize_frame_index


def _frames(pts: list[float | None], types: str = "", sizes: list[int] | None = None) -> str:
    types = types or "I" + "P" * (len(pts) - 1)
    sizes = sizes or [1000] * len(pts)
    lines = []
    for t, kind, size in zip(pts, types, sizes):
        stamp = "N/A" if t is None else f"{t:.6f}"
        key = 1 if kind == "I" else 0
        trailing = "," if key else ""
        lines.append(f"key_frame={key},best_effort_timestamp_time={stamp},pkt_size={size},pict_type={kind}{trailing}")
    return "\n".join(lines) + "\n"


def _entries(pts: list[float | None], types: str = "", sizes: list[int] | None = None):
    return parse_frame_index(_frames(pts, types, sizes))


def _cfr(count: int, fps: float = 25.0) -> list[float]:
    return [n / fps for n in range(count)]


def test_frame_index_parses_keyframe_trailing_commas_and_missing_values() -> None:
    text = (
        "key_frame=1,best_effort_timestamp_time=0.000000,pkt_size=4918,pict_type=I,\n"
        "key_frame=0,best_effort_timestamp_time=N/A,pkt_size=N/A,pict_type=?\n"
        "\n"
    )

    frames = parse_frame_index(text)

    assert frames[0] == FrameEntry(n=0, pts_time=0.0, key_frame=True, pict_type="I", pkt_size=4918)
    assert frames[1] == FrameEntry(n=1, pts_time=None, key_frame=False, pict_type="?", pkt_size=None)


def test_frame_index_csv_has_a_header_and_one_row_per_frame() -> None:
    csv_text = frame_index_csv(_entries([0.0, None]))

    assert csv_text == "n,pts_time,key_frame,pict_type,pkt_size\n0,0.000000,1,I,1000\n1,,0,P,1000\n"


def test_frame_index_measures_fps_from_the_median_pts_delta() -> None:
    summary = summarize_frame_index(_entries(_cfr(50, 15.0)))

    assert summary.frame_count == 50
    assert summary.measured_fps == pytest.approx(15.0, rel=1e-4)
    assert not summary.is_vfr and summary.gaps == ()


def test_frame_index_ignores_missing_timestamps_for_the_deltas() -> None:
    pts: list[float | None] = list(_cfr(10))
    pts[4] = None

    summary = summarize_frame_index(_entries(pts))

    assert summary.frame_count == 10
    assert summary.measured_fps == pytest.approx(25.0)


def test_frame_index_classifies_jittered_deltas_as_vfr() -> None:
    pts = [n * 0.04 + (0.012 if n % 4 == 1 else 0.0) for n in range(40)]

    assert summarize_frame_index(_entries(pts)).is_vfr


def test_frame_index_keeps_millisecond_rounding_of_30_fps_as_cfr() -> None:
    pts = [round(n / 30, 3) for n in range(90)]

    summary = summarize_frame_index(_entries(pts))

    assert not summary.is_vfr
    assert summary.measured_fps == pytest.approx(30.0, rel=0.05)


def test_frame_index_reports_gaps_longer_than_one_and_a_half_deltas() -> None:
    pts = _cfr(10) + [n / 25 + 0.4 for n in range(10, 20)]

    summary = summarize_frame_index(_entries(pts))

    assert len(summary.gaps) == 1
    gap = summary.gaps[0]
    assert gap.after_frame == 9 and gap.start == pytest.approx(0.36) and gap.end == pytest.approx(0.8)
    assert not summary.is_vfr


def test_frame_index_counts_repeated_minimum_p_frames_as_probable_duplicates() -> None:
    sizes = [5000, 1200, 18, 18, 1300, 18, 1250]

    summary = summarize_frame_index(_entries(_cfr(7), "IPPPPPP", sizes))

    assert summary.probable_duplicates == 3


def test_frame_index_does_not_count_a_single_smallest_p_frame() -> None:
    summary = summarize_frame_index(_entries(_cfr(4), "IPPP", [5000, 900, 1200, 1100]))

    assert summary.probable_duplicates == 0


def test_frame_index_describes_the_gop() -> None:
    types = "IBBPBBP" + "IBBPBBPBBP" + "IP"

    gop = summarize_frame_index(_entries(_cfr(len(types)), types)).gop

    assert gop.keyframes == 3
    assert (gop.min_length, gop.max_length, gop.median_length) == (7, 10, 8.5)
    assert gop.p_ratio == pytest.approx(6 / 19)
    assert gop.b_ratio == pytest.approx(10 / 19)


def test_frame_index_of_an_empty_stream_has_no_fps() -> None:
    summary = summarize_frame_index(())

    assert summary.frame_count == 0 and summary.measured_fps is None
    assert summary.gop.keyframes == 0 and summary.gop.max_length is None


def test_frame_index_summary_json_uses_camel_case() -> None:
    payload = summarize_frame_index(_entries(_cfr(3))).to_json()

    assert payload["frameCount"] == 3 and payload["measuredFps"] == pytest.approx(25.0)
    assert set(payload) == {"frameCount", "measuredFps", "medianDelta", "isVfr", "gaps", "probableDuplicates", "gop"}
