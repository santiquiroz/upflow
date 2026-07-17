from __future__ import annotations

import pytest

from app.services import video_encoders as ve


@pytest.mark.parametrize(
    "name, expected",
    [
        ("NVIDIA GeForce RTX 4090", "nvidia"),
        ("AMD Radeon RX 7800 XT", "amd"),
        ("AMD Radeon(TM) Graphics", "amd"),
        ("Intel Arc A770", "intel"),
        ("Intel(R) UHD Graphics 770", "intel"),
        ("Some Unknown Accelerator", None),
        (None, None),
        ("", None),
    ],
)
def test_vendor_from_device_name(name, expected) -> None:
    assert ve.vendor_from_device_name(name) == expected


@pytest.mark.parametrize(
    "name, codec, expected",
    [
        ("NVIDIA GeForce RTX 4090", "libx264", "h264_nvenc"),
        ("NVIDIA GeForce RTX 4090", "libx265", "hevc_nvenc"),
        ("AMD Radeon RX 7800 XT", "libx264", "h264_amf"),
        ("AMD Radeon RX 7800 XT", "libx265", "hevc_amf"),
        ("Intel Arc A770", "libx265", "hevc_qsv"),
        ("Mystery GPU", "libx265", None),
        (None, "libx265", None),
    ],
)
def test_resolve_hardware_encoder(name, codec, expected) -> None:
    assert ve.resolve_hardware_encoder(name, codec) == expected


def test_is_hardware_encoder() -> None:
    assert ve.is_hardware_encoder("h264_nvenc") is True
    assert ve.is_hardware_encoder("hevc_amf") is True
    assert ve.is_hardware_encoder("libx264") is False
    assert ve.is_hardware_encoder("libx265") is False


def test_encode_options_software_libx264_matches_legacy() -> None:
    opts = ve.encode_options(encoder="libx264", crf=20, preset="medium", x265_pools=8, software_threads=24)
    assert opts == ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "24"]


def test_encode_options_software_libx265_matches_legacy() -> None:
    opts = ve.encode_options(encoder="libx265", crf=18, preset="slow", x265_pools=8, software_threads=24)
    assert opts == [
        "-c:v", "libx265", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
        "-x265-params", "frame-threads=4:pools=8", "-threads", "8",
    ]


def test_encode_options_nvenc_uses_cq_not_crf() -> None:
    opts = ve.encode_options(encoder="hevc_nvenc", crf=20, preset="slow", x265_pools=8, software_threads=24)
    assert "-crf" not in opts
    assert "-cq" in opts and opts[opts.index("-cq") + 1] == "20"
    assert opts[:2] == ["-c:v", "hevc_nvenc"]


def test_encode_options_amf_uses_qp_not_crf() -> None:
    opts = ve.encode_options(encoder="h264_amf", crf=22, preset="medium", x265_pools=8, software_threads=24)
    assert "-crf" not in opts
    assert opts[opts.index("-qp_i") + 1] == "22"
    assert opts[opts.index("-qp_p") + 1] == "22"


def test_encode_options_qsv_uses_global_quality() -> None:
    opts = ve.encode_options(encoder="hevc_qsv", crf=19, preset="medium", x265_pools=8, software_threads=24)
    assert "-crf" not in opts
    assert opts[opts.index("-global_quality") + 1] == "19"


def test_encode_options_rejects_unknown_encoder() -> None:
    with pytest.raises(ValueError, match="unknown encoder"):
        ve.encode_options(encoder="bogus", crf=20, preset="medium", x265_pools=8, software_threads=24)
