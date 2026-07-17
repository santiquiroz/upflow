from __future__ import annotations

# ---------------------------------------------------------------------------
# Video encoder selection (SP12 audit): expose hardware encoders (AMF/NVENC/QSV)
# as an opt-in alongside the software x264/x265 default. HW encode collapses the
# final 4K encode stage (30-45% of wall-time by software) to near-zero GPU time.
#
# Pure functions of their inputs (device name string + codec) so the routing is
# unit-testable without a GPU or ffmpeg. The vendor is detected from the GPU's
# display NAME (already exposed by DevicesService) instead of the DXGI VendorId,
# which avoids deep ctypes plumbing for equivalent accuracy -- GPU names reliably
# carry the vendor.
# ---------------------------------------------------------------------------

VIDEO_ENCODER_SOFTWARE = "software"
VIDEO_ENCODER_AUTO = "auto"
VIDEO_ENCODERS = frozenset({VIDEO_ENCODER_SOFTWARE, VIDEO_ENCODER_AUTO})

# Software encoders (the codec the user picked maps straight to these).
_SOFTWARE_ENCODERS = frozenset({"libx264", "libx265"})

# Codec family per software codec: HW encoders are named per family (h264/hevc).
_CODEC_FAMILY = {"libx264": "h264", "libx265": "h265"}

# GPU-name substring -> vendor key. Checked case-insensitively, first match wins.
_NAME_VENDOR_SIGNATURES = (
    ("nvidia", ("nvidia", "geforce", "rtx", "gtx", "quadro", "tesla")),
    ("amd", ("amd", "radeon", "rx ")),
    ("intel", ("intel", "arc", "iris", "uhd graphics", "hd graphics")),
)

# vendor -> {family -> hardware encoder}
_VENDOR_ENCODERS = {
    "nvidia": {"h264": "h264_nvenc", "h265": "hevc_nvenc"},
    "amd": {"h264": "h264_amf", "h265": "hevc_amf"},
    "intel": {"h264": "h264_qsv", "h265": "hevc_qsv"},
}


def codec_family(video_codec: str) -> str:
    return _CODEC_FAMILY.get(video_codec, "h264")


def vendor_from_device_name(device_name: str | None) -> str | None:
    if not device_name:
        return None
    lowered = device_name.lower()
    for vendor, signatures in _NAME_VENDOR_SIGNATURES:
        if any(sig in lowered for sig in signatures):
            return vendor
    return None


def is_hardware_encoder(encoder: str) -> bool:
    return encoder not in _SOFTWARE_ENCODERS


def resolve_hardware_encoder(device_name: str | None, video_codec: str) -> str | None:
    """The HW encoder for this GPU + codec family, or None if the vendor can't be
    mapped (then the caller falls back to the software codec)."""
    vendor = vendor_from_device_name(device_name)
    if vendor is None:
        return None
    return _VENDOR_ENCODERS[vendor].get(codec_family(video_codec))


def encode_options(
    *,
    encoder: str,
    crf: int,
    preset: str,
    x265_pools: int,
    software_threads: int,
) -> list[str]:
    """FFmpeg -c:v ... quality flags for the chosen encoder. HW encoders reject
    -crf and the x264/x265 -preset names, so each family gets its own knobs; the
    crf value is reused as the family's equivalent quality target."""
    if encoder == "libx264":
        return [
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-threads", str(software_threads),
        ]
    if encoder == "libx265":
        return [
            "-c:v", "libx265", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-x265-params", f"frame-threads=4:pools={x265_pools}",
            "-threads", str(min(x265_pools, 8)),
        ]
    if encoder in ("h264_nvenc", "hevc_nvenc"):
        # NVENC: constant-quality VBR; -cq is the quality target (lower = better).
        return ["-c:v", encoder, "-preset", "p5", "-rc", "vbr", "-cq", str(crf), "-pix_fmt", "yuv420p"]
    if encoder in ("h264_amf", "hevc_amf"):
        # AMF: constant QP; qp_i/qp_p are the quality target.
        return [
            "-c:v", encoder, "-quality", "quality", "-rc", "cqp",
            "-qp_i", str(crf), "-qp_p", str(crf), "-pix_fmt", "yuv420p",
        ]
    if encoder in ("h264_qsv", "hevc_qsv"):
        return ["-c:v", encoder, "-global_quality", str(crf), "-pix_fmt", "yuv420p"]
    raise ValueError(f"unknown encoder {encoder!r}")
