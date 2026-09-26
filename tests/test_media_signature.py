from __future__ import annotations

from pathlib import Path

import pytest

from app.services.media_signature import (
    CONTAINER_GUESSES,
    SNIFF_SIZE,
    ContainerGuess,
    container_guess,
    sniff_container,
    sniff_file,
)

IMKH_HEADER = b"IMKH" + bytes(36) + b"\x00\x00\x01\xba" + bytes(20)
DHAV_HEADER = b"DHAV\xfd\x00\x00\x00" + bytes(40)
MPEG_PS_HEADER = b"\x00\x00\x01\xba\x44\x00\x04\x00\x04\x01" + bytes(20)
MP4_HEADER = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00" + bytes(20)
MOV_HEADER = b"\x00\x00\x00\x14ftypqt  " + bytes(20)
AVI_HEADER = b"RIFF\x24\x10\x00\x00AVI LIST" + bytes(20)
WAV_HEADER = b"RIFF\x24\x10\x00\x00WAVEfmt " + bytes(20)
MKV_HEADER = b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01" + bytes(20)
H264_SPS_4 = b"\x00\x00\x00\x01\x67\x64\x00\x28" + bytes(20)
H264_AUD_3 = b"\x00\x00\x01\x09\xf0" + bytes(20)
HEVC_VPS_4 = b"\x00\x00\x00\x01\x40\x01\x0c\x01" + bytes(20)
HEVC_SPS_3 = b"\x00\x00\x01\x42\x01\x01" + bytes(20)


@pytest.mark.parametrize(
    ("header", "kind", "label"),
    [
        (IMKH_HEADER, "hikvision_ps", "Hikvision PS"),
        (DHAV_HEADER, "dahua_dav", "Dahua DAV"),
        (MPEG_PS_HEADER, "mpeg_ps", "MPEG-PS"),
        (MP4_HEADER, "mp4", "MP4/MOV"),
        (MOV_HEADER, "mp4", "MP4/MOV"),
        (AVI_HEADER, "avi", "AVI"),
        (MKV_HEADER, "matroska", "Matroska"),
        (H264_SPS_4, "raw_h264", "Raw H.264"),
        (H264_AUD_3, "raw_h264", "Raw H.264"),
        (HEVC_VPS_4, "raw_hevc", "Raw H.265"),
        (HEVC_SPS_3, "raw_hevc", "Raw H.265"),
    ],
)
def test_sniff_container_identifies_each_signature(header: bytes, kind: str, label: str) -> None:
    guess = sniff_container(header)

    assert guess.kind == kind
    assert guess.label == label


@pytest.mark.parametrize(
    "header",
    [
        b"",
        b"IMK",
        b"\x89PNG\r\n\x1a\n" + bytes(20),
        WAV_HEADER,
        b"\x00\x00\x00\x01\xba" + bytes(20),
        b"\x00\x00\x00\x01\x00" + bytes(20),
        b"\x00\x00\x01\xff" + bytes(20),
        b"ftyp" + bytes(20),
    ],
)
def test_sniff_container_returns_unknown_for_unrecognised_bytes(header: bytes) -> None:
    guess = sniff_container(header)

    assert guess.kind == "unknown"
    assert guess.label == "Unknown"
    assert guess.ffmpeg_format is None
    assert guess.suggests_cctv is False


@pytest.mark.parametrize(
    ("header", "ffmpeg_format"),
    [
        (IMKH_HEADER, "mpeg"),
        (DHAV_HEADER, "dhav"),
        (MPEG_PS_HEADER, "mpeg"),
        (H264_SPS_4, "h264"),
        (HEVC_VPS_4, "hevc"),
        (MP4_HEADER, None),
        (AVI_HEADER, None),
        (MKV_HEADER, None),
    ],
)
def test_sniff_container_names_the_demuxer_the_remux_retries_force(header: bytes, ffmpeg_format: str | None) -> None:
    assert sniff_container(header).ffmpeg_format == ffmpeg_format


@pytest.mark.parametrize(
    ("header", "suggests"),
    [
        (IMKH_HEADER, True),
        (DHAV_HEADER, True),
        (H264_SPS_4, True),
        (HEVC_VPS_4, True),
        (MPEG_PS_HEADER, False),
        (MP4_HEADER, False),
        (AVI_HEADER, False),
        (MKV_HEADER, False),
    ],
)
def test_sniff_container_flags_the_dvr_exports_that_suggest_cctv_mode(header: bytes, suggests: bool) -> None:
    assert sniff_container(header).suggests_cctv is suggests


def test_sniff_file_ignores_the_extension(tmp_path: Path) -> None:
    disguised = tmp_path / "export.mp4"
    disguised.write_bytes(IMKH_HEADER + bytes(4096))

    assert sniff_file(disguised).kind == "hikvision_ps"


def test_sniff_file_reads_only_the_first_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clip = tmp_path / "clip.dav"
    clip.write_bytes(DHAV_HEADER + bytes(1_000_000))
    requested: list[int] = []
    real_open = Path.open

    def recording_open(self: Path, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        real_read = handle.read

        def recording_read(size: int = -1) -> bytes:
            requested.append(size)
            return real_read(size)

        handle.read = recording_read
        return handle

    monkeypatch.setattr(Path, "open", recording_open)

    assert sniff_file(clip).kind == "dahua_dav"
    assert requested == [SNIFF_SIZE]


def test_sniff_file_of_an_empty_file_is_unknown(tmp_path: Path) -> None:
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")

    assert sniff_file(empty).kind == "unknown"


def test_sniff_guesses_round_trip_by_kind() -> None:
    for guess in CONTAINER_GUESSES:
        assert container_guess(guess.kind) == guess


def test_sniff_guess_lookup_rejects_an_unknown_kind() -> None:
    with pytest.raises(ValueError):
        container_guess("quicktime-ish")


def test_sniff_guess_is_immutable() -> None:
    guess = sniff_container(IMKH_HEADER)

    with pytest.raises(AttributeError):
        guess.kind = "mp4"  # type: ignore[misc]
    assert isinstance(guess, ContainerGuess)
