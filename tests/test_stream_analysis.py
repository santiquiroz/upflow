from __future__ import annotations

from app.services.stream_analysis import parse_audio_tracks, parse_subtitle_tracks

FAKE_PROBE = {
    "streams": [
        {"index": 0, "codec_type": "video", "codec_name": "h264"},
        {
            "index": 1,
            "codec_type": "audio",
            "codec_name": "ac3",
            "channels": 2,
            "disposition": {"default": 1},
            "tags": {"language": "jpn"},
        },
        {
            "index": 2,
            "codec_type": "audio",
            "codec_name": "aac",
            "channels": 6,
            "disposition": {"default": 0},
            "tags": {"language": "eng"},
        },
        {
            "index": 3,
            "codec_type": "subtitle",
            "codec_name": "ass",
            "disposition": {"default": 0},
            "tags": {"language": "eng"},
        },
    ]
}


def test_parse_audio_tracks_returns_one_entry_per_audio_stream_in_order():
    tracks = parse_audio_tracks(FAKE_PROBE)
    assert [t.index for t in tracks] == [1, 2]


def test_parse_audio_tracks_reads_language_channels_and_default_flag():
    tracks = parse_audio_tracks(FAKE_PROBE)
    assert tracks[0].language == "jpn"
    assert tracks[0].channels == 2
    assert tracks[0].is_default is True
    assert tracks[1].language == "eng"
    assert tracks[1].channels == 6
    assert tracks[1].is_default is False


def test_parse_audio_tracks_missing_language_tag_is_none():
    probe = {"streams": [{"index": 0, "codec_type": "audio", "codec_name": "aac", "channels": 2, "disposition": {}}]}
    tracks = parse_audio_tracks(probe)
    assert tracks[0].language is None


def test_parse_audio_tracks_no_audio_streams_returns_empty_list():
    probe = {"streams": [{"index": 0, "codec_type": "video", "codec_name": "h264"}]}
    assert parse_audio_tracks(probe) == []


def test_parse_subtitle_tracks_returns_one_entry_per_subtitle_stream():
    tracks = parse_subtitle_tracks(FAKE_PROBE)
    assert len(tracks) == 1
    assert tracks[0].index == 3
    assert tracks[0].language == "eng"
    assert tracks[0].codec == "ass"


def test_parse_subtitle_tracks_no_subtitle_streams_returns_empty_list():
    assert parse_subtitle_tracks({"streams": []}) == []
