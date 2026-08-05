from __future__ import annotations

import pytest

from app.services.subtitles import (
    TranscriptSegment,
    merge_chunk_segments,
    segments_to_srt,
    segments_to_vtt,
    srt_timestamp,
    vtt_timestamp,
)

# ---------------------------------------------------------------------------
# Whisper devuelve los tiempos POR CHUNK, y cada chunk arranca de cero. Como el
# audio se parte en trozos de 30 s, el segundo chunk que dice (0.0, 4.2) en
# realidad es (30.0, 34.2). Sin ese corrimiento todos los subtitulos del archivo
# quedarian encimados sobre el primer medio minuto.
# ---------------------------------------------------------------------------


def seg(start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(start=start, end=end, text=text)


class TestSrtTimestamp:
    def test_uses_the_comma_that_srt_requires(self) -> None:
        # SRT usa coma para los milisegundos; WebVTT usa punto. Confundirlos deja
        # un archivo que el reproductor ignora en silencio.
        assert srt_timestamp(0) == "00:00:00,000"

    def test_keeps_three_digits_of_milliseconds(self) -> None:
        assert srt_timestamp(5.56) == "00:00:05,560"

    def test_carries_into_minutes_and_hours(self) -> None:
        assert srt_timestamp(3661.5) == "01:01:01,500"

    def test_never_emits_a_negative_time(self) -> None:
        assert srt_timestamp(-1) == "00:00:00,000"


class TestVttTimestamp:
    def test_uses_the_dot_that_webvtt_requires(self) -> None:
        assert vtt_timestamp(5.56) == "00:00:05.560"


class TestMergeChunkSegments:
    def test_shifts_each_chunk_by_its_place_in_the_audio(self) -> None:
        merged = merge_chunk_segments(
            [[seg(0.0, 5.5, "primero")], [seg(0.0, 4.2, "segundo")]],
            chunk_seconds=30,
        )
        assert [(s.start, s.end, s.text) for s in merged] == [
            (0.0, 5.5, "primero"),
            (30.0, 34.2, "segundo"),
        ]

    def test_drops_empty_segments_instead_of_writing_blank_cues(self) -> None:
        merged = merge_chunk_segments([[seg(0.0, 1.0, "  "), seg(1.0, 2.0, "hola")]], chunk_seconds=30)
        assert [s.text for s in merged] == ["hola"]

    def test_an_empty_transcription_produces_no_segments(self) -> None:
        assert merge_chunk_segments([[], []], chunk_seconds=30) == []


class TestSegmentsToSrt:
    def test_numbers_cues_from_one(self) -> None:
        body = segments_to_srt([seg(0.0, 1.0, "uno"), seg(1.0, 2.0, "dos")])
        assert body.startswith("1\n")
        assert "\n2\n" in body

    def test_writes_the_arrow_the_format_expects(self) -> None:
        body = segments_to_srt([seg(0.0, 5.56, "hola")])
        assert "00:00:00,000 --> 00:00:05,560" in body

    def test_ends_with_a_blank_line_so_players_accept_the_last_cue(self) -> None:
        assert segments_to_srt([seg(0.0, 1.0, "uno")]).endswith("\n\n")

    def test_no_segments_gives_an_empty_file_not_a_broken_one(self) -> None:
        assert segments_to_srt([]) == ""


class TestSegmentsToVtt:
    def test_starts_with_the_required_header(self) -> None:
        # Sin la cabecera WEBVTT el archivo no es valido y el reproductor lo ignora.
        assert segments_to_vtt([seg(0.0, 1.0, "uno")]).startswith("WEBVTT\n\n")

    def test_still_writes_the_header_when_there_is_nothing_to_say(self) -> None:
        assert segments_to_vtt([]) == "WEBVTT\n\n"


class TestSegmentText:
    def test_a_segment_keeps_its_text_trimmed(self) -> None:
        assert seg(0, 1, "  hola  ").text == "hola"

    @pytest.mark.parametrize("bad", [(2.0, 1.0)])
    def test_an_end_before_its_start_is_refused(self, bad: tuple[float, float]) -> None:
        """Un cue invertido rompe el reproductor, y salir con un archivo roto es
        peor que fallar al generarlo."""
        with pytest.raises(ValueError):
            TranscriptSegment(start=bad[0], end=bad[1], text="x")


# ---------------------------------------------------------------------------
# El procesador de Whisper devuelve los offsets como
# [{"text": ..., "timestamp": (inicio, fin)}]. Un chunk que termina justo en el
# borde de los 30 s puede traer `None` como fin: el modelo no llego a cerrar el
# segmento. Descartarlo perderia texto, asi que se cierra en el borde del chunk.
# ---------------------------------------------------------------------------


class TestSegmentsFromOffsets:
    def test_reads_the_shape_the_processor_returns(self) -> None:
        from app.services.subtitles import segments_from_offsets

        segments = segments_from_offsets(
            [
                {"text": " He hoped", "timestamp": (0.0, 5.56)},
                {"text": " mutton pieces", "timestamp": (5.56, 10.16)},
            ],
            chunk_seconds=30,
        )
        assert [(s.start, s.end, s.text) for s in segments] == [
            (0.0, 5.56, "He hoped"),
            (5.56, 10.16, "mutton pieces"),
        ]

    def test_closes_an_unterminated_segment_at_the_chunk_edge(self) -> None:
        from app.services.subtitles import segments_from_offsets

        segments = segments_from_offsets(
            [{"text": "cortado", "timestamp": (28.0, None)}], chunk_seconds=30
        )
        assert (segments[0].start, segments[0].end) == (28.0, 30.0)

    def test_ignores_an_entry_with_no_usable_start(self) -> None:
        from app.services.subtitles import segments_from_offsets

        assert segments_from_offsets([{"text": "x", "timestamp": (None, 3.0)}], chunk_seconds=30) == []


# ---------------------------------------------------------------------------
# La descarga puede pedir la transcripcion o el archivo de subtitulos. El nombre
# y el tipo MIME tienen que corresponder al formato pedido: un .srt servido como
# text/plain lo abre el navegador en vez de bajarlo.
# ---------------------------------------------------------------------------


class TestSubtitleFormats:
    def test_knows_the_three_deliverables(self) -> None:
        from app.services.subtitles import SUBTITLE_FORMATS

        assert set(SUBTITLE_FORMATS) == {"txt", "srt", "vtt"}

    def test_each_format_carries_its_extension_and_media_type(self) -> None:
        from app.services.subtitles import SUBTITLE_FORMATS

        assert SUBTITLE_FORMATS["srt"].extension == ".srt"
        assert SUBTITLE_FORMATS["srt"].media_type == "application/x-subrip"
        assert SUBTITLE_FORMATS["vtt"].media_type == "text/vtt"
        assert SUBTITLE_FORMATS["txt"].extension == ".txt"

    def test_renders_each_format_from_the_same_segments(self) -> None:
        from app.services.subtitles import render_segments

        segments = [seg(0.0, 1.5, "hola")]
        assert "-->" in render_segments(segments, "srt")
        assert render_segments(segments, "vtt").startswith("WEBVTT")
        assert render_segments(segments, "txt") == "hola"

    def test_an_unknown_format_is_refused_instead_of_guessed(self) -> None:
        from app.services.subtitles import render_segments

        with pytest.raises(ValueError, match="doc"):
            render_segments([], "doc")
