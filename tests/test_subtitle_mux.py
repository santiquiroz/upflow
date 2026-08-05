from __future__ import annotations

from pathlib import Path

import pytest

from app.services.subtitle_mux import (
    SUBTITLE_CODEC_BY_CONTAINER,
    build_subtitle_mux_command,
    subtitle_codec_for,
)

# ---------------------------------------------------------------------------
# Muxeo SUAVE: la pista de subtitulos se agrega al contenedor sin re-encodear el
# video. Es instantaneo y sin perdida, a diferencia de quemarlos en la imagen,
# que obliga a recodificar el video entero.
#
# Los invariantes de ffmpeg que ya cobro caro este repo (ver
# test_audio_subtitle_mux.py) se respetan igual: TODOS los `-i` antes de
# cualquier `-map`, y `-map 0:v:0` explicito o el video se cae en silencio.
# ---------------------------------------------------------------------------


def command(container: str = "mkv") -> list[str]:
    return build_subtitle_mux_command(
        ffmpeg="ffmpeg.exe",
        video=Path("origen.mp4"),
        subtitles=Path("subs.srt"),
        destination=Path(f"salida.{container}"),
        container=container,
        language="es",
    )


class TestSubtitleCodec:
    def test_mkv_carries_subrip_as_is(self) -> None:
        assert subtitle_codec_for("mkv") == "srt"

    def test_mp4_needs_mov_text_because_it_cannot_hold_subrip(self) -> None:
        # Un mp4 con `-c:s srt` falla: el contenedor no soporta ese codec.
        assert subtitle_codec_for("mp4") == "mov_text"

    def test_an_unknown_container_falls_back_to_the_one_mkv_takes(self) -> None:
        assert subtitle_codec_for("avi") == "srt"

    def test_the_table_covers_the_containers_the_app_offers(self) -> None:
        assert {"mp4", "mkv"} <= set(SUBTITLE_CODEC_BY_CONTAINER)


class TestMuxCommand:
    def test_takes_both_inputs_before_any_map(self) -> None:
        cmd = command()
        first_map = cmd.index("-map")
        inputs = [i for i, part in enumerate(cmd) if part == "-i"]
        assert inputs, "sin -i no hay nada que muxear"
        assert max(inputs) < first_map

    def test_maps_the_video_explicitly_so_it_is_not_dropped(self) -> None:
        cmd = command()
        assert "0:v:0" in cmd

    def test_brings_the_original_audio_along(self) -> None:
        assert "0:a?" in command()

    def test_takes_the_subtitles_from_the_second_input(self) -> None:
        assert "1:0" in command()

    def test_copies_the_video_instead_of_re_encoding_it(self) -> None:
        cmd = command()
        assert cmd[cmd.index("-c:v") + 1] == "copy"
        assert cmd[cmd.index("-c:a") + 1] == "copy"

    def test_uses_the_codec_the_container_accepts(self) -> None:
        assert command("mp4")[command("mp4").index("-c:s") + 1] == "mov_text"
        assert command("mkv")[command("mkv").index("-c:s") + 1] == "srt"

    def test_tags_the_language_so_players_name_the_track(self) -> None:
        assert "language=es" in command()

    def test_overwrites_so_a_retry_does_not_hang_on_a_prompt(self) -> None:
        assert "-y" in command()

    def test_writes_to_the_destination_last(self) -> None:
        assert command()[-1] == "salida.mkv"

    def test_without_a_language_it_does_not_invent_one(self) -> None:
        cmd = build_subtitle_mux_command(
            ffmpeg="ffmpeg.exe",
            video=Path("origen.mp4"),
            subtitles=Path("subs.srt"),
            destination=Path("salida.mkv"),
            container="mkv",
            language=None,
        )
        assert not any(part.startswith("language=") for part in cmd)


class TestRefusals:
    @pytest.mark.parametrize("container", ["", "   "])
    def test_an_empty_container_is_refused(self, container: str) -> None:
        with pytest.raises(ValueError):
            build_subtitle_mux_command(
                ffmpeg="ffmpeg.exe",
                video=Path("a.mp4"),
                subtitles=Path("s.srt"),
                destination=Path("o.mkv"),
                container=container,
                language=None,
            )
