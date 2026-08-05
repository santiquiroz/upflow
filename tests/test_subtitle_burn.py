from __future__ import annotations

from pathlib import Path

import pytest

from app.services.subtitle_burn import (
    build_subtitle_burn_command,
    escape_subtitles_filter_path,
)

# ---------------------------------------------------------------------------
# Quemar subtitulos = re-encodear el video con el texto pintado encima. Sirve
# para reproductores que no muestran pistas de texto (redes sociales, tele
# vieja), a cambio de tiempo y de una perdida de calidad que el muxeo suave no
# tiene.
#
# El punto delicado es la RUTA: el nombre del archivo viaja DENTRO de la cadena
# del filtro, donde `:` separa opciones, `\` escapa y `'` delimita. Una ruta de
# Windows tal cual (`C:\subs\a.srt`) rompe el parser de ffmpeg.
# ---------------------------------------------------------------------------


class TestFilterPathEscaping:
    def test_drive_colon_is_escaped_because_it_separates_filter_options(self) -> None:
        salida = escape_subtitles_filter_path(Path(r"C:\salidas\job.srt"))

        assert r"C\:" in salida
        assert ":" not in salida.replace(r"\:", "")

    def test_backslashes_become_forward_slashes(self) -> None:
        salida = escape_subtitles_filter_path(Path(r"C:\salidas\job.srt"))

        assert "\\salidas" not in salida
        assert "/salidas/job.srt" in salida

    def test_single_quotes_are_escaped_so_they_do_not_close_the_string(self) -> None:
        salida = escape_subtitles_filter_path(Path("/videos/o'brien.srt"))

        assert r"o\'brien" in salida


class TestBurnCommand:
    def command(self, **kwargs: object) -> list[str]:
        argumentos: dict = {
            "ffmpeg": "ffmpeg.exe",
            "video": Path("origen.mp4"),
            "subtitles": Path("subs.srt"),
            "destination": Path("salida.mp4"),
        }
        argumentos.update(kwargs)
        return build_subtitle_burn_command(**argumentos)  # type: ignore[arg-type]

    def test_overwrites_without_asking_or_the_job_hangs(self) -> None:
        assert "-y" in self.command()

    def test_the_subtitle_file_goes_inside_the_filter_not_as_an_input(self) -> None:
        comando = self.command()

        filtro = comando[comando.index("-vf") + 1]
        assert filtro.startswith("subtitles=")
        # Un segundo `-i` seria muxeo suave, no quemado.
        assert comando.count("-i") == 1

    def test_audio_is_copied_because_only_the_picture_changes(self) -> None:
        comando = self.command()

        assert comando[comando.index("-c:a") + 1] == "copy"

    def test_video_cannot_be_copied_because_the_picture_is_being_repainted(self) -> None:
        comando = self.command()

        assert "-c:v" not in comando or comando[comando.index("-c:v") + 1] != "copy"

    def test_the_destination_is_the_last_argument(self) -> None:
        assert self.command()[-1] == "salida.mp4"

    def test_a_quality_knob_is_passed_so_the_re_encode_is_not_left_to_chance(self) -> None:
        comando = self.command()

        assert "-crf" in comando

    def test_rejects_a_destination_equal_to_the_source(self) -> None:
        # ffmpeg leeria y escribiria el mismo archivo y lo dejaria corrupto.
        with pytest.raises(ValueError):
            self.command(destination=Path("origen.mp4"))
