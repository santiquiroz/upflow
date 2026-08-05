from __future__ import annotations

from pathlib import Path

import pytest

from app.services.dub_mux import build_dub_mux_command

# ---------------------------------------------------------------------------
# El doblaje NO reemplaza el audio original: lo suma como pista y se pone
# primero para que el reproductor lo elija solo. Tirar el original seria perder
# lo unico que no se puede reconstruir si el doblaje no gusto.
#
# Valen los invariantes de ffmpeg que ya cobro caro este repo: todos los `-i`
# antes de cualquier `-map`, y el video mapeado explicito o desaparece.
# ---------------------------------------------------------------------------


def command(**kwargs: object) -> list[str]:
    argumentos: dict = {
        "ffmpeg": "ffmpeg.exe",
        "video": Path("origen.mp4"),
        "dubbed_audio": Path("doblaje.wav"),
        "destination": Path("salida.mkv"),
        "language": "es",
    }
    argumentos.update(kwargs)
    return build_dub_mux_command(**argumentos)  # type: ignore[arg-type]


class TestDubMuxCommand:
    def test_every_input_comes_before_any_map(self) -> None:
        comando = command()

        ultimo_input = max(i for i, arg in enumerate(comando) if arg == "-i")
        primer_map = min(i for i, arg in enumerate(comando) if arg == "-map")
        assert ultimo_input < primer_map

    def test_the_video_is_mapped_explicitly_or_it_vanishes(self) -> None:
        assert "0:v:0" in command()

    def test_the_video_is_copied_because_only_the_audio_changed(self) -> None:
        comando = command()

        assert comando[comando.index("-c:v") + 1] == "copy"

    def test_the_dubbed_track_comes_first_so_players_pick_it(self) -> None:
        comando = command()
        mapeos = [comando[i + 1] for i, arg in enumerate(comando) if arg == "-map"]

        assert mapeos.index("1:a:0") < mapeos.index("0:a?")

    def test_the_original_audio_is_kept_as_a_second_track(self) -> None:
        assert "0:a?" in command()

    def test_the_dubbed_track_is_marked_default(self) -> None:
        # Sin esto, algunos reproductores igual arrancan con la pista original.
        comando = " ".join(command())

        assert "-disposition:a:0" in comando
        assert "default" in comando

    def test_the_dubbed_track_carries_its_language(self) -> None:
        comando = " ".join(command(language="es"))

        assert "language=es" in comando

    def test_without_a_language_no_empty_tag_is_written(self) -> None:
        comando = " ".join(command(language=None))

        assert "language=" not in comando

    def test_overwrites_without_asking_or_the_job_hangs(self) -> None:
        assert "-y" in command()

    def test_rejects_a_destination_equal_to_the_source(self) -> None:
        with pytest.raises(ValueError):
            command(destination=Path("origen.mp4"))
