"""Suma la pista doblada al video, sin re-encodear la imagen.

El doblaje NO reemplaza el audio original: se suma como pista y queda primera
para que el reproductor la elija sola. Tirar el original seria perder lo unico
que no se puede reconstruir si el doblaje no gusto.

Los invariantes de ffmpeg que ya cobro caro este repo valen igual: TODOS los
`-i` antes de cualquier `-map`, y el video mapeado explicito o desaparece sin
aviso.
"""

from __future__ import annotations

from pathlib import Path

# aac entra en cualquier contenedor y lo lee cualquier reproductor. El wav de la
# sintesis pesa demasiado para viajar crudo adentro del video.
DUB_AUDIO_CODEC = "aac"


def build_dub_mux_command(
    *,
    ffmpeg: str,
    video: Path,
    dubbed_audio: Path,
    destination: Path,
    language: str | None,
) -> list[str]:
    if Path(destination) == Path(video):
        raise ValueError("El destino no puede ser el mismo archivo de origen.")

    command = [
        str(ffmpeg),
        # Sin `-y` ffmpeg espera confirmacion y el job cuelga.
        "-y",
        "-i",
        str(video),
        "-i",
        str(dubbed_audio),
        "-map",
        "0:v:0",
        # La doblada primero: el orden de los -map es el orden de las pistas.
        "-map",
        "1:a:0",
        "-map",
        "0:a?",
        "-c:v",
        "copy",
        "-c:a",
        DUB_AUDIO_CODEC,
        # Sin esto, algunos reproductores arrancan igual con la pista original.
        "-disposition:a:0",
        "default",
        "-disposition:a:1",
        "0",
    ]
    if language:
        command += ["-metadata:s:a:0", f"language={language}"]
    command.append(str(destination))
    return command
