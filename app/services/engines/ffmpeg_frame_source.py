from __future__ import annotations

import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from app.services.frame_pipeline import drain_stream

_RGB24_BYTES_PER_PIXEL = 3
_STDERR_TAIL_CHARS = 500


class FfmpegFrameSource:
    """Decodea el video a frames rgb24 crudos por pipe (espejo del writer
    raw-pipe de salida ya probado en producción): cero PNGs intermedios y cero
    dependencias nuevas. Un thread del pipeline itera frames(); en cancel o
    error el proceso ffmpeg se mata SIEMPRE (nunca queda huérfano)."""

    def __init__(
        self,
        ffmpeg_binary: Path,
        source_path: Path,
        width: int,
        height: int,
        decode_threads: int,
        fps: str,
    ) -> None:
        self._ffmpeg_binary = ffmpeg_binary
        self._source_path = source_path
        self._width = width
        self._height = height
        self._decode_threads = decode_threads
        self._fps = fps
        self._frame_bytes = width * height * _RGB24_BYTES_PER_PIXEL

    def build_command(self) -> list[str]:
        # -fps_mode cfr (nunca -vsync, deprecado) normaliza la cadencia a tasa
        # CONSTANTE. Con passthrough se conservaban los timestamps VFR de la
        # fuente pero el encode los ignoraba y asumía fps fijo, así que el video
        # salía más corto que el audio y los subtítulos: en un BDrip real con
        # cadencia mixta 23.976/29.97 eran 177 s de deriva en 25 minutos.
        # Normalizando acá, frames == duración × fps por construcción y todo lo
        # de aguas abajo (conteo esperado, fps de encode) deja de mentir.
        # -threads va ANTES de -i: después de -i configuraría el encoder rawvideo
        # de salida en vez del decoder, dejando decode_threads sin efecto real.
        return [
            str(self._ffmpeg_binary),
            "-v", "error",
            "-threads", str(self._decode_threads),
            "-i", str(self._source_path),
            "-fps_mode", "cfr",
            "-r", self._fps,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ]

    def frames(self, cancel_event: threading.Event) -> Iterator[np.ndarray]:
        proc = self._spawn(self.build_command())
        stderr_buf: list[bytes] = []
        stderr_thread = threading.Thread(
            target=drain_stream, args=(proc.stderr, stderr_buf), daemon=True
        )
        stderr_thread.start()
        try:
            while not cancel_event.is_set():
                chunk = self._read_exact(proc.stdout, self._frame_bytes)
                if chunk is None:
                    break  # EOF limpio en un límite de frame
                yield self._to_frame(chunk)
            if cancel_event.is_set():
                return
            returncode = proc.wait()
            if returncode != 0:
                # join antes de leer el buffer: si se arma el mensaje mientras el
                # drenaje sigue corriendo, el tail sale vacío o cortado.
                stderr_thread.join(timeout=5)
                tail = b"".join(stderr_buf).decode("utf-8", errors="ignore").strip()
                raise RuntimeError(
                    f"ffmpeg falló al decodificar (código de salida {returncode}): "
                    f"{tail[-_STDERR_TAIL_CHARS:]}"
                )
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            stderr_thread.join(timeout=5)

    def _spawn(self, command: list[str]) -> subprocess.Popen:
        # Seam monkeypatcheable: los unit tests lo reemplazan por un proceso
        # fake (nunca corre el binario real en unit).
        return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _read_exact(self, stream, size: int) -> bytes | None:
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            data = stream.read(remaining)
            if not data:
                if chunks:
                    raise RuntimeError(
                        f"ffmpeg decode truncado: frame incompleto de {size - remaining} bytes"
                    )
                return None
            chunks.append(data)
            remaining -= len(data)
        return b"".join(chunks)

    def _to_frame(self, chunk: bytes) -> np.ndarray:
        frame = np.frombuffer(chunk, dtype=np.uint8).reshape(self._height, self._width, 3)
        # copy(): frombuffer devuelve un array read-only; aguas abajo GMFSS/ONNX
        # esperan poder pedir contigüidad/escritura sin sorpresas.
        return frame[np.newaxis, ...].copy()
