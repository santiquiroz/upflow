from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable

import numpy as np

from app.services.frame_pipeline import drain_stream


def _default_summarize(stderr: bytes) -> str:
    text = stderr.decode("utf-8", errors="ignore")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "ffmpeg encode failed"


class RawPipeEncoder:
    """Proceso ffmpeg de encode alimentado por stdin con frames rgb24 crudos.

    Extraído de VideoUpscaler._upscale_encode_streaming para que el raw-pipe
    clásico y el stream pipeline compartan el MISMO writer (spec: extraído/
    reusado, no duplicado). write_frame bloquea con el backpressure natural del
    pipe — por eso siempre se llama desde un worker thread, nunca del loop.
    """

    def __init__(
        self, command: list[str], summarize_error: Callable[[bytes], str] | None = None
    ) -> None:
        self._command = command
        self._summarize = summarize_error or _default_summarize
        self._proc: subprocess.Popen | None = None
        self._stderr_buf: list[bytes] = []
        self._stderr_thread: threading.Thread | None = None
        self.frames_written = 0

    def start(self) -> None:
        self._proc = self._spawn(self._command)
        self._stderr_thread = threading.Thread(
            target=drain_stream, args=(self._proc.stderr, self._stderr_buf), daemon=True
        )
        self._stderr_thread.start()

    def _spawn(self, command: list[str]) -> subprocess.Popen:
        # Seam monkeypatcheable: los unit tests lo reemplazan por un proceso fake.
        return subprocess.Popen(
            command, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL
        )

    def write_frame(self, frame_hwc: np.ndarray) -> None:
        assert self._proc is not None, "write_frame antes de start()"
        self._proc.stdin.write(frame_hwc.tobytes())
        self.frames_written += 1

    def finish(self) -> None:
        assert self._proc is not None, "finish antes de start()"
        self._proc.stdin.close()
        returncode = self._proc.wait()
        self._join_stderr()
        if returncode != 0:
            raise RuntimeError(self._summarize(b"".join(self._stderr_buf)))

    def kill(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait()
        self._join_stderr()

    def _join_stderr(self) -> None:
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=5)
